import os
import misc
import torch
from mmengine.config import Config
from mmdet3d_plugin import *
from lightning.pytorch import Trainer, seed_everything
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from lightning.pytorch import loggers as pl_loggers                  # loggers (TensorBoard, WandB, MLflow, etc.)
from lightning.pytorch.profilers import SimpleProfiler               # profiler API
from lightning.pytorch.strategies import DDPStrategy                 # distributed strategies
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor  # callbacks
from LightningTools.weight_averaging import EMAWeightAveraging


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--load', default=None)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--project_name', type=str,default='cgformer')
    parser.add_argument('--ema', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join(config['output_dir'], config['log_folder'])
    misc.check_path(log_folder)
    
    if config.wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project=config.project_name,
            name=config.log_folder,
            save_dir=log_folder,
            offline=config.offline,
        )

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )
    loggers = [tb_logger]
    if config.wandb:
        loggers.append(wandb_logger)
    
    if config.ema:
        ema_callback = EMAWeightAveraging(decay=config.ema_decay)

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpu}")
    if config.load:
        config.load_from = config.load
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_last=True,
        filename='best')
    
    if not config.eval:
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step')
        ]
        if config.ema:
            callbacks.append(ema_callback)

        trainer = Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            max_steps=config.training_steps,
            callbacks=callbacks,
            logger=loggers,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch']
        )
        trainer.fit(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])
    else:
        trainer = Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            logger=tb_logger,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])

    

# python3 main.py --config_path configs/CGFormer-DINOv3-SemanticKITTI.py --output_dir ../../results/cgformer --log_folder CGFormer-test --seed 7240 --log_every_n_steps 100
# python3 main.py --config_path configs/CGFormer-Efficient-Swin-SemanticKITTI.py --output_dir ../../results/cgformer --log_folder CGFormer-test --seed 7240 --log_every_n_steps 100