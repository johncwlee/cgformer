import os
import torch
import numpy as np
from PIL import Image

from .basemodel import LightningBaseModel
from .metric import SSCMetrics, SemanticSegmentationMetrics, FOVSSCMetrics
from mmdet3d.registry import MODELS
from .utils import (
    get_inv_map, create_colored_segmentation_map, get_fov_mask,
    SEMANTIC_KITTI_COLORS
)
from mmengine.runner import load_checkpoint



class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = MODELS.build(model_config)
        if 'load_from' in config:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']

        self.train_metrics = SSCMetrics(config['num_class']) if not self.pretrain \
                            else SemanticSegmentationMetrics(self.num_class)
        self.val_metrics = SSCMetrics(config['num_class']) if not self.pretrain \
                            else SemanticSegmentationMetrics(self.num_class)
        self.test_metrics = SSCMetrics(config['num_class']) if not self.pretrain \
                            else SemanticSegmentationMetrics(self.num_class)
        
        #* FOV SSC Metrics
        if not self.pretrain:
            self.train_metrics_fov = FOVSSCMetrics(config['num_class'])
            self.val_metrics_fov = FOVSSCMetrics(config['num_class'])
            self.test_metrics_fov = FOVSSCMetrics(config['num_class'])
    
    def forward(self, data_dict):
        return self.model(data_dict)
    
    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True)
        
        pred = output_dict['pred'].detach().cpu().numpy()
        
        if not self.pretrain:
            gt = output_dict['gt_occ'].detach().cpu().numpy()
        else:
            gt = output_dict['gt_semantics'].detach().cpu().numpy()
        
        self.train_metrics.add_batch(pred, gt)
        if not self.pretrain:
            #? Compute FOV masks
            fov_masks = self.get_fov_masks(batch)
            self.train_metrics_fov.add_batch(pred, gt, fov_masks)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        output_dict = self.forward(batch)
        
        pred = output_dict['pred'].detach().cpu().numpy()
        
        if not self.pretrain:
            gt = output_dict['gt_occ'].detach().cpu().numpy()
        else:
            gt = output_dict['gt_semantics'].detach().cpu().numpy()
        
        self.val_metrics.add_batch(pred, gt)
        if not self.pretrain:
            #? Compute FOV masks
            fov_masks = self.get_fov_masks(batch)
            self.val_metrics_fov.add_batch(pred, gt, fov_masks)
    
    def on_validation_epoch_end(self):
        metrics_list = [("train", self.train_metrics), ("val", self.val_metrics),
                        ("train_fov", self.train_metrics_fov), ("val_fov", self.val_metrics_fov)]
        
        metric_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else self.device

        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            if not self.pretrain:
                self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=metric_device), sync_dist=True)
                #? Add IoU for each class
                for name, iou in zip(self.class_names[1:], stats['iou_ssc'][1:]):
                    self.log("{}/IoU_{}".format(prefix, name), torch.tensor(iou, dtype=torch.float32, device=metric_device), sync_dist=True)
                self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32, device=metric_device), sync_dist=True)
                if "precision" in stats:
                    self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32, device=metric_device), sync_dist=True)
                if "recall" in stats:
                    self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32, device=metric_device), sync_dist=True)
            else:
                self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_mean"], dtype=torch.float32, device=metric_device), sync_dist=True)
                #? Add IoU for each class
                for name, iou in zip(self.class_names[1:], stats['iou_class'][1:]):
                    self.log("{}/{}".format(prefix, name), torch.tensor(iou, dtype=torch.float32, device=metric_device), sync_dist=True)
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        projected_gt = output_dict['projected_gt'].detach().cpu().numpy() \
            if 'projected_gt' in output_dict else None
        
        if not self.pretrain:
            gt_occ = output_dict['gt_occ']
            if gt_occ is not None:
                gt_occ = gt_occ.detach().cpu().numpy()
            else:
                gt_occ = None
        else:
            gt_occ = batch['gt_semantics'].detach().cpu().numpy()

        if self.save_path is not None:
            if not self.pretrain:
                if self.test_mapping:
                    inv_map = get_inv_map()
                    output_voxels = inv_map[pred].astype(np.uint16)
                else:
                    output_voxels = pred.astype(np.uint16)
                sequence_id = batch['img_metas']['sequence'][0]
                frame_id = batch['img_metas']['frame_id'][0]
                save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
                save_file = os.path.join(save_folder, "{}.label".format(frame_id))
                gt_save_folder = "{}/sequences/{}/gt".format(self.save_path, sequence_id)
                gt_save_file = os.path.join(gt_save_folder, "{}.label".format(frame_id))
                os.makedirs(save_folder, exist_ok=True)
                os.makedirs(gt_save_folder, exist_ok=True)
                with open(save_file, 'wb') as f:
                    output_voxels.tofile(f)
                    print('\n save to {}'.format(save_file))
                    gt_occ.astype(np.uint16).tofile(gt_save_file)
                    print('\n save gt to {}'.format(gt_save_file))
                
                if projected_gt is not None:
                    projected_gt_save_folder = "{}/sequences/{}/projected_gt".format(self.save_path, sequence_id)
                    os.makedirs(projected_gt_save_folder, exist_ok=True)
                    projected_gt_map = create_colored_segmentation_map(projected_gt, SEMANTIC_KITTI_COLORS)
                    for i in range(projected_gt.shape[0]):
                        projected_gt_save_file = os.path.join(projected_gt_save_folder, "{}.png".format(frame_id))
                        projected_gt_img = Image.fromarray(projected_gt_map[i], mode='RGBA')
                        projected_gt_img.save(projected_gt_save_file)
                        print('\n save projected gt to {}'.format(projected_gt_save_file))
            else:
                sequence_id = batch['img_metas']['sequence'][0]
                frame_id = batch['img_metas']['frame_id']
                save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
                gt_save_folder = "{}/sequences/{}/gt".format(self.save_path, sequence_id)
                
                os.makedirs(save_folder, exist_ok=True)
                os.makedirs(gt_save_folder, exist_ok=True)
                
                pred_labels = np.argmax(pred, axis=1)
                #TODO: change colors depending on the dataset
                seg_maps = create_colored_segmentation_map(pred_labels, SEMANTIC_KITTI_COLORS)
                gt_seg_maps = create_colored_segmentation_map(gt_occ.squeeze(1), SEMANTIC_KITTI_COLORS)
                for i in range(seg_maps.shape[0]):
                    frame_id_i = frame_id[i]
                    save_file = os.path.join(save_folder, "{}.png".format(frame_id_i))
                    gt_save_file = os.path.join(gt_save_folder, "{}.png".format(frame_id_i))
                    pred_img = Image.fromarray(seg_maps[i], mode='RGBA')
                    gt_img = Image.fromarray(gt_seg_maps[i], mode='RGBA')
                    pred_img.save(save_file)
                    gt_img.save(gt_save_file)
                    
            
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
            if not self.pretrain:
                #? Compute FOV masks
                fov_masks = self.get_fov_masks(batch)
                self.test_metrics_fov.add_batch(pred, gt_occ, fov_masks)
    
    def on_test_epoch_end(self):
        metric_list = [("test", self.test_metrics), ("test_fov", self.test_metrics_fov)]
        
        metrics_list = metric_list
        metric_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else self.device
        for prefix, metric in metrics_list:
            stats = metric.get_stats()

            if not self.pretrain:
                for name, iou in zip(self.class_names, stats['iou_ssc']):
                    print(name + ":", iou)

                self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=metric_device), sync_dist=True)
                self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32, device=metric_device), sync_dist=True)
                if "precision" in stats:
                    self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32, device=metric_device), sync_dist=True)
                if "recall" in stats:
                    self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32, device=metric_device), sync_dist=True)
            else:
                #? Add IoU for each class
                for name, iou in zip(self.class_names[1:], stats['iou_class'][1:]):
                    self.log("{}/{}".format(prefix, name), torch.tensor(iou, dtype=torch.float32, device=metric_device), sync_dist=True)

                self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_mean"], dtype=torch.float32, device=metric_device), sync_dist=True)
            metric.reset()

    def get_fov_masks(self, batch):
        cam_params = batch['img_inputs'][1:7]
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        transform = torch.cat([rots, trans[..., None]], dim=-1)
        B, N, _, _ = transform.shape
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=transform.device).view(1, 1, 1, 4).expand(B, N, -1, -1)
        transform = torch.cat([transform, bottom_row], dim=-2)
        transform = transform.cpu().numpy()
        intrins = intrins.cpu().numpy()
        transform = np.linalg.inv(transform)

        grid_size = batch['img_metas']['occ_size'].tolist()[0]
        origin = batch['img_metas']['pc_range'].tolist()[0][:3]
        img_H, img_W = batch['img_metas']['img_shape']
        img_H, img_W = img_H.item(), img_W.item()
        img_size = [img_W, img_H]
        fov_masks_batch = []
        for b in range(B):
            #TODO: check for num_cam > 1
            fov_mask = get_fov_mask(transform[b, 0], 
                                    intrins[b, 0],
                                    grid_size=grid_size,
                                    origin=origin,
                                    img_size=img_size)
            fov_mask = fov_mask.reshape(*grid_size)
            fov_masks_batch.append(fov_mask)
        fov_masks_batch = np.stack(fov_masks_batch, axis=0)
        
        return fov_masks_batch