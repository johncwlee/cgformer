from lightning.pytorch import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from mmdet3d.registry import DATASETS

class DataModule(LightningDataModule):
    def __init__(
        self,
        config      
    ):
        super().__init__()
        self.trainset_config = config.data.train
        self.testset_config = config.data.test
        self.valset_config = config.data.val

        self.train_dataloader_config = config.train_dataloader_config
        self.test_dataloader_config = config.test_dataloader_config
        self.val_dataloader_config = config.test_dataloader_config
        self.config = config
    
    def prepare_data(self):
        # Intentionally left blank; data is prepared in setup()
        # This avoids Lightning's is_overridden parent resolution issue.
        pass

    def setup(self, stage=None):
        self.train_dataset = DATASETS.build(self.trainset_config)
        self.test_dataset = DATASETS.build(self.testset_config)
        self.val_dataset = DATASETS.build(self.valset_config)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            drop_last=True,
            num_workers=self.train_dataloader_config.num_workers,
            shuffle=True,
            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.val_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.test_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True)