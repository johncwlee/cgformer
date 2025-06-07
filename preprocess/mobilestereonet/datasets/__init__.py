from .dataset import KITTIDataset, KITTI360Dataset, KITTI360PretrainDataset

__datasets__ = {
    "kitti": KITTIDataset,
    "kitti360": KITTI360Dataset,
    "kitti360_pretrain": KITTI360PretrainDataset,
}
