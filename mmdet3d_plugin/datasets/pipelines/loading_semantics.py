import torch
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES

from .learning_map import learning_map

@PIPELINES.register_module()
class LoadDenseAnnotations:
    def __init__(self, dataset='kitti360', is_train=True):
        self.dataset = dataset
        assert self.dataset in ['kitti360', 'ssckitti360'], \
            f"dataset must be one of ['kitti360', 'ssckitti360'], but got {self.dataset}"
        self.is_train = is_train
        
        self.learning_map = learning_map['kitti360']
    
    def __call__(self, results):
        assert 'seg_gt_path' in results, "seg_gt_path is not in results"
        seg_gt_path = results['seg_gt_path']
        seg_gt = np.array(Image.open(seg_gt_path), dtype=np.uint8)
        seg_gt = np.vectorize(self.learning_map.__getitem__)(seg_gt)
        results['seg_gt'] = torch.from_numpy(seg_gt)
        
        return results

