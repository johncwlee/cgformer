import torch
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES

from .learning_map import learning_map

@PIPELINES.register_module()
class MapDenseAnnotations:
    """
    Transform dense 2D semantic labels using a pre-defined mapping.

    Args:
        dataset (str): The dataset name.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        assert self.dataset in ['kitti360', 'ssckitti360'], \
            f"dataset must be one of ['kitti360', 'ssckitti360'], but got {self.dataset}"
        self.learning_map = learning_map[self.dataset]
    
    def __call__(self, results):
        assert 'gt_semantics' in results, "gt_semantics is not in results"
        seg_gt = results['gt_semantics']
        seg_gt = np.vectorize(self.learning_map.__getitem__)(seg_gt)
        results['gt_semantics'] = torch.from_numpy(seg_gt)
        
        return results

