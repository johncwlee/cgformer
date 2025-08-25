import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class LocalAggregator(BaseModule):
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
    ):
        super().__init__()
        self.local_encoder_backbone = MODELS.build(local_encoder_backbone)
        self.local_encoder_neck = MODELS.build(local_encoder_neck)
    
    def forward(self, x):
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)
        output = output[0]

        return output