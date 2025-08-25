import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class Fuser(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        global_aggregator=None,
        local_aggregator=None
    ):
        super().__init__()
        self.global_aggregator = MODELS.build(global_aggregator)
        self.local_aggregator = MODELS.build(local_aggregator)

        self.combine_coeff = nn.Sequential(
            nn.Conv3d(embed_dims, 4, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        #* Voxel branch
        local_feats = self.local_aggregator(x)  #* (bs, C, H, W, D)
        #* TPV branch
        global_feats = self.global_aggregator(x)
        #* [0]: (bs, C, H, W, 1)
        #* [1]: (bs, C, 1, W, D)
        #* [2]: (bs, C, H, 1, D)

        weights = self.combine_coeff(local_feats)

        out_feats = local_feats * weights[:, 0:1, ...] + global_feats[0] * weights[:, 1:2, ...] + \
            global_feats[1] * weights[:, 2:3, ...] + global_feats[2] * weights[:, 3:4, ...]

        return out_feats