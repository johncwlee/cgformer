import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d_plugin.utils.sigmoid import inverse_sigmoid


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 3D coordinates to positional embeddings. 
    Follow PETR-style positional encoding.
    Args:
        pos: (N_query, 3)
        num_pos_feats:
        temperature:
    Returns:
        posemb: (N_query, num_feats * 3)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)     # (num_feats, )
    # Use floor division with specified rounding mode to avoid deprecation warning
    dim_t = torch.div(dim_t, 2, rounding_mode='floor')
    dim_t = temperature ** (2 * dim_t / num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
    pos_x = pos[..., 0, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
    pos_y = pos[..., 1, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
    pos_z = pos[..., 2, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_z/10000^(0/128), pos_z/10000^(0/128), pos_z/10000^(2/128), pos_z/10000^(2/128), ...]

    # (N_query, num_feats/2, 2) --> (N_query, num_feats)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_x/10000^(0/128)), cos(pos_x/10000^(0/128)), sin(pos_x/10000^(2/128)), cos(pos_x/10000^(2/128)), ...]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_y/10000^(0/128)), cos(pos_y/10000^(0/128)), sin(pos_y/10000^(2/128)), cos(pos_y/10000^(2/128)), ...]
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_z/10000^(0/128)), cos(pos_z/10000^(0/128)), sin(pos_z/10000^(2/128)), cos(pos_z/10000^(2/128)), ...]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # (N_query, num_feats * 3)
    return posemb


@MODELS.register_module()
class Learned3DPositionalEncoding(BaseModule):
    """3DPosition embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for positional embedding
        volume_h (int): The height of the 3D volume
        volume_w (int): The width of the 3D volume
        volume_z (int): The depth of the 3D volume
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_feats: int,
        volume_h: int,
        volume_w: int,
        volume_z: int,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_feats = num_feats
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        
        #? 3D positional encoder
        self.encoder = nn.Sequential(
            nn.Linear(128*3, num_feats),
            nn.ReLU(),
            nn.Linear(num_feats, num_feats),
        )
        
        self.init_weights()

    def forward(self, ref_3d: Tensor) -> Tensor:
        """Forward function for `Learned3DPositionalEncoding`.

        Args:
            ref_3d (Tensor): The 3D coordinates of the reference points.
                Shape [h*w*z, 3].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [h*w*z, num_feats].
        """
        pos_embed = inverse_sigmoid(ref_3d)
        pos_embed = self.encoder(pos2posemb3d(pos_embed))
        return pos_embed