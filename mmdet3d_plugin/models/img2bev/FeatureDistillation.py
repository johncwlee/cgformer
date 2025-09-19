from typing import Callable, List, Optional

import os
import logging
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


logger = logging.getLogger(__name__)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@MODELS.register_module()
class FeatureDistillationHead(nn.Module):
    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        point_cloud_range,
        embed_dims,
        semantic_encoder,
        feat_idx,
        occ_size,
        class_frequencies,
        num_samples,
        loss_weight,
        hard_samples=None,
        **kwargs
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        
        self.data_config = data_config
        self.point_cloud_range = point_cloud_range
        self.num_samples = num_samples
        self.hard_samples = hard_samples
        if self.hard_samples is None:
            self.hard_samples = self.num_samples
        elif self.hard_samples > self.num_samples:
            self.hard_samples = self.num_samples
        elif self.hard_samples < 0:
            logger.warning(f"Hard samples is less than 0, setting to `{self.num_samples}`")
            self.hard_samples = self.num_samples
        self.semantic_encoder = MODELS.build(semantic_encoder)
        self.proj_head = Mlp(*embed_dims)
        self.feat_idx = feat_idx
        self.occ_size = occ_size
        
        class_freqs = torch.as_tensor(class_frequencies, dtype=torch.float32)
        alpha = 1.0
        inv_freq = (1.0 / (class_freqs + 1e-12)).pow(alpha)
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=False)
        self.loss_weight = loss_weight

    def loss(self, voxel_feats, img, depth, lss_encoder,cam_params, img_metas=None, gt_occ=None, **kwargs):
        """ Loss function.
        Args:
            voxel_feats (tuple[Tensor]): Voxel features from the upstream
                network, each is a 5D-tensor with shape
                (B | N, C, H_v, W_v, Z_v).
            img: Input image, (B | N, 3, H, W)
            depth: Depth distribution, (B | N, D, H_f, W_f)
            lss_encoder: LSS encoder, used to compute the LSS semantic features
            img_metas: Meta information
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
            gt_occ: Ground truth occupancy, (B | N, H, W, Z)
        """
        B, N, _, H_img, W_img = img.shape
        BN, C, H_v, W_v, Z_v = voxel_feats.shape
        
        #? Get semantic features
        sem_feats = self.semantic_encoder(img.view(B*N, 3, H_img, W_img))[self.feat_idx]  #* (B * N, C, H_f, W_f)
        sem_feats = sem_feats.view(B, N, -1, sem_feats.shape[-2], sem_feats.shape[-1])  #* (B, N, C, H_f, W_f)
        H_f, W_f = sem_feats.shape[-2], sem_feats.shape[-1]
        sem_feats = sem_feats.flatten(-2).permute(0, 1, 3, 2)   #* (B, N, H_f * W_f, C)
        sem_feats_proj = self.proj_head(sem_feats).permute(0, 1, 3, 2)  #* (B, N, C, H_f * W_f)
        sem_feats_proj = sem_feats_proj.view(B, N, -1, H_f, W_f)  #* (B, N, C, H_f, W_f)
        
        #? LSS semantic features
        lss_sem_feats = lss_encoder(sem_feats_proj, depth, cam_params)  #* (B | N, C, H_v, W_v, Z_v)
        
        #? Compute cosine similarity
        gt_occ_downsampled = F.interpolate(gt_occ.unsqueeze(1).float(), 
                                           size=self.occ_size, 
                                           mode='nearest-exact').long().contiguous()
        labels = gt_occ_downsampled.squeeze(1)

        #? Inverse-frequency soft-balanced sampling of voxel locations
        # labels: (B|N, H_v, W_v, Z_v)

        sampled_coords = []
        for b in range(BN):
            lbl = labels[b].reshape(-1).long()  # (H_v*W_v*Z_v,)
            #* Ignore unlabeled voxels
            valid_mask = (lbl != 255) & (lbl > 0) & (lbl < self.inv_freq.numel())
            if valid_mask.any():
                weights = torch.zeros_like(lbl, dtype=torch.float32)
                weights[valid_mask] = self.inv_freq[lbl[valid_mask]]
                wsum = weights.sum()
                if wsum > 0:
                    weights = weights / wsum
                    sel = torch.multinomial(weights, num_samples=self.num_samples, replacement=False)
                else:
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                    sel = valid_idx[torch.randint(0, valid_idx.numel(), (self.num_samples,), device=lbl.device)]
            else:
                sel = torch.randint(0, lbl.numel(), (self.num_samples,), device=lbl.device)

            #? unravel flat indices to (x, y, z) in (H_v, W_v, Z_v)
            plane = W_v * Z_v
            x = sel // plane
            rem = sel % plane
            y = rem // Z_v
            z = rem % Z_v
            coords = torch.stack([x, y, z], dim=1)  # (num_samples, 3)

            sampled_coords.append(coords)

        sampled_coords = torch.stack(sampled_coords, dim=0)  # (BN, num_samples, 3)

        #? Sample voxel features
        S = sampled_coords.shape[1]
        x = sampled_coords[..., 0]  # (BN, S)
        y = sampled_coords[..., 1]
        z = sampled_coords[..., 2]
        flat_idx = x * (W_v * Z_v) + y * Z_v + z  # (BN, S)

        #? Flatten spatial dims then gather along last dim
        teacher_flat = lss_sem_feats.reshape(BN, -1, H_v * W_v * Z_v)
        student_flat = voxel_feats.reshape(BN, -1, H_v * W_v * Z_v)
        sample_idx = flat_idx.long().unsqueeze(1).expand(BN, teacher_flat.shape[1], S)
        sampled_teacher_feats = teacher_flat.gather(dim=2, index=sample_idx)  # (BN, C, S)
        sampled_student_feats = student_flat.gather(dim=2, index=sample_idx)  # (BN, C, S)
        
        #? Compute cosine similarity (feature alignment)
        teacher_norm = F.normalize(sampled_teacher_feats, p=2, dim=1, eps=1e-6)  # (BN, C, S)
        student_norm = F.normalize(sampled_student_feats, p=2, dim=1, eps=1e-6)  # (BN, C, S)
        cos_sim = (teacher_norm * student_norm).sum(dim=1)  # (BN, S)
        
        #? Select hard samples (lowest cosine similarity per item)
        hard_sim, _ = torch.topk(cos_sim, k=self.hard_samples, dim=1, largest=False)
        feature_align_loss = 1.0 - hard_sim.mean()

        return {'loss_occ_feat_align': feature_align_loss * self.loss_weight}