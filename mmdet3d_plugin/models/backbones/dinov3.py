# Copyright (c) OpenMMLab. All rights reserved.

import logging
import math
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
try:
    from typing import Literal
except ImportError:  # Python 3.7 fallback
    from typing_extensions import Literal

import torch
import torch.nn.init
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


from .dinov3_layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from .dinov3_layers.utils import named_apply


logger = logging.getLogger(__name__)


ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


@MODELS.register_module()
class Dinov3(BaseModule):
    """"""

    def __init__(self,
                    *,
                    img_size: int = 224,
                    patch_size: int = 16,
                    in_chans: int = 3,
                    pos_embed_rope_base: float = 100.0,
                    pos_embed_rope_min_period: Optional[float] = None,
                    pos_embed_rope_max_period: Optional[float] = None,
                    pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
                    pos_embed_rope_shift_coords: Optional[float] = None,
                    pos_embed_rope_jitter_coords: Optional[float] = None,
                    pos_embed_rope_rescale_coords: Optional[float] = None,
                    pos_embed_rope_dtype: str = "bf16",
                    embed_dim: int = 768,
                    depth: int = 12,
                    num_heads: int = 12,
                    ffn_ratio: float = 4.0,
                    qkv_bias: bool = True,
                    drop_path_rate: float = 0.0,
                    layerscale_init: Optional[float] = None,
                    norm_layer: str = "layernorm",
                    ffn_layer: str = "mlp",
                    ffn_bias: bool = True,
                    proj_bias: bool = True,
                    n_storage_tokens: int = 0,
                    mask_k_bias: bool = False,
                    untie_cls_and_patch_norms: bool = False,
                    untie_global_and_local_cls_norm: bool = False,
                    out_indices: Union[int, Sequence[int]] = [2, 5, 8, 11],
                    init_cfg=[
                                dict(type='Kaiming', layer='Conv2d'),
                                dict(
                                    type='Constant',
                                    layer=['_BatchNorm', 'GroupNorm'],
                                    val=1)
                            ],
                    **ignored_kwargs):
        super(Dinov3, self).__init__(init_cfg)

        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.out_indices = out_indices

        ckpt_path = init_cfg['checkpoint']
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict, strict=True)

        # Freeze once at init and set eval
        self.requires_grad_(False)
        self.eval()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Union[Tensor, List[Tensor]], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, x):
        feats = self.get_intermediate_layers(x, 
                                            n=self.out_indices, 
                                            reshape=True,
                                            return_class_token=False)
        ms_feats: List[Tensor] = []
        ms_feats.append(F.interpolate(
            feats[-1], scale_factor=4, mode="bilinear", align_corners=False
        ))
        ms_feats.append(F.interpolate(
            feats[-1], scale_factor=2, mode="bilinear", align_corners=False
        ))
        ms_feats.append(feats[-1])
        ms_feats.append(F.interpolate(
            feats[-1], scale_factor=0.5, mode="bilinear", align_corners=False
        ))
        return ms_feats


    def train(self, mode=False):
        # Keep backbone in eval and frozen regardless of requested mode
        super(Dinov3, self).train(mode=False)
        return self


# # ---------------- ViT-Adapter for DINOv3 ---------------- #
# def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
#     if drop_prob == 0.0 or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor


# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x: Tensor) -> Tensor:
#         return drop_path(x, self.drop_prob, self.training)


# def _get_reference_points(spatial_shapes, device):
#     reference_points_list = []
#     for (H_, W_) in spatial_shapes:
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#             torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
#         )
#         ref_y = ref_y.reshape(-1)[None] / H_
#         ref_x = ref_x.reshape(-1)[None] / W_
#         ref = torch.stack((ref_x, ref_y), -1)
#         reference_points_list.append(ref)
#     reference_points = torch.cat(reference_points_list, 1)
#     reference_points = reference_points[:, :, None]
#     return reference_points


# def _deform_inputs(x: Tensor, patch_size: int):
#     bs, c, h, w = x.shape
#     spatial_shapes = torch.as_tensor(
#         [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device
#     )
#     level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#     reference_points = _get_reference_points([(h // patch_size, w // patch_size)], x.device)
#     deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

#     spatial_shapes2 = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
#     level_start_index2 = torch.cat((spatial_shapes2.new_zeros((1,)), spatial_shapes2.prod(1).cumsum(0)[:-1]))
#     reference_points2 = _get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
#     deform_inputs2 = [reference_points2, spatial_shapes2, level_start_index2]

#     return deform_inputs1, deform_inputs2


# def _ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
#     N_, S_, M_, D_ = value.shape
#     _, Lq_, M_, L_, P_, _ = sampling_locations.shape
#     value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#     sampling_grids = 2 * sampling_locations - 1
#     sampling_value_list = []
#     for lid_, (H_, W_) in enumerate(value_spatial_shapes):
#         value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
#         sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#         sampling_value_l_ = F.grid_sample(
#             value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
#         )
#         sampling_value_list.append(sampling_value_l_)
#     attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
#     output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
#     return output.transpose(1, 2).contiguous()


# class MSDeformAttn(nn.Module):
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError("d_model must be divisible by n_heads")
#         self.im2col_step = 64
#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points
#         self.ratio = ratio
#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.value_proj = nn.Linear(d_model, int(d_model * ratio))
#         self.output_proj = nn.Linear(int(d_model * ratio), d_model)
#         self._reset_parameters()

#     def _reset_parameters(self):
#         nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (
#             (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
#             .view(self.n_heads, 1, 1, 2)
#             .repeat(1, self.n_levels, self.n_points, 1)
#         )
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         nn.init.constant_(self.attention_weights.weight.data, 0.0)
#         nn.init.constant_(self.attention_weights.bias.data, 0.0)
#         nn.init.xavier_uniform_(self.value_proj.weight.data)
#         nn.init.constant_(self.value_proj.bias.data, 0.0)
#         nn.init.xavier_uniform_(self.output_proj.weight.data)
#         nn.init.constant_(self.output_proj.bias.data, 0.0)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape
#         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
#         value = self.value_proj(input_flatten)
#         if input_padding_mask is not None:
#             value = value.masked_fill(input_padding_mask[..., None], float(0))
#         value = value.view(N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
#         attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :]
#                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#             )
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = (
#                 reference_points[:, :, None, :, None, :2]
#                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#             )
#         else:
#             raise ValueError("Last dim of reference_points must be 2 or 4")
#         output = _ms_deform_attn_core_pytorch(
#             value,
#             input_spatial_shapes,
#             sampling_locations,
#             attention_weights,
#         )
#         output = self.output_proj(output)
#         return output


# class ConvFFN(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x: Tensor, H: int, W: int) -> Tensor:
#         x = self.fc1(x)
#         x = self.dwconv(x, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x: Tensor, H: int, W: int) -> Tensor:
#         B, N, C = x.shape
#         n = max(1, N // 21)
#         x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
#         x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
#         x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, max(1, H // 2), max(1, W // 2)).contiguous()
#         x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
#         x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
#         x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x


# class Extractor(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 6,
#         n_points: int = 4,
#         n_levels: int = 1,
#         deform_ratio: float = 1.0,
#         with_cffn: bool = True,
#         cffn_ratio: float = 0.25,
#         drop: float = 0.0,
#         drop_path: float = 0.0,
#         norm_layer=nn.LayerNorm,
#         with_cp: bool = False,
#     ):
#         super().__init__()
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
#         self.with_cffn = with_cffn
#         self.with_cp = with_cp
#         if with_cffn:
#             self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
#             self.ffn_norm = norm_layer(dim)
#             self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#     def forward(self, query: Tensor, reference_points: Tensor, feat: Tensor, spatial_shapes: Tensor, level_start_index: Tensor, H: int, W: int) -> Tensor:
#         def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
#             attn = self.attn(
#                 self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
#             )
#             query2 = query + attn
#             if self.with_cffn:
#                 query2 = query2 + self.drop_path(self.ffn(self.ffn_norm(query2), H, W))
#             return query2

#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
#         return query


# class InteractionBlockWithCls(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 6,
#         n_points: int = 4,
#         norm_layer=nn.LayerNorm,
#         drop: float = 0.0,
#         drop_path: float = 0.0,
#         with_cffn: bool = True,
#         cffn_ratio: float = 0.25,
#         init_values: float = 0.0,
#         deform_ratio: float = 1.0,
#         extra_extractor: bool = False,
#         with_cp: bool = False,
#     ):
#         super().__init__()
#         self.extractor = Extractor(
#             dim=dim,
#             n_levels=1,
#             num_heads=num_heads,
#             n_points=n_points,
#             norm_layer=norm_layer,
#             deform_ratio=deform_ratio,
#             with_cffn=with_cffn,
#             cffn_ratio=cffn_ratio,
#             drop=drop,
#             drop_path=drop_path,
#             with_cp=with_cp,
#         )
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(
#                 *[
#                     Extractor(
#                         dim=dim,
#                         num_heads=num_heads,
#                         n_points=n_points,
#                         norm_layer=norm_layer,
#                         with_cffn=with_cffn,
#                         cffn_ratio=cffn_ratio,
#                         deform_ratio=deform_ratio,
#                         drop=drop,
#                         drop_path=drop_path,
#                         with_cp=with_cp,
#                     )
#                     for _ in range(2)
#                 ]
#             )
#         else:
#             self.extra_extractors = None

#     def forward(self, x: Tensor, c: Tensor, cls: Tensor, deform_inputs1, deform_inputs2, H_c: int, W_c: int, H_toks: int, W_toks: int):
#         c = self.extractor(
#             query=c,
#             reference_points=deform_inputs2[0],
#             feat=x,
#             spatial_shapes=deform_inputs2[1],
#             level_start_index=deform_inputs2[2],
#             H=H_c,
#             W=W_c,
#         )
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(
#                     query=c,
#                     reference_points=deform_inputs2[0],
#                     feat=x,
#                     spatial_shapes=deform_inputs2[1],
#                     level_start_index=deform_inputs2[2],
#                     H=H_c,
#                     W=W_c,
#                 )
#         return x, c, cls


# class SpatialPriorModule(nn.Module):
#     def __init__(self, inplanes: int = 64, embed_dim: int = 384, with_cp: bool = False):
#         super().__init__()
#         self.with_cp = with_cp
#         self.stem = nn.Sequential(
#             *[
#                 nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(inplanes),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.SyncBatchNorm(inplanes),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.SyncBatchNorm(inplanes),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             ]
#         )
#         self.conv2 = nn.Sequential(
#             *[
#                 nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(2 * inplanes),
#                 nn.ReLU(inplace=True),
#             ]
#         )
#         self.conv3 = nn.Sequential(
#             *[
#                 nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(4 * inplanes),
#                 nn.ReLU(inplace=True),
#             ]
#         )
#         self.conv4 = nn.Sequential(
#             *[
#                 nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.SyncBatchNorm(4 * inplanes),
#                 nn.ReLU(inplace=True),
#             ]
#         )
#         self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x: Tensor):
#         def _inner_forward(x: Tensor):
#             c1 = self.stem(x)
#             c2 = self.conv2(c1)
#             c3 = self.conv3(c2)
#             c4 = self.conv4(c3)
#             c1 = self.fc1(c1)
#             c2 = self.fc2(c2)
#             c3 = self.fc3(c3)
#             c4 = self.fc4(c4)
#             bs, dim, _, _ = c1.shape
#             c2 = c2.view(bs, dim, -1).transpose(1, 2)
#             c3 = c3.view(bs, dim, -1).transpose(1, 2)
#             c4 = c4.view(bs, dim, -1).transpose(1, 2)
#             return c1, c2, c3, c4

#         if self.with_cp and x.requires_grad:
#             outs = cp.checkpoint(_inner_forward, x)
#         else:
#             outs = _inner_forward(x)
#         return outs


# @MODELS.register_module()
# class Dinov3Adapter(BaseModule):
#     def __init__(
#         self,
#         backbone,
#         interaction_indexes: Sequence[int] = (2, 5, 8, 11),
#         pretrain_size: int = 224,
#         conv_inplane: int = 64,
#         n_points: int = 4,
#         deform_num_heads: int = 16,
#         drop_path_rate: float = 0.3,
#         init_values: float = 0.0,
#         with_cffn: bool = True,
#         cffn_ratio: float = 0.25,
#         deform_ratio: float = 0.5,
#         add_vit_feature: bool = True,
#         use_extra_extractor: bool = True,
#         with_cp: bool = True,
#         init_cfg=None,
#     ):
#         super().__init__(init_cfg)
#         # Build or use provided backbone
#         if isinstance(backbone, dict):
#             self.backbone = MODELS.build(backbone)
#         else:
#             self.backbone = backbone
#         # Freeze backbone
#         self.backbone.requires_grad_(False)
#         self.backbone.eval()

#         self.pretrain_size = (pretrain_size, pretrain_size)
#         self.interaction_indexes = list(interaction_indexes)
#         self.add_vit_feature = add_vit_feature
#         embed_dim = self.backbone.embed_dim
#         self.patch_size = self.backbone.patch_size

#         self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
#         self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
#         self.interactions = nn.Sequential(
#             *[
#                 InteractionBlockWithCls(
#                     dim=embed_dim,
#                     num_heads=deform_num_heads,
#                     n_points=n_points,
#                     init_values=init_values,
#                     drop_path=drop_path_rate,
#                     norm_layer=lambda c: nn.LayerNorm(c, eps=1e-6),
#                     with_cffn=with_cffn,
#                     cffn_ratio=cffn_ratio,
#                     deform_ratio=deform_ratio,
#                     extra_extractor=((True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor),
#                     with_cp=with_cp,
#                 )
#                 for i in range(len(self.interaction_indexes))
#             ]
#         )
#         self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
#         self.norm1 = nn.SyncBatchNorm(embed_dim)
#         self.norm2 = nn.SyncBatchNorm(embed_dim)
#         self.norm3 = nn.SyncBatchNorm(embed_dim)
#         self.norm4 = nn.SyncBatchNorm(embed_dim)

#         # Initialize
#         self._init_adapter_weights()

#     def _init_adapter_weights(self):
#         def _init_weights(m: nn.Module):
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)
#             elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 fan_out //= m.groups
#                 m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#         self.up.apply(_init_weights)
#         self.spm.apply(_init_weights)
#         self.interactions.apply(_init_weights)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         nn.init.normal_(self.level_embed)

#     def _add_level_embed(self, c2: Tensor, c3: Tensor, c4: Tensor):
#         c2 = c2 + self.level_embed[0]
#         c3 = c3 + self.level_embed[1]
#         c4 = c4 + self.level_embed[2]
#         return c2, c3, c4

#     def forward(self, x: Tensor):
#         deform_inputs1, deform_inputs2 = _deform_inputs(x, self.patch_size)
#         c1, c2, c3, c4 = self.spm(x)
#         c2, c3, c4 = self._add_level_embed(c2, c3, c4)
#         c = torch.cat([c2, c3, c4], dim=1)

#         H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
#         H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

#         with torch.no_grad():
#             all_layers = self.backbone.get_intermediate_layers(
#                 x, n=self.interaction_indexes, return_class_token=True, reshape=False
#             )

#         x_for_shape, _ = all_layers[0]
#         bs, _, dim = x_for_shape.shape
#         del x_for_shape

#         outs = []
#         for i, layer in enumerate(self.interactions):
#             tokens_i, cls_i = all_layers[i]
#             _, c, _ = layer(
#                 tokens_i,
#                 c,
#                 cls_i,
#                 deform_inputs1,
#                 deform_inputs2,
#                 H_c,
#                 W_c,
#                 H_toks,
#                 W_toks,
#             )
#             outs.append(tokens_i.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

#         # split & reshape c back to 3 levels and fuse
#         c2_ = c[:, 0 : c2.size(1), :]
#         c3_ = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
#         c4_ = c[:, c2.size(1) + c3.size(1) :, :]

#         c2_ = c2_.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
#         c3_ = c3_.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
#         c4_ = c4_.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
#         c1 = self.up(c2_) + c1

#         if self.add_vit_feature:
#             x1, x2, x3, x4 = outs
#             x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
#             x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
#             x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
#             x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
#             c1, c2_, c3_, c4_ = c1 + x1, c2_ + x2, c3_ + x3, c4_ + x4

#         f1 = self.norm1(c1)
#         f2 = self.norm2(c2_)
#         f3 = self.norm3(c3_)
#         f4 = self.norm4(c4_)
#         return [f1, f2, f3, f4]
