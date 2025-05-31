

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule, force_fp32

from .modules.dinov2 import DINOv2
from .modules.dpt_layers import FeatureFusionBlock
from .modules.zoedepth.attractor import AttractorLayer, AttractorLayerUnnormed
from .modules.zoedepth.dist_layers import ConditionalLogBinomial
from .modules.zoedepth.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output
    return hook

class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward_features(self, out_features, patch_h, patch_w):
        """Return the intermediate features of the depth head."""
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        return path_1
    
    def forward(self, out_features, patch_h, patch_w):
        feats = self.forward_features(out_features, patch_h, patch_w)
        out = self.scratch.output_conv1(feats)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out

class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of)
                 * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode='bilinear', align_corners=True)


@NECKS.register_module()
class DepthAnything(BaseModule):
    model_names = {
        's': 'vits',
        'b': 'vitb',
        'l': 'vitl'
    }
    layer_idx_dict = {
        's': [2, 5, 8, 11],
        'b': [2, 5, 8, 11], 
        'l': [4, 11, 17, 23]
    }
    features_dim_dict = {
        's': 64,
        'b': 128,
        'l': 256
    }
    out_channels_dict = {
        's': [48, 96, 192, 384],
        'b': [96, 192, 384, 768],
        'l': [256, 512, 1024, 1024]
    }
    dinov2_dims = {
        's': 384,
        'b': 768,
        'l': 1024
    }
    def __init__(
        self, 
        encoder_size: str, 
        img_size: Union[int, tuple] = 518,
        resize_method: str = "minimal",
        keep_aspect_ratio: bool = True,
        use_last_layers: bool = False,
        use_bn=False, 
        use_clstoken=False,
        n_bins: int = 64,
        min_depth: float = 1e-5,
        max_depth: float = 100.0,
        attractor_alpha: float = 300,
        attractor_gamma: float = 2,
        attractor_kind: str = 'sum',
        attractor_type: str = 'exp',
        min_temp: float = 5,
        max_temp: float = 50,
        pretrained=None,
    ):
        super(DepthAnything, self).__init__()

        self.embed_dim = self.features_dim_dict[encoder_size]
        self.patch_size = 14
        out_channels = self.out_channels_dict[encoder_size]
        self.layer_idx = self.layer_idx_dict[encoder_size] if not use_last_layers else 4
        
        self.pretrained = DINOv2(model_name=self.model_names[encoder_size])
        self.depth_head = DPTHead(self.dinov2_dims[encoder_size], 
                                  self.embed_dim, 
                                  use_bn, 
                                  out_channels=out_channels, 
                                  use_clstoken=use_clstoken)
        self.handles = []
        self.depth_head_out = {}
        self.layer_names = ['out_conv','l4_rn', 'r4', 'r3', 'r2', 'r1']
        self.attach_hooks()

        num_out_features = [256, 256, 256, 256]
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # btlnck conv
        self.seed_bin_regressor = SeedBinRegressorUnnormed(256, 
                                                           n_bins=n_bins, 
                                                           min_depth=min_depth, 
                                                           max_depth=max_depth)
        self.seed_projector = Projector(256, 128)
        self.projectors = nn.ModuleList([
            Projector(num_out, 128)
            for num_out in num_out_features
        ])
        n_attractors = [16, 8, 4, 1]
        self.attractors = nn.ModuleList([
            AttractorLayerUnnormed(128, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])
        
        last_in = 33  # 32 for the seed bin regressor + 1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, 128, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)
        
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            checkpoint = {k.replace('core.core.', ''): v for k, v in checkpoint.items()}
            self.load_state_dict(checkpoint, strict=True)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        self.resize = Resize(net_w, 
                             net_h, 
                             keep_aspect_ratio=keep_aspect_ratio, 
                             ensure_multiple_of=14, 
                             resize_method=resize_method)

    def attach_hooks(self):
        if len(self.handles) > 0:
            self.remove_hooks()
        if "out_conv" in self.layer_names:
            self.handles.append(list(self.depth_head.scratch.output_conv2.children())[
                                1].register_forward_hook(get_activation("out_conv", self.depth_head_out)))
        if "r4" in self.layer_names:
            self.handles.append(self.depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.depth_head_out)))
        if "r3" in self.layer_names:
            self.handles.append(self.depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.depth_head_out)))
        if "r2" in self.layer_names:
            self.handles.append(self.depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.depth_head_out)))
        if "r1" in self.layer_names:
            self.handles.append(self.depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.depth_head_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(self.depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.depth_head_out)))

        return self

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()
    
    def forward(self, x):
        B, N, C, H, W = x.shape
        
        x = self.resize(x.view(B * N, C, H, W))  # Resize to (B*N, C, H', W')
        assert x.shape[-2] % 14 == 0 and x.shape[-1] % 14 == 0, \
            f"Input image size ({x.shape[-2]}, {x.shape[-1]}) must be divisible by 14."
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        dino_features = self.pretrained.get_intermediate_layers(x, self.layer_idx, return_class_token=True)
        out = self.depth_head(dino_features, patch_h, patch_w)
        rel_depth = F.relu(out).squeeze(1)

        out = [self.depth_head_out[k] for k in self.layer_names]
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]
        
        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)
        b_prev = seed_b_centers
        
        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        depth = torch.sum(x * b_centers, dim=1, keepdim=True)

        #? Interpolate to original size
        #* This is also done in Depth Anything as the final step
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)

        return depth

#TODO: remove all this processing and load Depth Anything depth maps directly