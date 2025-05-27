

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import NECKS
from mmcv.runner import BaseModule, force_fp32

from .modules.dinov2 import DINOv2
from .modules.dpt_layers import FeatureFusionBlock


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

    def forward_features(self, out_features, patch_h, patch_w, idx=0):
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
        #* path_4 has the same shape as the last layer output of dinov2
        if idx == 0:
            return path_4
        
        path = path_4
        layers_rn = [layer_3_rn, layer_2_rn, layer_1_rn]
        refinenets = [self.scratch.refinenet3, self.scratch.refinenet2, self.scratch.refinenet1]
        for i, (layer_rn, refinenet) in enumerate(zip(layers_rn, refinenets, strict=True)):
            path = refinenet(path, layer_rn, size=layers_rn[i+1].shape[2:])
            if i == idx - 1:
                return path
        raise ValueError(f"Invalid index {idx} for DPTHead features.")
    
    def forward(self, out_features, patch_h, patch_w):
        feats = self.forward_features(out_features, patch_h, patch_w)
        out = self.scratch.output_conv1(feats).contiguous()
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


@NECKS.register_module()
class DepthAnythingV2(BaseModule):
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
        use_bn=False, 
        use_clstoken=False,
        pretrained=None,
    ):
        super(DepthAnythingV2, self).__init__()

        self.embed_dim = self.features_dim_dict[encoder_size]
        self.patch_size = 14
        out_channels = self.out_channels_dict[encoder_size]
        self.layer_idx = self.layer_idx_dict[encoder_size]
        
        self.pretrained = DINOv2(model_name=self.model_names[encoder_size])
        self.depth_head = DPTHead(self.dinov2_dims[encoder_size], 
                                  self.embed_dim, 
                                  use_bn, 
                                  out_channels=out_channels, 
                                  use_clstoken=use_clstoken)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            self.load_state_dict(checkpoint, strict=True)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.layer_idx, return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        return depth
