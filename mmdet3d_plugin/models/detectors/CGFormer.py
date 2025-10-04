import torch
from torch.nn import functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class CGFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None,
        depth_anything=None,
        distillation_head=None,
    ):
        super().__init__()

        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)

        self.depth_net = MODELS.build(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = MODELS.build(img_view_transformer)
        self.proposal_layer = MODELS.build(proposal_layer)
        self.VoxFormer_head = MODELS.build(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = MODELS.build(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = MODELS.build(occ_encoder_neck)
        
        self.pts_bbox_head = MODELS.build(pts_bbox_head)

        self.depth_loss = depth_loss

        if depth_anything is not None:
            self.depth_anything = MODELS.build(depth_anything)
            self.depth_anything.eval()
            for param in self.depth_anything.parameters():
                param.requires_grad = False
        else:
            self.depth_anything = None
        
        if distillation_head is not None:
            self.distillation_head = MODELS.build(distillation_head)
        else:
            self.distillation_head = None

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0])

        mlp_input = self.depth_net.get_mlp_input(*img_inputs[1:7])
        context, depth = self.depth_net([img_enc_feats] + img_inputs[1:7] + [mlp_input], img_metas)
        
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, img_inputs[1:7])
        else:
            coarse_queries = None

        proposal = self.proposal_layer(img_inputs[1:7], img_metas)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=img_inputs[1:7],
            lss_volume=coarse_queries,
            img_metas=img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)]
        )
        
        if self.training:
            return x, depth, img_enc_feats

        return x, depth
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        if self.depth_anything is not None:
            img_metas['stereo_depth'] = self.depth_anything(img_inputs[0])

        img_voxel_feats, depth, img_feats = self.extract_img_feat(img_inputs, img_metas)
        
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        #TODO: not sure why we need to do this; perhaps for batch size of 1?
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        #* Occ head
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_inputs['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)
        
        if self.distillation_head is not None:
            B, N, C, H, W = img_feats.shape
            img_feats = img_feats.view(B * N, C, H, W)
            losses_distillation = self.distillation_head.loss(
                img_feats=img_feats,
                voxel_feats=voxel_feats_enc[0],
                img=img_inputs[0],
                depth=depth,
                lss_encoder=self.img_view_transformer,
                img_metas=img_metas,
                cam_params=img_inputs[1:7],
                gt_occ=gt_occ,
            )
            losses.update(losses_distillation)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']
        
        if self.depth_anything is not None:
            img_metas['stereo_depth'] = self.depth_anything(img_inputs[0])

        img_voxel_feats, _, = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)