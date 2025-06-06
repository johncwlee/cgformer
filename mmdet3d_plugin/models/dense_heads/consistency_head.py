import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class ConsistencyHead(nn.Module):
    def __init__(self, learnable_fuse: bool = False, temperature: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.learnable_fuse = learnable_fuse
        if self.learnable_fuse:
            raise NotImplementedError("Learnable fuse is not implemented yet")
            #* Idea: combine the sampled logits with the voxel logits with learnable weights
            #* Then, use the combined logits to compute the loss
            #TODO: need to use these combined logits as output of the entire network
        
    def loss(self, seg_logits, voxel_logits, cam_params, img_metas):
        """
        Compute the consistency loss between 2D segmentation and 3D voxel predictions.
        
        Args:
            seg_logits (torch.Tensor): Predicted 2D segmentation logits with shape (B, C, H, W).
            voxel_logits (torch.Tensor): Predicted 3D voxel logits with shape (B, C, H, W, Z).
            cam_params (list): Camera parameters [rots, trans, intrins, post_rots, post_trans, bda].
            img_metas (list): Image metadata containing camera info.
            
        Returns:
            loss (dict): Dictionary containing the consistency loss.
        """
        #? 1. Get voxel center coordinates in world space
        voxel_centers = self.get_voxel_centers(img_metas)

        #? 2. Project voxel centers to camera views to get a sampling grid
        sampling_grid, valid_mask = self.project_voxels_to_image(voxel_centers, cam_params, img_metas)
        
        if valid_mask.sum() == 0:
            return {'loss_consistency': torch.tensor(0.0, device=seg_logits.device)}
        
        #? 3. Sample the 2D segmentation logits at the projected locations
        sampled_logits = F.grid_sample(
            seg_logits, 
            sampling_grid,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) #* (B, N_cls, 1, occ_X * occ_Y * occ_Z)
        sampled_logits = sampled_logits.permute(0, 2, 3, 1).squeeze(1) #* (B, occ_X * occ_Y * occ_Z, N_cls)
        
        #? 4. Get mask for the current camera
        #TODO: check for num_cam > 1
        cam_valid_mask = valid_mask[0] #* (B, occ_X * occ_Y * occ_Z)
        
        #? 5. Select logits for valid points only
        seg_logits_valid = sampled_logits[cam_valid_mask]
        B, C, occ_X, occ_Y, occ_Z = voxel_logits.shape
        voxel_logits = voxel_logits.view(B, C, -1) #* (B, C, occ_X * occ_Y * occ_Z)
        voxel_logits = voxel_logits.permute(0, 2, 1) #* (B, occ_X * occ_Y * occ_Z, C)
        voxel_logits_valid = voxel_logits[cam_valid_mask]
        
        #? 6. Compute KL-Divergence Loss
        log_p_2d = F.log_softmax(seg_logits_valid / self.temperature, dim=1)
        log_p_3d = F.log_softmax(voxel_logits_valid / self.temperature, dim=1)
    
        #* KL(3D || 2D) - 3D learns from 2D
        kl_loss = (self.temperature ** 2) * F.kl_div(log_p_3d, log_p_2d, log_target=True, reduction='batchmean')
        
        return {'loss_consistency': kl_loss * self.loss_weight}

    def visualize_occ_in_2d(self, occ_labels, cam_params, img_metas):
        """
        Visualize the occupancy labels in 2D.
        """
        voxel_centers = self.get_voxel_centers(img_metas)
        pix_coords, valid_mask = self.project_voxels_to_image(voxel_centers, cam_params, img_metas)
        pix_coords = pix_coords[:, 0, ...]  #* First camera
        valid_mask = valid_mask[:, 0, ...]
        
        ogfH, ogfW = img_metas['img_shape']
        ogfH, ogfW = ogfH[0].item(), ogfW[0].item()
        seg_map = torch.zeros(pix_coords.shape[0], ogfH, ogfW, dtype=occ_labels.dtype, device=occ_labels.device)
        
        for b in range(pix_coords.shape[0]):
            pix_coords_valid = pix_coords[b, valid_mask[b]]
            x_coords = torch.floor(pix_coords_valid[:, 0] * ogfW)
            y_coords = torch.floor(pix_coords_valid[:, 1] * ogfH)
            seg_map[b, y_coords.long(), x_coords.long()] = occ_labels[b].view(-1)[valid_mask[b]]
        
        return seg_map

    def get_voxel_centers(self, img_metas):
        """
        Get the voxel centers in world coordinates.
        Returns:
            voxel_centers (torch.Tensor): Voxel centers in world coordinates with shape (B, 1, occ_X * occ_Y * occ_Z, 3)
        """
        occ_X, occ_Y, occ_Z = img_metas['occ_size'][0]
        pc_range = img_metas['pc_range'][0]
        B, device = img_metas['pc_range'].shape[0], pc_range.device

        xv, yv, zv = torch.meshgrid(
            torch.arange(occ_X, device=device), 
            torch.arange(occ_Y, device=device), 
            torch.arange(occ_Z, device=device), 
            indexing='ij'
        )
        ref_3d = torch.cat([
            (xv.reshape(-1, 1) + 0.5) / occ_X,  # Normalized X coordinates 
            (yv.reshape(-1, 1) + 0.5) / occ_Y,  # Normalized Y coordinates
            (zv.reshape(-1, 1) + 0.5) / occ_Z   # Normalized Z coordinates
        ], dim=-1)
        ref_3d = ref_3d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)    #* (bs, 1, occ_X * occ_Y * occ_Z, 3)
        ref_3d[..., 0] = ref_3d[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref_3d[..., 1] = ref_3d[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref_3d[..., 2] = ref_3d[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        
        return ref_3d

    def project_voxels_to_image(self, voxel_centers, cam_params, img_metas):
        """
        Project 3D voxel centers to 2D image coordinates.
        Uses the same projection pipeline as VoxFormerHead for consistency.
        
        Args:
            voxel_centers (torch.Tensor): Voxel centers in world coordinates with shape (B, 1, occ_X * occ_Y * occ_Z, 3)
            cam_params (list): Camera parameters [rots, trans, intrins, post_rots, post_trans, bda]
            img_metas (list): Image metadata containing camera info
            
        Returns:
            pixel_coords (torch.Tensor): Projected voxel centers in image coordinates 
                with shape (B, num_cam, occ_X * occ_Y * occ_Z, 2)
            valid_pixels (torch.Tensor): Validity mask with shape (B, num_cam, occ_X * occ_Y * occ_Z)
        """
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, num_cam = rots.shape[:2]
        ogfH, ogfW = img_metas['img_shape']  # Use first camera's dimensions
        ogfH, ogfW = ogfH[0].item(), ogfW[0].item()
        eps = 1e-5
        
        #? Reshape to match encoder format: [D, B, num_query, 3]
        reference_points = voxel_centers.permute(1, 0, 2, 3)
        D, B, num_points = reference_points.size()[:3]
        
        #? Expand for all cameras: [D, B, num_cam, num_query, 3]
        reference_points = reference_points.view(
            D, B, 1, num_points, 3).repeat(1, 1, num_cam, 1, 1)
        #TODO: check for num_cam > 1
        
        #? Apply BDA transformation if present
        if bda.shape[-1] == 4:
            reference_points = torch.cat((reference_points, torch.ones(*reference_points.shape[:-1], 1).type_as(reference_points)), dim=-1)
            reference_points = torch.inverse(bda).view(1, B, 1, 1, 4, 4).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
            reference_points = reference_points[..., :3]
        else:
            reference_points = torch.inverse(bda).view(1, B, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        
        #? Transform from ego to camera coordinates
        reference_points = reference_points - trans.view(1, B, num_cam, 1, 3)
        inv_rots = rots.inverse().view(1, B, num_cam, 1, 3, 3)
        reference_points = (inv_rots @ reference_points.unsqueeze(-1)).squeeze(-1)
        
        #? Project to image coordinates
        if intrins.shape[3] == 4:            
            reference_points = torch.cat((reference_points, torch.ones(*reference_points.shape[:-1], 1).type_as(reference_points)), dim=-1)
            reference_points_cam = (intrins.view(1, B, num_cam, 1, 4, 4) @ reference_points.unsqueeze(-1)).squeeze(-1)
        else:
            reference_points_cam = (intrins.view(1, B, num_cam, 1, 3, 3) @ reference_points.unsqueeze(-1)).squeeze(-1)
        
        #? Convert to pixel coordinates
        points_d = reference_points_cam[..., 2:3]
        reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(
            points_d, torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )
        
        #? Apply post-processing transformations
        reference_points_cam[..., 0:2] = (post_rots[:, :, :2, :2].view(1, B, num_cam, 1, 2, 2) @ reference_points_cam[..., 0:2].unsqueeze(-1)).squeeze(-1)
        reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] + post_trans[:, :, :2].view(1, B, num_cam, 1, 2)
        
        #? Extract final normalized coordinates
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        pixel_coords = reference_points_cam[..., 0:2]   #* (1, B, num_cam, occ_X * occ_Y * occ_Z, 2)

        #? Create validity mask
        valid_depth = points_d > eps
        valid_pixels = (valid_depth 
            & (pixel_coords[..., 0:1] > eps) 
            & (pixel_coords[..., 0:1] < (1.0 - eps)) 
            & (pixel_coords[..., 1:2] > eps) 
            & (pixel_coords[..., 1:2] < (1.0 - eps))
        )
        
        pixel_coords = pixel_coords.permute(2, 1, 3, 0, 4).squeeze(-2)  # [num_cam, B, num_query, 2]
        valid_pixels = valid_pixels.permute(2, 1, 3, 0, 4).squeeze(-1).squeeze(-1)  # [num_cam, B, num_query]
        
        return pixel_coords, valid_pixels