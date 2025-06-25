import yaml
import numpy as np


SEMANTIC_KITTI_COLORS = np.array([
  [0, 0, 0, 255],  # void
  [100, 150, 245, 255],
  [100, 230, 245, 255],
  [30, 60, 150, 255],
  [80, 30, 180, 255],
  [100, 80, 250, 255],
  [255, 30, 30, 255],
  [187, 67, 255, 255],
  [150, 30, 90, 255],
  [255, 0, 255, 255],
  [255, 150, 255, 255],
  [75, 0, 75, 255],
  [175, 0, 75, 255],
  [255, 200, 0, 255],
  [255, 120, 50, 255],
  [0, 175, 0, 255],
  [135, 60, 0, 255],
  [150, 240, 80, 255],
  [255, 240, 150, 255],
  [255, 0, 0, 255],
], dtype=np.uint8)

def get_inv_map():
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''
  config_path = "./configs/semantickitti/SemanticKITTI.yaml"
  dataset_config = yaml.safe_load(open(config_path, 'r'))
  # make lookup table for mapping
  inv_map = np.zeros(20, dtype=np.int32)
  inv_map[list(dataset_config['learning_map_inv'].keys())] = list(dataset_config['learning_map_inv'].values())

  return inv_map


def create_colored_segmentation_map(pred_labels, colors):
    """
    Convert predicted class labels to colored segmentation maps.
    
    Args:
        pred_labels: numpy array of shape (B, H, W) with integer class labels
        colors: numpy array of shape (num_classes, 4) with RGBA colors
    
    Returns:
        colored_maps: numpy array of shape (B, H, W, 4) with colored segmentation maps
    """
    B, H, W = pred_labels.shape
    colored_maps = np.zeros((B, H, W, 4), dtype=np.uint8)
    
    for b in range(B):
        for c in range(colors.shape[0]):  # Fixed: use colors.shape[0] instead of colors.shape
            mask = (pred_labels[b] == c)
            colored_maps[b][mask] = colors[c]
    
    return colored_maps

def get_fov_mask(transform, 
                 intr, 
                 grid_size=[256, 256, 32],
                 origin=[0, -25.6, -2],
                 img_size=[1408, 384]):
	xv, yv, zv = np.meshgrid(
            range(grid_size[0]),
            range(grid_size[1]),
            range(grid_size[2]),
            indexing='ij'
          )
	vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T
	vox_size = 0.2
	offsets = np.array([0.5, 0.5, 0.5]).reshape(1, 3)
	vol_origin = np.array(origin)
	vol_origin = vol_origin.astype(np.float32)
	vox_coords = vox_coords.astype(np.float32)
	cam_pts = vox_coords * vox_size + vox_size * offsets + vol_origin.reshape(1, 3)
	cam_pts = np.hstack([cam_pts, np.ones((len(cam_pts), 1), dtype=np.float32)])
	cam_pts = np.dot(transform, cam_pts.T).T

	intr = intr.astype(np.float32)
	fx, fy = intr[0, 0], intr[1, 1]
	cx, cy = intr[0, 2], intr[1, 2]
	pix = cam_pts[:, 0:2]
	pix[:, 0] = np.round((pix[:, 0] * fx) / cam_pts[:, 2] + cx)
	pix[:, 1] = np.round((pix[:, 1] * fy) / cam_pts[:, 2] + cy)
	pix = pix.astype(np.int32)

	pix_z = cam_pts[:, 2]
	pix_x, pix_y = pix[:, 0], pix[:, 1]
	fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_size[0],
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_size[1],
                pix_z > 0))))
	return fov_mask