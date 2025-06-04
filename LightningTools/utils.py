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