import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    kitti360_root = Path("/home/johnl/data/KITTI-360")
    SSCBench_root = Path("/home/johnl/data/SSCBenchKITTI360")
    sequence = "2013_05_28_drive_0010_sync"
    seg_gt_path = kitti360_root / "data_2d_semantics" / sequence / "image_00" / "semantic"
    
    labels_to_check = [28, 31]
    skip_files = []
    check_files = [2745, 2395, 2275, 1175, ]
    
    for file_name in seg_gt_path.glob("*.png"):
        file_id = int(file_name.stem)
        if file_id in skip_files:
            continue
        seg_gt = np.array(Image.open(file_name))
        
        #? Check if the unique labels overlap with the labels_to_check
        unique_labels = np.unique(seg_gt)
        if any(label in unique_labels for label in labels_to_check):
            overlap_labels = [label for label in unique_labels if label in labels_to_check]
            print(f"{file_id}: Found overlap {overlap_labels}")
            
            # #? Find image from SSCBench that is visually the same as the current image
            # current_image = np.array(Image.open(kitti360_root / "data_2d_raw" / sequence / "image_00" / "data_rect" / file_name.name))
            
            # for ssc_file_name in sorted((SSCBench_root / "data_2d_raw" / sequence / "image_00" / "data_rect").glob("*.png")):
            #     ssc_file_id = int(ssc_file_name.stem)
            #     if ssc_file_id < file_id - 500:
            #         continue
                
            #     ssc_image = np.array(Image.open(ssc_file_name))
            #     if np.array_equal(current_image, ssc_image):
            #         if ssc_file_id % 5 == 0:
            #             print(f"Found SSC image: {ssc_file_name}")

def image_labels(seg_path: Path):
    seg_gt = np.array(Image.open(seg_path))
    unique_labels = np.unique(seg_gt)
    return unique_labels

def get_class_mask(seg_path: Path, class_id: int):
    seg_gt = np.array(Image.open(seg_path))
    return seg_gt == class_id


if __name__ == "__main__":
    main()
    # seg_path = Path("/home/johnl/data/KITTI-360/data_2d_semantics/2013_05_28_drive_0010_sync/image_00/semantic/0000003011.png")
    # print(image_labels(seg_path))
    # #? Save boolean mask
    # mask = get_class_mask(seg_path, 40)
    # Image.fromarray(mask).save("../../misc/cgformer/mask.png")