"""
This script compares the filenames in the KITTI-360 dataset with the filenames in the
SSCBench-KITTI360 dataset which is a subset of the KITTI-360 dataset.
For each sequence, it creates a csv file with the following columns:
- sscbench_filename
- kitti360_filename

Each file in SSCBench-KITTI360 has a corresponding file in KITTI-360 
but not all files in KITTI-360 are in SSCBench-KITTI360.

The script compares images in SSCBench-KITTI360 with images in KITTI-360.
If the image is the same, it writes the kitti360_filename to the csv file.

Optionally, it can center-crop the images to save time.
"""
import argparse
import csv
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def compare_images(img1_path, img2_path, crop_size=None):
    """
    Compares two images, with an option to compare their center crops.

    Args:
        img1_path (Path): Path to the first image.
        img2_path (Path): Path to the second image.
        crop_size (int, optional): The size of the square center crop. 
                                   If None, compares full images. Defaults to None.

    Returns:
        bool: True if the images (or their crops) are identical, False otherwise.
    """
    try:
        with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
            if img1.size != img2.size or img1.mode != img2.mode:
                return False

            if crop_size:
                w, h = img1.size
                left = (w - crop_size) / 2
                top = (h - crop_size) / 2
                right = (w + crop_size) / 2
                bottom = (h + crop_size) / 2
                img1 = img1.crop((left, top, right, bottom))
                img2 = img2.crop((left, top, right, bottom))

            return np.array_equal(np.array(img1), np.array(img2))
    except Exception as e:
        print(f"Error comparing images {img1_path} and {img2_path}: {e}")
        return False

def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare and match images between SSCBench-KITTI360 and KITTI-360."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Name of the sequence to process (e.g., '2013_05_28_drive_0000_sync').",
    )
    parser.add_argument(
        "--kitti360_root",
        type=Path,
        required=True,
        help="Root directory of the KITTI-360 dataset.",
    )
    parser.add_argument(
        "--sscbench_root",
        type=Path,
        required=True,
        help="Root directory of the SSCBench-KITTI360 dataset.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to save the output CSV file.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=None,
        help="Size of the square center crop for comparison. If not specified, full images are compared.",
    )
    args = parser.parse_args()

    #? Define image directories
    image_dir_suffix = Path("data_2d_raw") / args.seq / "image_00/data_rect"
    kitti360_img_dir = args.kitti360_root / image_dir_suffix
    sscbench_img_dir = args.sscbench_root / image_dir_suffix

    if not kitti360_img_dir.is_dir():
        print(f"Error: KITTI-360 directory not found: {kitti360_img_dir}")
        return
    if not sscbench_img_dir.is_dir():
        print(f"Error: SSCBench directory not found: {sscbench_img_dir}")
        return

    #? Get sorted lists of image filenames
    sscbench_images = sorted(os.listdir(sscbench_img_dir))
    kitti360_images = sorted(os.listdir(kitti360_img_dir))

    #? Create output directory if it doesn't exist
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    matches = []
    kitti_search_start_index = 0
    matched_kitti_files = set()

    print(f"Starting comparison for sequence: {args.seq}")
    print(f"SSCBench images: {len(sscbench_images)}")
    print(f"KITTI-360 images: {len(kitti360_images)}")

    with tqdm(total=len(sscbench_images), desc="Matching images") as pbar:
        for ssc_img_name in sscbench_images:
            ssc_img_path = sscbench_img_dir / ssc_img_name
            found_match = False
            
            for i in range(kitti_search_start_index, len(kitti360_images)):
                kitti_img_name = kitti360_images[i]
                kitti_img_path = kitti360_img_dir / kitti_img_name
                
                if compare_images(ssc_img_path, kitti_img_path, args.crop_size):
                    if kitti_img_name in matched_kitti_files:
                        raise ValueError(
                            f"Duplicate match found! KITTI-360 file '{kitti_img_name}' "
                            f"was already matched with a previous SSCBench file."
                        )
                    matched_kitti_files.add(kitti_img_name)
                    matches.append(
                        {"sscbench_filename": ssc_img_name, "kitti360_filename": kitti_img_name}
                    )
                    kitti_search_start_index = i + 1
                    found_match = True
                    break
            
            if not found_match:
                print(f"Warning: No match found for {ssc_img_name}")
            
            pbar.update(1)

    #? Write matches to CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sscbench_filename", "kitti360_filename"])
        writer.writeheader()
        writer.writerows(matches)

    print(f"\nComparison finished. Found {len(matches)} matches.")
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()