#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Default Configuration ---
KITTI360_ROOT="/mnt/storage/KITTI-360"
SSCBENCH_ROOT="/mnt/storage/SSCBenchKITTI360"
OUTPUT_DIR="./output/kitti360_mapping"
CROP_SIZE=512

# --- Usage function ---
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --kitti360_root <path>    Path to KITTI-360 root directory."
    echo "                            Default: $KITTI360_ROOT"
    echo "  --sscbench_root <path>    Path to SSCBench-KITTI360 root directory."
    echo "                            Default: $SSCBENCH_ROOT"
    echo "  --output_dir <path>       Directory to save output CSV files."
    echo "                            Default: $OUTPUT_DIR"
    echo "  --crop_size <int>         Size for center crop comparison. Omit for full image."
    echo "                            Default: $CROP_SIZE"
    echo "  -h, --help                Show this help message."
    exit 1
}

# --- Parse command-line arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --kitti360_root) KITTI360_ROOT="$2"; shift ;;
        --sscbench_root) SSCBENCH_ROOT="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --crop_size) CROP_SIZE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# --- Sequence Configuration ---
# Add the names of the sequences you want to process to this list
SEQUENCES=(
    "2013_05_28_drive_0000_sync"
    "2013_05_28_drive_0002_sync"
    "2013_05_28_drive_0003_sync"
    "2013_05_28_drive_0004_sync"
    "2013_05_28_drive_0005_sync"
    "2013_05_28_drive_0006_sync"
    "2013_05_28_drive_0007_sync"
    "2013_05_28_drive_0008_sync"
    "2013_05_28_drive_0009_sync"
    "2013_05_28_drive_0010_sync"
)
# ---------------------

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each sequence and run the Python script
for SEQ in "${SEQUENCES[@]}"; do
    echo "-----------------------------------------------------"
    echo "Processing sequence: $SEQ"
    echo "-----------------------------------------------------"

    OUTPUT_CSV="$OUTPUT_DIR/${SEQ}.csv"
    
    # Build the command
    CMD="python tools/check_kitti360_files.py \
        --seq \"$SEQ\" \
        --kitti360_root \"$KITTI360_ROOT\" \
        --sscbench_root \"$SSCBENCH_ROOT\" \
        --output_csv \"$OUTPUT_CSV\""

    if [ -n "$CROP_SIZE" ]; then
        CMD="$CMD --crop_size $CROP_SIZE"
    fi

    # Execute the command
    eval $CMD

    echo "Finished processing $SEQ. Output saved to $OUTPUT_CSV"
    echo ""
done

echo "All sequences processed successfully." 