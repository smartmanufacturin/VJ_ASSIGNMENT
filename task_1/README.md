# Segmentation Mask Generator

This repository contains a Python project that processes COCO annotations to generate multi-class segmentation masks suitable for training segmentation models.

## Setup & Dependencies

Follow these steps to set up your environment, download the COCO dataset, and run the mask generation script. 

### 1. Environment setup
1. **Clone this repository**:
    ```bash
    git clonehttps://github.com/smartmanufacturin/VJ_ASSIGNMENT.git
    ```

2. **Create and activate a virtual environment** (recommended):
    ```bash
    cd mask_generation_segmentation
    python3 -m venv .mgs
    source .mgs/bin/activate  # For Linux/macOS
    # or ".mgs\Scripts\activate" on Windows
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Download the COCO Dataset:
    ```bash
    mkdir -p coco_dataset && cd coco_dataset
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip  
    cd ..
    ```

### 3. Run the Segmentation Mask Generation Script: 

**For the training set**:
    ```bash
    python3 generate_masks.py \
    --images_dir coco_dataset/train2017 \
    --annotations_file coco_dataset/annotations/instances_train2017.json \
    --output_dir coco_dataset/masks_train \
    --max_images 5000
    ```

**For the validation set**:
    ```bash
    python3 generate_masks.py \
    --images_dir coco_dataset/val2017 \
    --annotations_file coco_dataset/annotations/instances_val2017.json \
    --output_dir coco_dataset/masks_val \
    --max_images 1000
    ```

## Edge Cases Handling:
1. **Overlapping masks**: More priority class annotations overwrite earlier ones to ensure the most relevant segmentation.
2. **Malformed or incorrect polygon data**: Such annotations are logged with warnings and skipped.
3. **Image loading failures**: Images that cannot be loaded are skipped with a warning.
