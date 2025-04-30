#!/usr/bin/env python3
import os
import cv2
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def decode_segmentation(ann: Dict[str, Any], height: int, width: int, fill_value: int) -> Optional[np.ndarray]:
    """Decode a segmentation annotation into a mask."""
    segmentation = ann.get('segmentation')
    if segmentation is None:
        logging.warning("Missing 'segmentation' key.")
        return None

    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in segmentation:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [pts], color=fill_value)
        return mask

    elif isinstance(segmentation, dict):
        try:
            rle = mask_utils.frPyObjects(segmentation, height, width)
            decoded_mask = mask_utils.decode(rle)
            return (decoded_mask * fill_value).squeeze().astype(np.uint8)
        except Exception as e:
            logging.warning("Skipping RLE due to decoding error: %s", e)
            return None

    else:
        logging.warning("Unknown segmentation format.")
        return None


def create_masks(images: List[Dict[str, Any]],
                 imgid_to_anns: Dict[int, List[Dict[str, Any]]],
                 reduced_category_map: Dict[int, int],
                 output_dir: str,
                 binary: bool,
                 id_to_class: Dict[int, str],
                 grouping: bool) -> None:
    """
    Generate and save masks for a batch of images.

    In multi-class mode, if grouping is enabled, a per-pixel priority map is maintained so that at each pixel
    the annotation with the highest group priority wins. If grouping is disabled, annotations are applied
    in order and later annotations override earlier ones.
    """
    os.makedirs(output_dir, exist_ok=True)
    for img in tqdm(images, desc="Generating masks"):
        img_id = img['id']
        file_name = img['file_name']
        height, width = img['height'], img['width']
        ann_list = imgid_to_anns.get(img_id, [])
        
        final_mask = np.zeros((height, width), dtype=np.uint8)
        if not binary and grouping:
            priority_mask = np.zeros((height, width), dtype=np.float32)
        
        for ann in ann_list:
            cat_id = ann['category_id']
            if binary:
                fill_value = 1
            else:
                fill_value = reduced_category_map.get(cat_id, 0)

            seg = decode_segmentation(ann, height, width, fill_value)
            if seg is None:
                continue

            if binary:
                final_mask[seg > 0] = 1
            else:
                if grouping:
                    # Group-based logic with priority.
                    cat_name = id_to_class.get(cat_id)
                    group = new_group_mapping.get(cat_name)
                    annotation_priority = new_priority_weights.get(group, 0) if group is not None else 0

                    indices = seg > 0
                    update_indices = indices & (annotation_priority > priority_mask)
                    final_mask[update_indices] = fill_value
                    priority_mask[update_indices] = annotation_priority
                else:
                    # Without grouping, simply update mask; later annotations override earlier ones.
                    final_mask[seg > 0] = fill_value

        mask_filename = os.path.splitext(file_name)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, final_mask)


def save_unique_classes(categories: List[Dict[str, Any]], output_dir: str) -> None:
    """Print and save unique class names to a file."""
    os.makedirs(output_dir, exist_ok=True)
    unique_classes = [cat['name'] for cat in categories]
    print("Unique classes:")
    for cls in unique_classes:
        print(f"\t{cls}")
    print(f"Total number of unique classes: {len(unique_classes)}")
    txt_file_path = os.path.join(os.path.dirname(output_dir), "classes.txt")

    if not os.path.exists(txt_file_path):
        with open(txt_file_path, 'w') as f:
            for cls in unique_classes:
                f.write(f"{cls}\n") 


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser("Generate segmentation masks from COCO annotations.")
    parser.add_argument('--images_dir', required=True, help="Directory containing images.")
    parser.add_argument('--annotations_file', required=True, help="COCO annotations JSON file.")
    parser.add_argument('--output_dir', required=True, help="Directory to save masks.")
    parser.add_argument('--max_images', type=int, default=5000, help="Max images per batch.")
    parser.add_argument('--binary', action='store_true', help="Generate binary masks.")
    # New flag to disable grouping
    parser.add_argument('--no_grouping', action='store_true', help="Disable grouping. "
                        "If set, annotations are applied in order and later ones override earlier ones.")
    return parser.parse_args()


# ----------------- COCO Classes (Ordered) -----------------
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Mapping from label ID to class name.
id_to_class = {i + 1: cls for i, cls in enumerate(coco_classes)}

# ----------------- New Group Mapping -----------------
new_group_mapping = {
    # living beings
    'person': 'living',
    'bird': 'living',
    'cat': 'living',
    'dog': 'living',
    'horse': 'living',
    'sheep': 'living',
    'cow': 'living',
    'elephant': 'living',
    'bear': 'living',
    'zebra': 'living',
    'giraffe': 'living',
    # vehicles
    'bicycle': 'vehicles',
    'car': 'vehicles',
    'motorcycle': 'vehicles',
    'airplane': 'vehicles',
    'bus': 'vehicles',
    'train': 'vehicles',
    'truck': 'vehicles',
    'boat': 'vehicles',
    # urban objects
    'traffic light': 'urban',
    'fire hydrant': 'urban',
    'stop sign': 'urban',
    'parking meter': 'urban',
    'bench': 'urban',
    # sports
    'frisbee': 'sports',
    'skis': 'sports',
    'snowboard': 'sports',
    'sports ball': 'sports',
    'kite': 'sports',
    'baseball bat': 'sports',
    'baseball glove': 'sports',
    'skateboard': 'sports',
    'surfboard': 'sports',
    'tennis racket': 'sports',
    # personal accessories
    'backpack': 'personal',
    'umbrella': 'personal',
    'handbag': 'personal',
    'tie': 'personal',
    'suitcase': 'personal',
    'teddy bear': 'household',
    'hair drier': 'household',
    'toothbrush': 'household',
    # kitchen items (utensils and appliances, prepared food items)
    'bottle': 'kitchen',
    'wine glass': 'kitchen',
    'cup': 'kitchen',
    'fork': 'kitchen',
    'knife': 'kitchen',
    'spoon': 'kitchen',
    'bowl': 'kitchen',
    'sandwich': 'kitchen',
    'broccoli': 'kitchen',
    'carrot': 'kitchen',
    'hot dog': 'kitchen',
    'pizza': 'kitchen',
    'donut': 'kitchen',
    'cake': 'kitchen',
    'microwave': 'kitchen',
    'oven': 'kitchen',
    'toaster': 'kitchen',
    'sink': 'kitchen',
    'refrigerator': 'kitchen',
    # fruit items
    'banana': 'fruit',
    'apple': 'fruit',
    'orange': 'fruit',
    # remaining household/household items
    'chair': 'household',
    'couch': 'household',
    'potted plant': 'household',
    'bed': 'household',
    'dining table': 'household',
    'toilet': 'household',
    'book': 'household',
    'clock': 'household',
    'vase': 'household',
    'scissors': 'household',
    # electronics
    'tv': 'electronics',
    'laptop': 'electronics',
    'mouse': 'electronics',
    'remote': 'electronics',
    'keyboard': 'electronics',
    'cell phone': 'electronics',
}

# ----------------- New Priority Weights -----------------
new_priority_weights = {
    'living': 4.0,
    'vehicles': 3.0,
    'urban': 2.5,
    'sports': 1.5,
    'personal': 1.0,
    'kitchen': 2.0,
    'fruit': 2.0,
    'household': 1.0,
    'electronics': 2.5
}

# ----------------- New Group Labels -----------------
new_group_labels = {
    'living': 25,
    'vehicles': 50,
    'urban': 75,
    'sports': 100,
    'personal': 125,
    'kitchen': 150,
    'fruit': 175,
    'household': 200,
    'electronics': 225
}

# ----------------- Main Function -----------------
def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    try:
        with open(args.annotations_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        logging.error("Error reading annotation file: %s", e)
        return

    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    save_unique_classes(categories, args.output_dir)

    # Map each image id to its list of annotations.
    imgid_to_anns = defaultdict(list)
    for ann in annotations:
        imgid_to_anns[ann['image_id']].append(ann)

    # Build a reduced category mapping using new group labels.
    reduced_category_map = {}
    for cat in categories:
        cat_name = cat['name']
        group = new_group_mapping.get(cat_name)
        if group is not None:
            reduced_category_map[cat['id']] = new_group_labels[group]
        else:
            reduced_category_map[cat['id']] = 0  # Unknown or background

    total_images = len(images)
    batch_size = args.max_images

    # Determine if grouping is enabled (default: grouping enabled).
    grouping = not args.no_grouping

    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        logging.info("Processing images %d to %d...", start, end)
        create_masks(images[start:end],
                     imgid_to_anns,
                     reduced_category_map,
                     args.output_dir,
                     args.binary,
                     id_to_class,
                     grouping)
        # Remove the break to process all batches; kept here for the one batch.
        break

if __name__ == "__main__":
    main()
