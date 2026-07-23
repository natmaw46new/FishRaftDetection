import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

# ============================================================
# CONFIG
# ============================================================
VAL_JSON = "./data/output/tiles/instances_val.json"
VAL_IMAGES_DIR = "./data/output/tiles/val/images"
# written directly to cfg.OUTPUT_DIR by COCOEvaluator (Trainer.build_evaluator
# passes output_dir=cfg.OUTPUT_DIR, not an "inference" subfolder) - this file
# holds only the most recent eval run's predictions, gets overwritten each time
PREDICTIONS_JSON = "./data/output/model/coco_instances_results.json"
OUTPUT_DIR = "./data/output/val_visualizations"

SCORE_THRESH = 0.5          # drop predictions below this confidence
ONLY_IMAGES_WITH_GT = True  # skip background-only tiles to keep output small
MAX_IMAGES = 100            # cap output volume; set to None to write every image

GT_COLOR = (0, 255, 0)      # green, BGR (cv2 convention)
PRED_COLOR = (0, 0, 255)    # red, BGR
ALPHA = 0.4                 # fill opacity for the ground-truth mask


def polygon_to_mask(segmentation, height, width):
    # ground-truth segmentation is stored as COCO polygons (list of [x1,y1,x2,y2,...])
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def rle_to_mask(segmentation):
    # predicted segmentation is stored as COCO RLE (COCOEvaluator's format)
    return mask_util.decode(segmentation).astype(np.uint8)


def fill_mask(img, mask, color, alpha=ALPHA):
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def outline_mask(img, mask, color, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def main():
    with open(VAL_JSON) as f:
        val_coco = json.load(f)
    with open(PREDICTIONS_JSON) as f:
        predictions = json.load(f)

    images_by_id = {img["id"]: img for img in val_coco["images"]}

    gt_by_image = defaultdict(list)
    for ann in val_coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    preds_by_image = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= SCORE_THRESH:
            preds_by_image[pred["image_id"]].append(pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_ids = list(images_by_id.keys())
    if ONLY_IMAGES_WITH_GT:
        image_ids = [i for i in image_ids if len(gt_by_image[i]) > 0]
    if MAX_IMAGES is not None:
        image_ids = image_ids[:MAX_IMAGES]

    n_written = 0
    for image_id in image_ids:
        img_info = images_by_id[image_id]
        img_path = Path(VAL_IMAGES_DIR) / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        height, width = img_info["height"], img_info["width"]

        # ground truth: filled + outlined in green
        for ann in gt_by_image[image_id]:
            gt_mask = polygon_to_mask(ann["segmentation"], height, width)
            img = fill_mask(img, gt_mask, GT_COLOR)
            img = outline_mask(img, gt_mask, GT_COLOR)

        # predictions: outlined in red, labeled with confidence score
        for pred in preds_by_image[image_id]:
            pred_mask = rle_to_mask(pred["segmentation"])
            img = outline_mask(img, pred_mask, PRED_COLOR)
            x, y, _, _ = pred["bbox"]
            label = f"{pred['score']:.2f}"
            cv2.putText(
                img, label, (int(x), max(int(y) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA,
            )

        out_path = Path(OUTPUT_DIR) / img_info["file_name"]
        cv2.imwrite(str(out_path), img)
        n_written += 1

    print(f"wrote {n_written} visualizations to {OUTPUT_DIR}")
    print(f"legend: green fill/outline = ground truth, red outline = prediction (score >= {SCORE_THRESH})")


if __name__ == "__main__":
    main()
