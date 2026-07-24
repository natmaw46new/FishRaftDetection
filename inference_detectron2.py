import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# ============================================================
# CONFIG
# ============================================================
IMAGES_DIR = "./data/rotation_augment+newsplit/inference_tiles/images"
MODEL_WEIGHTS = "./data/rotation_augment+newsplit/model/model_final.pth"
OUTPUT_JSON = "./data/rotation_augment+newsplit/inference_tiles/predictions.json"

BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES = 1
SCORE_THRESH = 0.5

MIN_SIZE_TEST = 1024
MAX_SIZE_TEST = 1024


def build_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.flatten().astype(float).tolist())
    return polygons


def instances_to_results(instances, image_id, category_id=1):
    results = []
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

    for i in range(len(instances)):
        x1, y1, x2, y2 = boxes[i]
        result = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[i]),
        }
        if masks is not None:
            result["segmentation"] = mask_to_polygon(masks[i])
        results.append(result)
    return results


def main():
    predictor = build_predictor()

    image_paths = sorted(Path(IMAGES_DIR).glob("*.png"))
    all_results = []
    image_id_map = {}

    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        image_id_map[path.name] = idx
        all_results.extend(instances_to_results(instances, image_id=idx))

    output = {"image_id_map": image_id_map, "results": all_results}

    os.makedirs(Path(OUTPUT_JSON).parent, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f)

    n_dets = len(all_results)
    print(f"ran inference on {len(image_paths)} tiles, {n_dets} raft detections above threshold {SCORE_THRESH}")
    print(f"wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()