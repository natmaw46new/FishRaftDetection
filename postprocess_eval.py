import contextlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

# ============================================================
# CONFIG
# ============================================================
VAL_JSON = "./data/output/tiles/instances_val.json"
VAL_IMAGES_DIR = "./data/output/tiles/val/images"
PREDICTIONS_JSON = "./data/output/model/coco_instances_results.json"
METRICS_JSON = "./data/output/model/metrics.json"

OUTPUT_DIR = "./data/output/eval_report"
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORT_PATH = os.path.join(OUTPUT_DIR, "eval_report.txt")

SCORE_THRESH = 0.5  # confidence cutoff - same operating point as inference_detectron2.py
IOU_THRESH = 0.5    # mask IoU required for a prediction to count as a true positive
ONLY_IMAGES_WITH_GT = True
MAX_IMAGES = 100

GT_COLOR = (0, 255, 0)    # green, BGR
PRED_COLOR = (0, 0, 255)  # red, BGR
ALPHA = 0.4

SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2

AP_HISTORY_KEYS = [
    "iteration",
    "bbox/AP", "bbox/AP50", "bbox/AP75", "bbox/APs", "bbox/APm", "bbox/APl",
    "segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl",
]


# ============================================================
# shared helpers
# ============================================================
def bucket(area):
    if area < SMALL_MAX:
        return "small"
    elif area < MEDIUM_MAX:
        return "medium"
    return "large"


def safe_div(num, denom):
    return num / denom if denom else float("nan")


def polygon_to_mask(segmentation, height, width):
    # only used for drawing - the PR/F1 matching below stays in RLE space
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def rle_to_mask(segmentation):
    return mask_util.decode(segmentation).astype(bool)


def fill_mask(img, mask, color, alpha=ALPHA):
    overlay = img.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def outline_mask(img, mask, color, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


# ============================================================
# section 1: AP trend across every eval period so far, from metrics.json
# ============================================================
def report_ap_history(log):
    log("=" * 78)
    log("AP HISTORY  (metrics.json, every completed EVAL_PERIOD)")
    log("=" * 78)

    if not Path(METRICS_JSON).exists():
        log(f"metrics.json not found at {METRICS_JSON} - skipping")
        log("")
        return

    rows = []
    with open(METRICS_JSON) as f:
        for line in f:
            record = json.loads(line)
            if "bbox/AP" in record:
                rows.append(record)

    if not rows:
        log("No evaluation entries found yet - has an EVAL_PERIOD completed?")
        log("")
        return

    header = " | ".join(f"{k:>10s}" for k in AP_HISTORY_KEYS)
    log(header)
    log("-" * len(header))
    for row in rows:
        log(" | ".join(f"{row.get(k, float('nan')):>10.2f}" for k in AP_HISTORY_KEYS))
    log("")


# ============================================================
# section 2: precision / recall / F1 at a fixed operating point. Matching is
# done entirely in RLE space via pycocotools' compiled mask_util.iou (the same
# call COCOeval itself uses) - never decodes a mask to a full pixel array here,
# which is what made the previous pixel-loop version hang on dense tiles.
# ============================================================
def report_precision_recall_f1(log, coco_gt, coco_dt):
    log("=" * 78)
    log(f"PRECISION / RECALL / F1  (score >= {SCORE_THRESH}, mask IoU >= {IOU_THRESH})")
    log("=" * 78)

    images_by_id = coco_gt.imgs

    overall = {"tp": 0, "fp": 0, "fn": 0}
    by_bucket = defaultdict(lambda: {"tp": 0, "fn": 0})

    # collected here purely for reuse in the visualization section below, so
    # nothing has to be loaded or decoded a second time
    gt_by_image = defaultdict(list)
    preds_by_image = {}

    for image_id in images_by_id:
        gt_anns = coco_gt.imgToAnns.get(image_id, [])
        gt_rles, gt_buckets = [], []
        for ann in gt_anns:
            rle = coco_gt.annToRLE(ann)
            area = ann.get("area") or mask_util.area(rle)
            b = bucket(area)
            gt_rles.append(rle)
            gt_buckets.append(b)
            gt_by_image[image_id].append({"segmentation": ann["segmentation"], "bucket": b})

        dt_anns = [d for d in coco_dt.imgToAnns.get(image_id, []) if d["score"] >= SCORE_THRESH]
        dt_anns.sort(key=lambda d: d["score"], reverse=True)
        preds_by_image[image_id] = dt_anns
        dt_rles = [d["segmentation"] for d in dt_anns]

        ious = mask_util.iou(dt_rles, gt_rles, [0] * len(gt_rles)) if dt_rles and gt_rles else None

        matched = [False] * len(gt_anns)
        for i in range(len(dt_anns)):
            best_iou, best_idx = 0.0, -1
            if ious is not None:
                for j in range(len(gt_anns)):
                    if matched[j]:
                        continue
                    iou = ious[i][j]
                    if iou > best_iou:
                        best_iou, best_idx = iou, j
            if best_idx >= 0 and best_iou >= IOU_THRESH:
                matched[best_idx] = True
                overall["tp"] += 1
                by_bucket[gt_buckets[best_idx]]["tp"] += 1
            else:
                overall["fp"] += 1

        for j, was_matched in enumerate(matched):
            if not was_matched:
                overall["fn"] += 1
                by_bucket[gt_buckets[j]]["fn"] += 1

    precision = safe_div(overall["tp"], overall["tp"] + overall["fp"])
    recall = safe_div(overall["tp"], overall["tp"] + overall["fn"])
    f1 = (
        safe_div(2 * precision * recall, precision + recall)
        if precision == precision and recall == recall
        else float("nan")
    )

    log(f"overall  TP={overall['tp']:<5d} FP={overall['fp']:<5d} FN={overall['fn']:<5d}"
        f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    log("")
    log("recall by ground-truth size bucket (single-threshold recall - not the")
    log("multi-threshold official AR computed in the next section):")
    for b in ["small", "medium", "large"]:
        tp, fn = by_bucket[b]["tp"], by_bucket[b]["fn"]
        r = safe_div(tp, tp + fn)
        if r == r:
            log(f"  {b:>6s}: TP={tp:<5d} FN={fn:<5d} recall={r:.3f}")
        else:
            log(f"  {b:>6s}: no instances in val set")
    log("")

    return images_by_id, gt_by_image, preds_by_image


# ============================================================
# section 3: official COCO AP + AR via pycocotools directly, reusing the same
# coco_gt/coco_dt already loaded in main(). AP here should closely match
# metrics.json's latest row (same underlying library call) - a sanity check.
# AR is the actual new information Detectron2 never saves anywhere.
# ============================================================
def report_official_ap_ar(log, coco_gt, coco_dt):
    log("=" * 78)
    log("OFFICIAL COCO AP + AR  (pycocotools COCOeval, current checkpoint)")
    log("=" * 78)

    from pycocotools.cocoeval import COCOeval

    names = ["AP", "AP50", "AP75", "APs", "APm", "APl",
             "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"]

    for iou_type in ["bbox", "segm"]:
        log(f"\n--- {iou_type} ---")
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
        for name, value in zip(names, coco_eval.stats):
            log(f"  {name:>7s}: {value * 100:6.2f}")
    log("")


# ============================================================
# section 4: overlay visualizations, reusing data already loaded in section 2
# ============================================================
def write_visualizations(log, images_by_id, gt_by_image, preds_by_image):
    log("=" * 78)
    log("VISUALIZATIONS")
    log("=" * 78)

    os.makedirs(VIS_DIR, exist_ok=True)

    image_ids = list(images_by_id.keys())
    if ONLY_IMAGES_WITH_GT:
        image_ids = [i for i in image_ids if len(gt_by_image.get(i, [])) > 0]
    if MAX_IMAGES is not None:
        image_ids = image_ids[:MAX_IMAGES]

    n_written = 0
    for image_id in image_ids:
        img_info = images_by_id[image_id]
        img_path = Path(VAL_IMAGES_DIR) / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"WARNING: could not read {img_path}, skipping")
            continue

        for g in gt_by_image.get(image_id, []):
            gt_mask = polygon_to_mask(g["segmentation"], img_info["height"], img_info["width"])
            img = fill_mask(img, gt_mask, GT_COLOR)
            img = outline_mask(img, gt_mask, GT_COLOR)

        for pred in preds_by_image.get(image_id, []):
            pred_mask = rle_to_mask(pred["segmentation"])
            img = outline_mask(img, pred_mask, PRED_COLOR)
            x, y, _, _ = pred["bbox"]
            cv2.putText(
                img, f"{pred['score']:.2f}", (int(x), max(int(y) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA,
            )

        cv2.imwrite(str(Path(VIS_DIR) / img_info["file_name"]), img)
        n_written += 1

    log(f"wrote {n_written} visualizations to {VIS_DIR}")
    log(f"legend: green fill/outline = ground truth, red outline = prediction (score >= {SCORE_THRESH})")
    log("")


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    report_ap_history(log)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(VAL_JSON)
        coco_dt = coco_gt.loadRes(PREDICTIONS_JSON)

    images_by_id, gt_by_image, preds_by_image = report_precision_recall_f1(log, coco_gt, coco_dt)

    try:
        report_official_ap_ar(log, coco_gt, coco_dt)
    except Exception as e:
        log(f"Skipped official COCO AP/AR section due to error: {e}")
        log("")

    write_visualizations(log, images_by_id, gt_by_image, preds_by_image)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nfull report also written to {REPORT_PATH}")


if __name__ == "__main__":
    main()