import json
import os
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
JSON_PATH = "./data/rotation_augment+newsplit/tiles/instances_train.json"
IMAGES_DIR = "./data/rotation_augment+newsplit/tiles/images"  # full tile pool; train/val symlinks both point here
OUTPUT_DIR = "./data/rotation_augment+newsplit/smallest_instances"

N_SMALLEST = 20    # how many of the tiniest instances to crop out for visual inspection
CROP_PADDING = 25  # px of surrounding context in each saved crop
ZOOM = 4           # upscale factor so tiny instances are actually visible

# finer bins than COCO's 3 buckets, by approximate side length (sqrt(area)) in px
SIDE_BIN_EDGES = [0, 8, 16, 24, 32, 48, 64, 96, 128, 200]


def main():
    with open(JSON_PATH) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    records = []
    for ann in coco["annotations"]:
        area = ann.get("area")
        if area is None:
            _, _, w, h = ann["bbox"]
            area = w * h
        records.append({"ann": ann, "area": area, "side": area ** 0.5})

    records.sort(key=lambda r: r["area"])

    counts = [0] * (len(SIDE_BIN_EDGES) - 1)
    for r in records:
        idx = None
        for i in range(len(SIDE_BIN_EDGES) - 1):
            if SIDE_BIN_EDGES[i] <= r["side"] < SIDE_BIN_EDGES[i + 1]:
                idx = i
                break
        counts[idx if idx is not None else -1] += 1

    print(f"{len(records)} total instances in {JSON_PATH}\n")
    print("side length (px, ~sqrt(area)) histogram:")
    for i, c in enumerate(counts):
        lo, hi = SIDE_BIN_EDGES[i], SIDE_BIN_EDGES[i + 1]
        pct = 100 * c / len(records) if records else 0
        print(f"  [{lo:>3d}-{hi:<3d}) px: {c:5d} ({pct:5.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_written = 0
    for rank, r in enumerate(records[:N_SMALLEST]):
        ann = r["ann"]
        img_info = images_by_id[ann["image_id"]]
        img_path = Path(IMAGES_DIR) / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        x, y, w, h = ann["bbox"]
        x0 = max(int(x - CROP_PADDING), 0)
        y0 = max(int(y - CROP_PADDING), 0)
        x1 = min(int(x + w + CROP_PADDING), img.shape[1])
        y1 = min(int(y + h + CROP_PADDING), img.shape[0])
        crop = img[y0:y1, x0:x1].copy()

        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] -= x0
            pts[:, 1] -= y0
            cv2.polylines(crop, [pts.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)

        crop = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        out_name = f"{rank:02d}_area{r['area']:.0f}px2_side{r['side']:.0f}px_{img_info['file_name']}"
        cv2.imwrite(str(Path(OUTPUT_DIR) / out_name), crop)
        n_written += 1

    print(f"\nwrote {n_written} zoomed ({ZOOM}x) crops of the smallest instances to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
