import json
import random
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ANNOTATIONS_PATH = "/app/outputs/tiles/annotations.json"
OUTPUT_DIR = "/app/outputs/train_val_split"

BLOCK_SIZE_METERS = 5000  # spatial grouping size - well above tile stride so
                          # overlapping tiles always land in the same block
VAL_FRACTION = 0.2
RANDOM_SEED = 42


def block_id_for_bbox(bbox, block_size):
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    return (int(cx // block_size), int(cy // block_size))


def main():
    with open(ANNOTATIONS_PATH) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    block_images = defaultdict(list)
    block_has_raft = defaultdict(bool)
    for img in coco["images"]:
        if "geo_bbox" not in img:
            raise ValueError(
                f"image {img['file_name']} has no geo_bbox - rerun tile_dataset.py "
                "(updated version) before splitting"
            )
        bid = block_id_for_bbox(img["geo_bbox"], BLOCK_SIZE_METERS)
        block_images[bid].append(img["id"])
        if len(anns_by_image[img["id"]]) > 0:
            block_has_raft[bid] = True

    positive_blocks = [b for b in block_images if block_has_raft[b]]
    background_blocks = [b for b in block_images if not block_has_raft[b]]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(positive_blocks)
    rng.shuffle(background_blocks)

    n_val_pos = max(1, round(len(positive_blocks) * VAL_FRACTION)) if positive_blocks else 0
    n_val_bg = round(len(background_blocks) * VAL_FRACTION)

    val_blocks = set(positive_blocks[:n_val_pos]) | set(background_blocks[:n_val_bg])

    train_image_ids, val_image_ids = [], []
    for bid, img_ids in block_images.items():
        target = val_image_ids if bid in val_blocks else train_image_ids
        target.extend(img_ids)

    def build_split(image_ids):
        images = [images_by_id[i] for i in image_ids]
        annotations = [a for i in image_ids for a in anns_by_image[i]]
        return {"images": images, "annotations": annotations, "categories": coco["categories"]}

    train_coco = build_split(train_image_ids)
    val_coco = build_split(val_image_ids)

    with open(Path(OUTPUT_DIR) / "instances_train.json", "w") as f:
        json.dump(train_coco, f)
    with open(Path(OUTPUT_DIR) / "instances_val.json", "w") as f:
        json.dump(val_coco, f)

    print(f"blocks: {len(positive_blocks)} raft-containing, {len(background_blocks)} background-only")
    print(f"val blocks selected: {len(val_blocks)} ({n_val_pos} raft-containing + {n_val_bg} background-only)")
    print(f"train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} instances")
    print(f"val:   {len(val_coco['images'])} images, {len(val_coco['annotations'])} instances")


if __name__ == "__main__":
    main()
