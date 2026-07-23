import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
ANNOTATIONS_PATH = "./data/1-to-1-ratiobackground/tiles/annotations.json"
IMAGES_DIR = "./data/1-to-1-ratiobackground/tiles/images"
OUTPUT_DIR = "./data/1-to-1-ratiobackground/tiles"

BLOCK_SIZE_METERS = 5000  # spatial grouping size - well above tile stride so
                          # overlapping tiles always land in the same block
VAL_FRACTION = 0.2
RANDOM_SEED = 42

# creates train/images and val/images with symlinks (not copies) into IMAGES_DIR,
# matching the conventional COCO folder layout without duplicating disk space
CREATE_SPLIT_FOLDERS = True


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
    skipped = []

    for img in coco["images"]:
        if "geo_bbox" not in img or img["geo_bbox"] is None:
            skipped.append(img["file_name"])
            continue

        bid = block_id_for_bbox(img["geo_bbox"], BLOCK_SIZE_METERS)
        block_images[bid].append(img["id"])
        if len(anns_by_image[img["id"]]) > 0:
            block_has_raft[bid] = True

    if skipped:
        print(
            f"WARNING: {len(skipped)} images have no geo_bbox and were skipped entirely "
            f"(not written to either split). This usually means annotations.json was "
            f"generated before tile_dataset.py recorded geo_bbox - rerun tiling and this "
            f"list should be empty."
        )
        for name in skipped[:10]:
            print("  -", name)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

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

    def report(name, split_coco, image_ids):
        n_images = len(split_coco["images"])
        n_positive = sum(1 for i in image_ids if len(anns_by_image[i]) > 0)
        n_background = n_images - n_positive
        print(
            f"{name}: {n_images} images ({n_positive} with rafts, "
            f"{n_background} background-only), {len(split_coco['annotations'])} instances"
        )

    report("train", train_coco, train_image_ids)
    report("val", val_coco, val_image_ids)

    total_written = len(train_image_ids) + len(val_image_ids)
    print(
        f"total images written across both splits: {total_written} / {len(coco['images'])} "
        f"input images ({len(skipped)} skipped due to missing geo_bbox)"
    )

    if CREATE_SPLIT_FOLDERS:
        create_symlink_folder("train", train_coco["images"])
        create_symlink_folder("val", val_coco["images"])


def create_symlink_folder(split_name, images):
    dest_dir = Path(OUTPUT_DIR) / split_name / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    created, skipped_existing = 0, 0
    for img in images:
        src = Path(IMAGES_DIR) / img["file_name"]
        dst = dest_dir / img["file_name"]
        if dst.exists() or dst.is_symlink():
            skipped_existing += 1
            continue
        os.symlink(src.resolve(), dst)
        created += 1

    print(f"{split_name}/images: {created} symlinks created ({skipped_existing} already existed) -> {dest_dir}")


if __name__ == "__main__":
    main()