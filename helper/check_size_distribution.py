import json

TRAIN_JSON = "./data/output/tiles/instances_train.json"
VAL_JSON = "./data/output/tiles/instances_val.json"

# Same thresholds COCOEvaluator uses, in px^2, on the same 1024px tile pixel
# space your annotations are already stored in.
SMALL_MAX = 32 ** 2   # < 1024 px^2
MEDIUM_MAX = 96 ** 2  # < 9216 px^2


def bucket(area):
    if area < SMALL_MAX:
        return "small"
    elif area < MEDIUM_MAX:
        return "medium"
    return "large"


def report(name, path):
    with open(path) as f:
        coco = json.load(f)

    counts = {"small": 0, "medium": 0, "large": 0}
    areas = []
    for ann in coco["annotations"]:
        area = ann.get("area")
        if area is None:
            _, _, w, h = ann["bbox"]
            area = w * h
        areas.append(area)
        counts[bucket(area)] += 1

    total = len(areas)
    print(f"\n{name}: {total} instances")
    for k, v in counts.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:7s}: {v:5d} ({pct:5.1f}%)")
    if areas:
        print(f"  area range: {min(areas):.0f} - {max(areas):.0f} px^2")
        print(f"  largest instance side (~sqrt(area)): {max(areas) ** 0.5:.0f}px")


if __name__ == "__main__":
    report("TRAIN", TRAIN_JSON)
    report("VAL", VAL_JSON)
