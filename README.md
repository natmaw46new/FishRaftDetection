# Fish Raft Segmentation Pipeline

Instance segmentation of fish rafts from georeferenced orthomosaic imagery, using
Detectron2 (Mask R-CNN). This covers the pipeline from raw `.tif` + `.shp` inputs
through a trained model.

## Directory layout

Everything under `./data` is mounted into the container at `/app/data` /
`/app/data/output` (the whole repo is mounted as `.:/app`, so relative paths in
the scripts and absolute `/app/...` paths refer to the same place).

```
data/
  tifs/                      # input orthomosaic tiles (.tif)
  rafts.shp                  # input raft label polygons
  output/
    tiles/
      images/                # all tiles written by tile_dataset.py
      annotations.json       # COCO annotations for all tiles (pre-split)
      instances_train.json   # written by train_val_split.py
      instances_val.json     # written by train_val_split.py
      train/images/          # symlinks into tiles/images, train subset
      val/images/            # symlinks into tiles/images, val subset
    model/                   # Detectron2 checkpoints + config.yaml
```

## Running it

```bash
docker compose up --build -d
docker exec -it fish-raft-dev bash

# inside the container:
python tile_dataset.py
python train_val_split.py
python train_detectron2.py
```

GPU is enabled via the `deploy.resources.reservations.devices` block in
`docker-compose.yml` — this requires the NVIDIA Container Toolkit installed on
the host. The image itself stays on a plain `ubuntu:22.04` base; `torch`/
`torchvision` are installed from the `cu124` wheel index, which already bundles
the CUDA runtime it needs, so a dedicated `nvidia/cuda` base image isn't
required.

## Why the pipeline is built this way

### Tiling: bigger capture window than what's fed to the network

The largest known raft is about 600px wide at this imagery's GSD (1:5000 scale).
Tiles are captured at `TILE_SIZE = 2048px` — over 3x the largest raft — with
`OVERLAP = 1024px` between adjacent tiles (50% stride), so a raft sitting near a
tile boundary is still fully contained in at least one tile rather than being
split across two. `MIN_VISIBLE_FRAC = 0.8` then drops any label that's still
more than 20% cut off even after that margin, instead of training on
badly-truncated partial instances.

The captured 2048px tile is then downsized to `OUTPUT_TILE_SIZE = 1024px`
before it's written to disk and fed to the network. This keeps Detectron2's
input resolution — and its compute/memory cost per image — unchanged from a
straightforward 1024px tiling scheme, while still getting the larger
containment margin above. Raft polygons are scaled by the same
`resize_scale = OUTPUT_TILE_SIZE / TILE_SIZE` factor so annotations stay
pixel-aligned with the resized image actually saved to disk.

### Tile selection: not just "tiles containing a raft"

`generate_tile_bboxes` doesn't only keep tiles that intersect a labeled raft —
it keeps every tile intersecting the raft region **buffered by the tile
overlap distance**. That buffer matters for the same reason as the tiling
margin above: a raft near the edge of its "home" tile might only be fully
contained in a neighboring tile, so the neighbor needs to be a candidate too,
even though its center isn't near any label.

Separately, `sample_background_tiles` draws random tiles from anywhere in the
survey that does **not** intersect the (buffered) raft region, at
`BACKGROUND_TILE_RATIO = 4.0` background tiles per raft-adjacent tile. Without
negative examples, the model never sees "empty water" and has no signal for
what *isn't* a raft — background tiles teach it that.

### Keeping `geo_bbox` on every tile

Each tile written by `tile_dataset.py` stores its source `geo_bbox` (world
coordinates) in the COCO `images` entry. `train_val_split.py` uses this to
group tiles into spatial blocks (`BLOCK_SIZE_METERS = 5000`, well above the
tile stride so overlapping tiles always land in the same block) and splits
train/val **by block, not by individual tile**.

This matters because adjacent/overlapping tiles are highly correlated — if
train/val were split tile-by-tile, near-duplicate crops of the same raft could
end up on both sides of the split, and validation would silently overstate
performance. Splitting by spatial block keeps a given raft (and its
overlapping tiles) entirely on one side. The split is also stratified: raft-
containing blocks and background-only blocks are shuffled and split
separately (`VAL_FRACTION = 0.2` each), so val isn't accidentally all-positive
or all-background.

### Train/val folders as symlinks, not copies

`train_val_split.py` creates `train/images/` and `val/images/` as symlinks
into the single `tiles/images/` folder rather than copying files. Detectron2's
`register_coco_instances` needs one images directory per registered dataset,
but the underlying tiles only need to exist on disk once — a raft tile in the
train split and the same raft's neighboring background tile in val (say)
don't need two physical copies of the same pixels.

### Training config

- COCO-pretrained `mask_rcnn_R_50_FPN_3x` backbone, fine-tuned to `NUM_CLASSES = 1`
  (fish_raft only).
- `MIN_SIZE_TRAIN` uses a scale-jitter range (768–1280px) centered on the
  1024px tile size, as a hedge against the resolution gap between this
  imagery's native scale and whatever scale future imagery arrives at.
- `MIN_SIZE_TEST`/`MAX_SIZE_TEST` are pinned to 1024, matching the actual
  tile size, since eval/inference shouldn't jitter.
