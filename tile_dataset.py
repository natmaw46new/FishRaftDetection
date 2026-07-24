import os
import json
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.affinity import scale as affine_scale
from shapely.ops import unary_union
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
TIF_DIR = "./data/tifs"
SHP_PATH = "./data/rafts.shp"
OUTPUT_DIR = "./data/rotation_augment+newsplit/tiles"

TILE_SIZE = 2048        # native-resolution capture size - sized for full containment
                         # of the largest known raft (~600px at this GSD)
OVERLAP = 1024           # ~1.7x margin over the 600px max raft; see margin/redundancy notes
STRIDE = TILE_SIZE - OVERLAP

OUTPUT_TILE_SIZE = 1024  # actual size saved to disk / fed to the network - keeps
                         # Detectron2 input resolution (and its compute cost) unchanged

MIN_VISIBLE_FRAC = 0.8
CATEGORY_NAME = "fish_raft"
CATEGORY_ID = 1
TILE_IMAGE_FORMAT = "png"

BACKGROUND_TILE_RATIO = 1.0  # background tiles sampled per raft-adjacent tile
RANDOM_SEED = 42


# ============================================================
# Footprint index — header-only reads, instant regardless of file size
# ============================================================
def build_tif_index(tif_dir):
    paths = sorted(Path(tif_dir).glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {tif_dir}")

    records = []
    ref_crs = None
    ref_gsd = None
    for p in paths:
        with rasterio.open(p) as src:
            if ref_crs is None:
                ref_crs = src.crs
                ref_gsd = abs(src.transform.a)
            records.append({"path": str(p), "geometry": box(*src.bounds)})

    index = gpd.GeoDataFrame(records, crs=ref_crs)
    print(f"indexed {len(index)} tifs | crs={ref_crs} | gsd~={ref_gsd:.4f} units/px")
    return index, ref_crs, ref_gsd


# ============================================================
# Labels — reproject to raster CRS, explode multiparts to one row per instance
# ============================================================
def load_labels(shp_path, target_crs, buffer_dist):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf["raft_id"] = gdf.index

    region_of_interest = unary_union(gdf.geometry).buffer(buffer_dist)
    return gdf, region_of_interest


# ============================================================
# Tile grid — restricted to tiles intersecting the buffered label region
# ============================================================
def generate_tile_bboxes(tif_index, region_of_interest, tile_size_px, stride_px, gsd):
    tile_size_units = tile_size_px * gsd
    stride_units = stride_px * gsd

    minx, miny, maxx, maxy = tif_index.total_bounds
    bboxes = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile_box = box(x, y, x + tile_size_units, y + tile_size_units)
            if tile_box.intersects(region_of_interest):
                bboxes.append((x, y, x + tile_size_units, y + tile_size_units))
            y += stride_units
        x += stride_units

    print(f"generated {len(bboxes)} candidate tiles intersecting the label region")
    return bboxes


# ============================================================
# Random background sampling - uniform draws from anywhere in the survey
# that ISN'T already covered by the raft-adjacent tiles above
# ============================================================
def sample_background_tiles(tif_index, region_of_interest, n_samples, tile_size_px, gsd, seed=RANDOM_SEED):
    rng = random.Random(seed)
    tile_size_units = tile_size_px * gsd
    minx, miny, maxx, maxy = tif_index.total_bounds

    bboxes = []
    attempts = 0
    max_attempts = n_samples * 50  # safety valve in case the region excludes too much

    while len(bboxes) < n_samples and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(minx, maxx - tile_size_units)
        y = rng.uniform(miny, maxy - tile_size_units)
        tile_box = box(x, y, x + tile_size_units, y + tile_size_units)

        if tile_box.intersects(region_of_interest):
            continue  # too close to a labeled raft - already covered by the positive pass
        if not tif_index.intersects(tile_box).any():
            continue  # falls in a gap with no actual imagery

        bboxes.append((x, y, x + tile_size_units, y + tile_size_units))

    print(f"sampled {len(bboxes)} random background tiles ({attempts} attempts)")
    return bboxes


# ============================================================
# Pixel extraction — single-file window read, or merge only when a tile
# straddles a boundary between two source tifs
# ============================================================
def extract_tile_pixels(bbox, tif_index, gsd):
    tile_geom = box(*bbox)
    hits = tif_index[tif_index.intersects(tile_geom)]
    if hits.empty:
        return None, None

    if len(hits) == 1:
        path = hits.iloc[0]["path"]
        with rasterio.open(path) as src:
            window = src.window(*bbox)
            data = src.read(window=window, boundless=True, fill_value=0)
            transform = src.window_transform(window)
    else:
        datasets = [rasterio.open(p) for p in hits["path"]]
        try:
            data, transform = merge(datasets, bounds=bbox, res=(gsd, gsd))
        finally:
            for ds in datasets:
                ds.close()

    return data, transform


# ============================================================
# Label clipping — drop instances too truncated by the tile edge to keep
# ============================================================
def world_geom_to_pixels(geom, transform):
    inv = ~transform
    if geom.geom_type == "Polygon":
        parts = [geom]
    elif geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type == "Polygon"]
    else:
        parts = []

    pixel_polys = []
    for part in parts:
        if part.is_empty:
            continue
        exterior_px = [inv * (x, y) for x, y in part.exterior.coords]
        pixel_polys.append(Polygon(exterior_px))
    return pixel_polys


def clip_polygons_to_tile(gdf, bbox, transform, min_visible_frac):
    tile_geom = box(*bbox)
    kept = []

    candidates = gdf[gdf.intersects(tile_geom)]
    for _, row in candidates.iterrows():
        original = row.geometry
        clipped = original.intersection(tile_geom)
        if clipped.is_empty:
            continue

        visible_frac = clipped.area / original.area
        if visible_frac < min_visible_frac:
            continue

        for poly in world_geom_to_pixels(clipped, transform):
            if poly.is_valid and poly.area > 0:
                kept.append({"raft_id": row["raft_id"], "polygon": poly})

    return kept


# ============================================================
# COCO helpers
# ============================================================
def polygon_to_coco_segmentation(polygon):
    coords = list(polygon.exterior.coords)
    flat = [c for xy_pair in coords for c in xy_pair]
    return [flat]


def polygon_to_coco_bbox(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return [minx, miny, maxx - minx, maxy - miny]


# ============================================================
# Main tiling loop
# ============================================================
def main():
    images_dir = Path(OUTPUT_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tif_index, raster_crs, gsd = build_tif_index(TIF_DIR)
    buffer_units = OVERLAP * gsd
    labels, region_of_interest = load_labels(SHP_PATH, raster_crs, buffer_units)
    raft_bboxes = generate_tile_bboxes(tif_index, region_of_interest, TILE_SIZE, STRIDE, gsd)

    n_background = round(len(raft_bboxes) * BACKGROUND_TILE_RATIO)
    background_bboxes = sample_background_tiles(tif_index, region_of_interest, n_background, TILE_SIZE, gsd)

    bboxes = raft_bboxes + background_bboxes
    print(f"total tiles to process: {len(bboxes)} ({len(raft_bboxes)} raft-adjacent + {len(background_bboxes)} background)")

    resize_scale = OUTPUT_TILE_SIZE / TILE_SIZE  # e.g. 0.5 for a 2048 -> 1024 crop

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }

    image_id = 0
    annotation_id = 0

    for bbox in bboxes:
        data, transform = extract_tile_pixels(bbox, tif_index, gsd)
        if data is None:
            continue

        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            rgb = np.transpose(np.repeat(data[:1], 3, axis=0), (1, 2, 0))

        if rgb.shape[0] != TILE_SIZE or rgb.shape[1] != TILE_SIZE:
            padded = np.zeros((TILE_SIZE, TILE_SIZE, rgb.shape[2]), dtype=rgb.dtype)
            padded[: rgb.shape[0], : rgb.shape[1], :] = rgb
            rgb = padded

        instances = clip_polygons_to_tile(labels, bbox, transform, MIN_VISIBLE_FRAC)

        # captured at native TILE_SIZE for containment margin, then downsized to
        # OUTPUT_TILE_SIZE for the network - polygons get the same scale factor
        # applied so they stay aligned with what's actually saved to disk
        tile_image = Image.fromarray(rgb.astype(np.uint8)).resize(
            (OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE), Image.LANCZOS
        )
        file_name = f"tile_{image_id:06d}.{TILE_IMAGE_FORMAT}"
        tile_image.save(images_dir / file_name)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": OUTPUT_TILE_SIZE,
                "height": OUTPUT_TILE_SIZE,
                "geo_bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            }
        )

        for inst in instances:
            poly = affine_scale(inst["polygon"], xfact=resize_scale, yfact=resize_scale, origin=(0, 0))
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "segmentation": polygon_to_coco_segmentation(poly),
                    "bbox": polygon_to_coco_bbox(poly),
                    "area": poly.area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    with open(Path(OUTPUT_DIR) / "annotations.json", "w") as f:
        json.dump(coco, f)

    print(
        f"wrote {image_id} tiles ({TILE_SIZE}px captured -> {OUTPUT_TILE_SIZE}px saved) "
        f"and {annotation_id} raft instances to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()