import os

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# ============================================================
# CONFIG
# ============================================================
TRAIN_DIR = "./data/1-to-1-ratiobackground/tiles/train"
VAL_DIR = "./data/1-to-1-ratiobackground/tiles/val"

TRAIN_JSON = "./data/1-to-1-ratiobackground/tiles/instances_train.json"
VAL_JSON = "./data/1-to-1-ratiobackground/tiles/instances_val.json"
OUTPUT_DIR = "./data/1-to-1-ratiobackground/model"

BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES = 1  # fish_raft only

IMS_PER_BATCH = 8
BASE_LR = 0.001
MAX_ITER = 5000
CHECKPOINT_PERIOD = 500
EVAL_PERIOD = 500
NUM_WORKERS = 4

# Target "presence frequency" for the repeat-factor sampler below: roughly the
# share of sampled images per epoch that should contain at least one fish_raft.
# Raise this closer to 1.0 for more aggressive upsampling of raft tiles, lower
# it toward 0 to fall back closer to plain random sampling.
REPEAT_THRESHOLD = 0.3

# scale-jitter range for training - centered on the 1024px tile size, a hedge
# against the 1:5000 / 1:1000 resolution domain gap discussed earlier
MIN_SIZE_TRAIN_CHOICES = (768, 896, 1024, 1152, 1280)
MAX_SIZE_TRAIN = 1333
MIN_SIZE_TEST = 1024
MAX_SIZE_TEST = 1024


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def main():
    setup_logger()

    register_coco_instances("fish_raft_train", {}, TRAIN_JSON, os.path.join(TRAIN_DIR, "images"))
    register_coco_instances("fish_raft_val", {}, VAL_JSON, os.path.join(VAL_DIR, "images"))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.DATASETS.TRAIN = ("fish_raft_train",)
    cfg.DATASETS.TEST = ("fish_raft_val",)
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    # Detectron2 defaults to True here, which drops every image with 0 annotations
    # from training — that's exactly the background tiles tile_dataset.py sampled
    # on purpose as negatives. Keep them.
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # With only one category, the "category frequency" this sampler balances on
    # collapses to "fraction of images that contain a fish_raft at all." This
    # upsamples raft-containing tiles relative to background-only ones, purely
    # via sampling weights — no tiles are duplicated on disk or dropped. It
    # recalculates automatically from whatever's actually in TRAIN_JSON, so it
    # keeps working correctly even after you change BACKGROUND_TILE_RATIO.
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = REPEAT_THRESHOLD

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE_CONFIG)  # COCO-pretrained backbone
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.STEPS = (int(MAX_ITER * 0.6), int(MAX_ITER * 0.85))
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD

    cfg.INPUT.MIN_SIZE_TRAIN = MIN_SIZE_TRAIN_CHOICES
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = MAX_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TEST = MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST = MAX_SIZE_TEST

    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()