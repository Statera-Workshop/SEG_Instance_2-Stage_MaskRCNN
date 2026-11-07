"""Evaluate seg mask_rcnn_km (KITTI-MOTS) on the test split.

    UNIFIED_SEG_DATASET=kittimots UNIFIED_IMAGE_H=384 python evaluate_seg_mask_rcnn_km.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from common.seg.init_from_det import with_init
from detectors.mask_rcnn.build import build_model

if __name__ == "__main__":
    main("evaluate", task="seg", version="mask_rcnn_km", build_model=with_init(build_model, "mask_rcnn", "faster_rcnn"))
