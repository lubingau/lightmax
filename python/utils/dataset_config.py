"""
dataset_config.py

Centralized dataset cache paths for the saturation evaluation pipeline.
All scripts should import this module to resolve local dataset locations.
"""

import os
from pathlib import Path


# Root datasets directory
DATASETS_ROOT = Path("/media/lubin/Datasets")

# HuggingFace dataset cache (for load_dataset)
HF_DATASETS_CACHE = DATASETS_ROOT / "huggingface" / "datasets"

# HuggingFace hub cache (for AutoModel.from_pretrained)
HF_HUB_CACHE = DATASETS_ROOT / "huggingface" / "hub"

# Aeon data dir (for UCR time series)
AEON_DATA_DIR = DATASETS_ROOT / "aeon"

# ImageNet validation directory
IMAGENET_DIR = DATASETS_ROOT / "Imagenet" / "val"

# COCO annotation/images directory
COCO_DIR = DATASETS_ROOT / "Coco" / "coco"

# ---------------------------------------------------------------
# Apply these paths to environment variables so that underlying
# libraries (datasets, aeon, huggingface_hub) pick them up.
# ---------------------------------------------------------------
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("AEON_DATA", str(AEON_DATA_DIR))
