"""Central paths and constants shared across the project."""

from pathlib import Path

# Repo root: src/aigc_detect/config.py -> aigc_detect -> src -> <repo root>
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_MANIFEST = PROCESSED_DIR / "train.csv"
VAL_MANIFEST = PROCESSED_DIR / "val.csv"

# Three-tier data separation:
#   raw/      = training pool (scripts/make_splits.py globs ONLY this dir)
#   heldout/  = untouched IN-DISTRIBUTION test set, NEVER trained on and NEVER
#               globbed by make_splits.py. Currently Tiny-GenImage's own HF
#               "validation" split. NOTE: this is NOT a cross-generator set --
#               it contains the same 7 generators as the train split (ADM,
#               BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong). For unseen-
#               generator evaluation use `main.py split --holdout-generators`.
#   demo_val/ = the challenge brief's external, self-reported benchmark (5.4),
#               also never trained on -- see below
HELDOUT_DIR = DATA_DIR / "heldout"
HELDOUT_MANIFEST = HELDOUT_DIR / "heldout.csv"

# Self-reported "demonstration purposes only" benchmark (challenge brief 5.4):
# COCO val2017 (real) + WildFake "DALL·E Advanced" subset (AIGC). Kept under a
# directory separate from RAW_DIR/PROCESSED_DIR on purpose — the brief says
# explicitly not to train on it, so it must never be picked up by
# scripts/make_splits.py (which only globs RAW_DIR).
DEMO_VAL_DIR = DATA_DIR / "demo_val"
DEMO_VAL_MANIFEST = DEMO_VAL_DIR / "demo_val.csv"

# Cached frozen-backbone pooled embeddings (Wave 2: "Simplicity Prevails"
# recipe). One .npz per (backbone, manifest) pair -- see aigc_detect/embed.py.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Canonical labels used everywhere in the pipeline.
LABEL_REAL = 0
LABEL_AIGC = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_AIGC: "aigc"}

# Model input resolution and normalization (ImageNet stats — matches most
# torchvision/timm backbones under the 2B-parameter limit).
IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Train/val split.
VAL_FRACTION = 0.15
RANDOM_SEED = 42
