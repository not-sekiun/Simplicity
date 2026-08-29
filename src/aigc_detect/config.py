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

# Fourth tier: a deliberately HARD out-of-distribution benchmark, evaluation
# only, never trained on and never globbed by make_splits.py.
#
# Exists because our first three evaluation sets stopped discriminating. Under
# the Run 6 head, demo-val has 16 of 18 grid views at or above 0.99 (11 at or
# above 0.999) and val has 11 of 18 -- so a backbone race on them would be
# comparing numbers inside their own standard error. See NARRATIVE.md.
#
# TheKernel01/AIGC-Detection-Benchmark (ungated, Apache-2.0, parquet, test
# split only, 125,026 images / 32GB, streamed and capped here). Same publisher
# as Tiny-GenImage, so the streaming path is already proven.
#
# What makes it hard: 18 generator classes, of which TEN are absent from our
# training pool -- CycleGAN, DALL-E 2, GauGAN, ProGAN, SDXL, StarGAN, StyleGAN,
# StyleGAN2, SD14, WhichFaceIsReal. Five of those are GAN families, which are
# architecturally unlike the mostly-diffusion generators the head trained on.
# This is the unseen-generator test the project has never actually run.
OOD_DIR = DATA_DIR / "ood"
OOD_MANIFEST = OOD_DIR / "ood.csv"

# Generator architecture families, for reporting the OOD grid broken down by
# family rather than as one number.
#
# This split is load-bearing for scoping: the competition is expected to use
# DIFFUSION generators, so a collapse confined to the GAN families is out of
# scope and must not be allowed to drag the headline down or drive a backbone
# choice. Reporting one pooled OOD number would hide exactly that distinction.
#
# Orthogonal to seen/unseen: BigGAN is a GAN that IS in our training pool,
# while SD14/SDXL/DALLE2 are diffusion models that are NOT. Always report both
# axes -- family answers "is this in scope", seen/unseen answers "is this
# generalization".
GENERATOR_FAMILY = {
    # Diffusion / autoregressive-diffusion -- the in-scope families.
    "ADM": "diffusion",
    "DALLE2": "diffusion",
    "GLIDE": "diffusion",
    "Midjourney": "diffusion",
    "SD14": "diffusion",
    "SD15": "diffusion",
    "SDXL": "diffusion",
    "VQDM": "diffusion",
    "Wukong": "diffusion",
    # GANs -- treated as out of scope for the competition, reported separately.
    "BigGAN": "gan",
    "CycleGAN": "gan",
    "GauGAN": "gan",
    "ProGAN": "gan",
    "StarGAN": "gan",
    "StyleGAN": "gan",
    "StyleGAN2": "gan",
    # Real-image sources.
    "Real": "real",
    "WhichFaceIsReal": "real",
}

# Generators present in data/raw/ (the training pool).
TRAIN_GENERATORS = frozenset({"Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"})

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
