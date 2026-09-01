"""Central paths and constants shared across the project."""

from pathlib import Path

# Repo root: src/aigc_detect/config.py -> aigc_detect -> src -> <repo root>
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_MANIFEST = PROCESSED_DIR / "train.csv"
# Union of train.csv and the data/train_ext/ generator-diverse slice. A SEPARATE
# manifest on purpose: adding those images to data/raw/ and re-running `split`
# would rewrite train.csv/val.csv in place, silently invalidating every cache
# keyed on their fingerprints. A new name means a new cache stem, so nothing
# already computed goes stale. See scripts/make_train_ext.py.
TRAIN_EXT_MANIFEST = PROCESSED_DIR / "train_ext.csv"
VAL_MANIFEST = PROCESSED_DIR / "val.csv"

# REAL-only corpora, concatenated onto training via --extra-train-manifest.
# Each keeps its own manifest, and therefore its own cache stem and fingerprint,
# so adding one never invalidates the base pool's cached embeddings.
#
# They exist because the training pool's real half is ImageNet and nothing else,
# and any real image from an absent domain gets mapped confidently into AIGC
# territory. Adding sid_real + unsplash_real took WildRF FPR@0.5 from .330 to
# .183 with no cost on ood.
SID_REAL_MANIFEST = PROCESSED_DIR / "sid_real.csv"          # SID_Set's real half (OpenImages)
UNSPLASH_REAL_MANIFEST = PROCESSED_DIR / "unsplash_real.csv"  # curated photography
PEXELS_REAL_MANIFEST = PROCESSED_DIR / "pexels_real.csv"      # curated photography
PHOTO_REAL_MANIFEST = PROCESSED_DIR / "photo_real.csv"        # union of the two above

# WildRF (arXiv:2406.09398): real + AI images scraped from Reddit, X and
# Facebook, carrying genuine platform re-encoding. Split in two on purpose --
# the reals are a TRAINING source, the test tier is EVAL ONLY, and they come
# from the same scrape, so a low FPR here after training on wildrf_real is an
# in-domain reading. The arm trained without it is the honest one.
# MODERN AI generators (2024-2025), added because our AIGC half is entirely
# GenImage-era 2022 and the remaining gap is diffusion, not GANs. Three separate
# publishers on purpose: one dataset is one provenance, and a detector will learn
# "this source" as readily as "this generator". DALLE3 is pulled to its own
# manifest and HELD OUT as the modern-diffusion eval tier -- never merge it in.
AIGC_EXT_DIR = DATA_DIR / "aigc_ext"
NANO_BANANA_MANIFEST = PROCESSED_DIR / "nano_banana.csv"        # Gemini 2.5 Flash Image
MIDJOURNEY_V6_MANIFEST = PROCESSED_DIR / "midjourney_v6.csv"
# SD3_MANIFEST -- REJECTED 2026-08-30, mislabelled real photos. Quarantined
# under data/quarantine/. See that README before ever re-adding this source.
DALLE3_HOLDOUT_MANIFEST = PROCESSED_DIR / "dalle3_holdout.csv"  # EVAL ONLY
AIGC_MODERN_MANIFEST = PROCESSED_DIR / "aigc_modern.csv"        # union, excludes DALLE3

REAL_EXT_DIR = DATA_DIR / "real_ext"
WILDRF_REAL_MANIFEST = PROCESSED_DIR / "wildrf_real.csv"
WILDRF_DIR = DATA_DIR / "wildrf"
WILDRF_TEST_MANIFEST = WILDRF_DIR / "wildrf_test.csv"

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
    # NOT a real source, despite the name and despite the upstream dataset card
    # calling it "Real human face sourced from the WhichFaceIsReal dataset".
    # whichfaceisreal.com shows an FFHQ photo BESIDE a StyleGAN fake; this HF
    # port ships only the fake half. Upstream's own label column agrees --
    # every sampled row is label=1 with names ['real','fake'] -- and the pixels
    # agree too (incoherent backgrounds, melted hair, blob artefacts). The card
    # prose is the only signal saying "real", and it loses 2-to-1.
    #
    # We had this as "real", which let build_ood's folder-name label inference
    # rewrite 250 StyleGAN faces to label=0. That single mapping produced the
    # entire "100% portrait FPR" finding: the model scored them >0.997 because
    # they ARE fake. Correcting it moved ood clean AUC 0.9670 -> 0.9971.
    "WhichFaceIsReal": "gan",
    # Real-image sources. Kept as distinct entries rather than one "Real" label
    # because pooling them is exactly how a 100% failure on one real
    # subpopulation stayed invisible behind a healthy aggregate (FINDINGS 2h).
    "Real": "real",                        # ImageNet, via Tiny-GenImage/GenImage
    "Real_OpenImages": "real",             # SID_Set's real half
    "Real_Unsplash": "real",               # curated photography
    "Real_Pexels": "real",                 # curated photography
    "Real_WildRF_train": "real",           # WildRF social-media reals (training)
    "Real_WildRF_reddit": "real",          # WildRF eval tier, per platform
    "Real_WildRF_twitter": "real",
    "Real_WildRF_facebook": "real",
    # WildRF's AI half: real-world social-media fakes, unknown provenance.
    "WildRF_reddit": "social",
    "WildRF_twitter": "social",
    "WildRF_facebook": "social",
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
