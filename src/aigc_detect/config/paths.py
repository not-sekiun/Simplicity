"""Every path the project reads or writes.

The directory layout encodes a four-tier separation that is a correctness
property, not organisation: what may be trained on, what may only be evaluated
on, and what is quarantined. The prose explaining *why* each corpus sits where
it does has moved to ``docs/data.md``; what stays here is the reason a path
exists at all, plus any rule a future change must not break.

``DATA_DIR`` comes from ``AIGC_DATA_ROOT`` (see :mod:`aigc_detect.config.settings`)
so the ~24 GB of imagery can live on another drive. Everything below is derived
from it, so pointing that variable elsewhere moves the whole tree coherently.
"""

from __future__ import annotations

from aigc_detect.config.settings import ROOT_DIR, get_settings

_settings = get_settings()

DATA_DIR = _settings.data_root
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Cached frozen-backbone pooled embeddings, one .npz per (backbone, manifest,
# view). NOTE: deliberately still under DATA_DIR rather than the configurable
# cache_root -- 274 caches are keyed to this location and the content-addressed
# store that replaces them will own cache_root. Moving it early would strand
# them.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# --- Training pool -----------------------------------------------------------

TRAIN_MANIFEST = PROCESSED_DIR / "train.csv"
VAL_MANIFEST = PROCESSED_DIR / "val.csv"

# Union of train.csv and the data/train_ext/ generator-diverse slice. A SEPARATE
# manifest on purpose: adding those images to data/raw/ and re-running `split`
# would rewrite train.csv/val.csv in place, silently invalidating every cache
# keyed on their fingerprints. A new name means a new cache stem, so nothing
# already computed goes stale.
TRAIN_EXT_MANIFEST = PROCESSED_DIR / "train_ext.csv"

# REAL-only corpora, concatenated onto training via --extra-train-manifest.
# Each keeps its own manifest, and therefore its own cache stem, so adding one
# never invalidates the base pool's cached embeddings.
#
# They exist because the training pool's real half is ImageNet and nothing else,
# and any real image from an absent domain gets mapped confidently into AIGC
# territory. Adding sid_real + unsplash_real took WildRF FPR@0.5 from .330 to
# .183 with no cost on ood.
SID_REAL_MANIFEST = PROCESSED_DIR / "sid_real.csv"            # SID_Set's real half (OpenImages)
UNSPLASH_REAL_MANIFEST = PROCESSED_DIR / "unsplash_real.csv"  # curated photography
PEXELS_REAL_MANIFEST = PROCESSED_DIR / "pexels_real.csv"      # curated photography
PHOTO_REAL_MANIFEST = PROCESSED_DIR / "photo_real.csv"        # union of the two above

# MODERN AI generators (2024-2025), added because our AIGC half is entirely
# GenImage-era 2022 and the remaining gap is diffusion, not GANs. Three separate
# publishers on purpose: one dataset is one provenance, and a detector will learn
# "this source" as readily as "this generator".
AIGC_EXT_DIR = DATA_DIR / "aigc_ext"
NANO_BANANA_MANIFEST = PROCESSED_DIR / "nano_banana.csv"      # Gemini 2.5 Flash Image
MIDJOURNEY_V6_MANIFEST = PROCESSED_DIR / "midjourney_v6.csv"
AIGC_MODERN_MANIFEST = PROCESSED_DIR / "aigc_modern.csv"      # union, EXCLUDES dalle3
# SD3 -- REJECTED 2026-08-30, mislabelled real photos. Quarantined under
# data/quarantine/. Read that README before ever re-adding this source.

# WildRF (arXiv:2406.09398): real + AI images scraped from Reddit, X and
# Facebook, carrying genuine platform re-encoding. Split in two on purpose --
# the reals are a TRAINING source, the test tier is EVAL ONLY, and they come
# from the same scrape, so a low FPR on the test tier after training on
# wildrf_real is an in-domain reading. The arm trained without it is the honest
# one.
REAL_EXT_DIR = DATA_DIR / "real_ext"
WILDRF_REAL_MANIFEST = PROCESSED_DIR / "wildrf_real.csv"

# --- Evaluation tiers: NEVER trained on --------------------------------------
#
# Each lives outside RAW_DIR, which is the only directory the split builder
# globs. That structural separation is what makes "never trained on" true by
# construction rather than by discipline.

WILDRF_DIR = DATA_DIR / "wildrf"
WILDRF_TEST_MANIFEST = WILDRF_DIR / "wildrf_test.csv"

# Held-out modern-diffusion tier. Pulled alongside nano_banana and
# midjourney_v6 and deliberately kept back, so "does this generalise to a
# modern generator we never saw" is a question we can actually answer.
DALLE3_HOLDOUT_MANIFEST = PROCESSED_DIR / "dalle3_holdout.csv"

# In-distribution held-out set: Tiny-GenImage's own HF "validation" split.
# NOT a cross-generator set -- it contains the same 7 generators as train.
HELDOUT_DIR = DATA_DIR / "heldout"
HELDOUT_MANIFEST = HELDOUT_DIR / "heldout.csv"

# The challenge brief's external, self-reported benchmark (5.4): COCO val2017
# plus WildFake "DALL-E Advanced". The brief says explicitly not to train on it.
DEMO_VAL_DIR = DATA_DIR / "demo_val"
DEMO_VAL_MANIFEST = DEMO_VAL_DIR / "demo_val.csv"

# The hard out-of-distribution tier, and the only one with room left to
# discriminate: 18 generator classes, TEN absent from training, five of those
# GAN families. Under the shipping head demo-val has 16 of 18 views at or above
# 0.99; this tier has none.
OOD_DIR = DATA_DIR / "ood"
OOD_MANIFEST = OOD_DIR / "ood.csv"

# --- Rejected corpora ---------------------------------------------------------

# Corpora withdrawn after an audit found them mislabelled. Kept with their
# evidence rather than deleted, because the finding is worth more than the disk.
QUARANTINE_DIR = DATA_DIR / "quarantine"

__all__ = [
    "AIGC_EXT_DIR",
    "AIGC_MODERN_MANIFEST",
    "DALLE3_HOLDOUT_MANIFEST",
    "DATA_DIR",
    "DEMO_VAL_DIR",
    "DEMO_VAL_MANIFEST",
    "EMBEDDINGS_DIR",
    "HELDOUT_DIR",
    "HELDOUT_MANIFEST",
    "MIDJOURNEY_V6_MANIFEST",
    "NANO_BANANA_MANIFEST",
    "OOD_DIR",
    "OOD_MANIFEST",
    "PEXELS_REAL_MANIFEST",
    "PHOTO_REAL_MANIFEST",
    "PROCESSED_DIR",
    "QUARANTINE_DIR",
    "RAW_DIR",
    "REAL_EXT_DIR",
    "ROOT_DIR",
    "SID_REAL_MANIFEST",
    "TRAIN_EXT_MANIFEST",
    "TRAIN_MANIFEST",
    "UNSPLASH_REAL_MANIFEST",
    "VAL_MANIFEST",
    "WILDRF_DIR",
    "WILDRF_REAL_MANIFEST",
    "WILDRF_TEST_MANIFEST",
]
