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

# Since Tier 5 a corpus is a directory under CORPORA_DIR holding images/,
# index.csv and corpus.yaml, and a manifest is a recipe under MANIFESTS_DIR
# resolved into MANIFESTS_RESOLVED_DIR. RAW_DIR and PROCESSED_DIR survive only
# for the download scripts that still write into them; Tier 6 retires both.
CORPORA_DIR = DATA_DIR / "corpora"
MANIFESTS_DIR = DATA_DIR / "manifests"
MANIFESTS_RESOLVED_DIR = MANIFESTS_DIR / "resolved"


def _manifest(name: str):
    """Where a resolved manifest lives.

    Every constant below used to name a hand-built CSV in whichever directory
    the script that wrote it happened to use -- five directories for ten files.
    They are all one recipe away now, and the constants survive so no importer
    had to change.
    """
    return MANIFESTS_RESOLVED_DIR / f"{name}.csv"

# Cached frozen-backbone pooled embeddings, one .npz per (backbone, manifest,
# view). NOTE: deliberately still under DATA_DIR rather than the configurable
# cache_root -- 274 caches are keyed to this location and the content-addressed
# store that replaces them will own cache_root. Moving it early would strand
# them.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# --- Training pool -----------------------------------------------------------

TRAIN_MANIFEST = _manifest("train")
VAL_MANIFEST = _manifest("val")

# Union of train.csv and the data/train_ext/ generator-diverse slice. A SEPARATE
# manifest on purpose: adding those images to data/raw/ and re-running `split`
# would rewrite train.csv/val.csv in place, silently invalidating every cache
# keyed on their fingerprints. A new name means a new cache stem, so nothing
# already computed goes stale.
TRAIN_EXT_MANIFEST = _manifest("train_ext")

# REAL-only corpora, concatenated onto training via --extra-train-manifest.
# Each keeps its own manifest, and therefore its own cache stem, so adding one
# never invalidates the base pool's cached embeddings.
#
# They exist because the training pool's real half is ImageNet and nothing else,
# and any real image from an absent domain gets mapped confidently into AIGC
# territory. Adding sid_real + unsplash_real took WildRF FPR@0.5 from .330 to
# .183 with no cost on ood.
SID_REAL_MANIFEST = _manifest("sid_real")            # SID_Set's real half (OpenImages)
UNSPLASH_REAL_MANIFEST = _manifest("unsplash_real")  # curated photography
# The pexels corpus was DELETED in Tier 5 -- fully orphaned, and this CSV was
# never written in the first place. The constant survives only so
# scripts/download_real_domains.py still imports; Tier 6 removes both.
PEXELS_REAL_MANIFEST = _manifest("pexels_real")
PHOTO_REAL_MANIFEST = _manifest("photo_real")        # union of the two above

# MODERN AI generators (2024-2025), added because our AIGC half is entirely
# GenImage-era 2022 and the remaining gap is diffusion, not GANs. Three separate
# publishers on purpose: one dataset is one provenance, and a detector will learn
# "this source" as readily as "this generator".
AIGC_EXT_DIR = DATA_DIR / "aigc_ext"
NANO_BANANA_MANIFEST = _manifest("nano_banana")      # Gemini 2.5 Flash Image
MIDJOURNEY_V6_MANIFEST = _manifest("midjourney_v6")
AIGC_MODERN_MANIFEST = _manifest("aigc_modern")      # union, EXCLUDES dalle3
# SD3 -- REJECTED 2026-08-30, mislabelled real photos. Quarantined under
# data/quarantine/. Read that README before ever re-adding this source.

# WildRF (arXiv:2406.09398): real + AI images scraped from Reddit, X and
# Facebook, carrying genuine platform re-encoding. Split in two on purpose --
# the reals are a TRAINING source, the test tier is EVAL ONLY, and they come
# from the same scrape, so a low FPR on the test tier after training on
# wildrf_real is an in-domain reading. The arm trained without it is the honest
# one.
REAL_EXT_DIR = DATA_DIR / "real_ext"
WILDRF_REAL_MANIFEST = _manifest("wildrf_real")

# --- Evaluation tiers: NEVER trained on --------------------------------------
#
# Each lives outside RAW_DIR, which is the only directory the split builder
# globs. That structural separation is what makes "never trained on" true by
# construction rather than by discipline.

WILDRF_DIR = DATA_DIR / "wildrf"
WILDRF_TEST_MANIFEST = _manifest("wildrf_test")

# Held-out modern-diffusion tier. Pulled alongside nano_banana and
# midjourney_v6 and deliberately kept back, so "does this generalise to a
# modern generator we never saw" is a question we can actually answer.
DALLE3_HOLDOUT_MANIFEST = _manifest("dalle3_holdout")

# In-distribution held-out set: Tiny-GenImage's own HF "validation" split.
# NOT a cross-generator set -- it contains the same 7 generators as train.
HELDOUT_DIR = DATA_DIR / "heldout"
HELDOUT_MANIFEST = _manifest("heldout")

# The challenge brief's external, self-reported benchmark (5.4): COCO val2017
# plus WildFake "DALL-E Advanced". The brief says explicitly not to train on it.
DEMO_VAL_DIR = DATA_DIR / "demo_val"
DEMO_VAL_MANIFEST = _manifest("demo_val")

# The hard out-of-distribution tier, and the only one with room left to
# discriminate: 18 generator classes, TEN absent from training, five of those
# GAN families. Under the shipping head demo-val has 16 of 18 views at or above
# 0.99; this tier has none.
OOD_DIR = DATA_DIR / "ood"
OOD_MANIFEST = _manifest("ood")

# --- Rejected corpora ---------------------------------------------------------

# Corpora withdrawn after an audit found them mislabelled. Kept with their
# evidence rather than deleted, because the finding is worth more than the disk.
QUARANTINE_DIR = DATA_DIR / "quarantine"

__all__ = [
    "AIGC_EXT_DIR",
    "AIGC_MODERN_MANIFEST",
    "CORPORA_DIR",
    "DALLE3_HOLDOUT_MANIFEST",
    "DATA_DIR",
    "DEMO_VAL_DIR",
    "DEMO_VAL_MANIFEST",
    "EMBEDDINGS_DIR",
    "HELDOUT_DIR",
    "HELDOUT_MANIFEST",
    "MANIFESTS_DIR",
    "MANIFESTS_RESOLVED_DIR",
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
