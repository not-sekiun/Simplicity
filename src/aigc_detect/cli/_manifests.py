from __future__ import annotations

import sys

from aigc_detect.config import (
    AIGC_MODERN_MANIFEST,
    DALLE3_HOLDOUT_MANIFEST,
    DEMO_VAL_MANIFEST,
    HELDOUT_MANIFEST,
    MIDJOURNEY_V6_MANIFEST,
    NANO_BANANA_MANIFEST,
    OOD_MANIFEST,
    PHOTO_REAL_MANIFEST,
    SID_REAL_MANIFEST,
    TRAIN_EXT_MANIFEST,
    TRAIN_MANIFEST,
    UNSPLASH_REAL_MANIFEST,
    VAL_MANIFEST,
    WILDRF_REAL_MANIFEST,
    WILDRF_TEST_MANIFEST,
)

# Single source of truth for --manifest. Four argparse choices lists used to
# carry their own hardcoded copy of these names, and they drifted the moment new
# corpora were added: _resolve_manifest knew about them, embed-views did not, so
# a valid manifest was rejected at the CLI boundary with a confusing error.
MANIFESTS = {
    "train": TRAIN_MANIFEST,
    "train-ext": TRAIN_EXT_MANIFEST,
    "sid-real": SID_REAL_MANIFEST,
    "photo-real": PHOTO_REAL_MANIFEST,
    "unsplash-real": UNSPLASH_REAL_MANIFEST,
    "wildrf-real": WILDRF_REAL_MANIFEST,
    "wildrf-test": WILDRF_TEST_MANIFEST,
    "nano-banana": NANO_BANANA_MANIFEST,
    "midjourney-v6": MIDJOURNEY_V6_MANIFEST,
    "dalle3-holdout": DALLE3_HOLDOUT_MANIFEST,
    "aigc-modern": AIGC_MODERN_MANIFEST,
    "val": VAL_MANIFEST,
    "heldout": HELDOUT_MANIFEST,
    "demo-val": DEMO_VAL_MANIFEST,
    "ood": OOD_MANIFEST,
}
MANIFEST_CHOICES = list(MANIFESTS)


def _resolve_manifest(name: str):
    """Map a --manifest choice to its resolved CSV, exiting with a hint if absent.

    demo-val is embeddable for EVALUATION ONLY (brief 5.4 forbids training on
    it), which the manifest's own `never_train: true` now enforces rather than
    leaving to convention.

    The hint used to be a per-manifest table naming which of seven `make_*.py`
    scripts to run. There is one answer now, because there is one way a manifest
    comes into existence.
    """
    manifest = MANIFESTS[name]
    if not manifest.exists():
        recipe = name.replace("-", "_")
        print(f"No {name} manifest at {manifest}.")
        print(f"Run: uv run aigc manifest resolve {recipe}")
        print("     (`aigc manifest list` shows every recipe; `aigc corpus list` what backs them)")
        sys.exit(1)
    return manifest
