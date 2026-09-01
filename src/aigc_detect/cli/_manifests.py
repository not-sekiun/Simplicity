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
    PEXELS_REAL_MANIFEST,
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
    "pexels-real": PEXELS_REAL_MANIFEST,
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
    """Map a --manifest choice to its path, exiting with a hint if it's missing.

    demo-val is embeddable for EVALUATION ONLY (brief 5.4 forbids training on
    it). train_head never looks at it -- it hardcodes TRAIN_MANIFEST/VAL_MANIFEST.
    """
    manifests = MANIFESTS
    manifest = manifests[name]
    if not manifest.exists():
        hint = {"demo-val": "build-demo-val", "heldout": "build-heldout",
                "ood": "download-ood` then `main.py build-ood",
                "train-ext": "python scripts/make_train_ext.py",
                "sid-real": "python scripts/make_sid_real.py",
                "photo-real": "python scripts/download_real_domains.py --merge",
                "wildrf-real": "python scripts/make_wildrf.py",
                "wildrf-test": "python scripts/make_wildrf.py",
                "nano-banana": "python scripts/download_aigc_modern.py --source nano-banana",
                "midjourney-v6": "python scripts/download_aigc_modern.py --source midjourney-v6",
                "dalle3-holdout": "python scripts/download_aigc_modern.py --source dalle3-holdout",
                "aigc-modern": "python scripts/download_aigc_modern.py --merge"}.get(name, "split")
        print(f"No {name} manifest at {manifest}. Run `main.py {hint}` first.")
        sys.exit(1)
    return manifest
