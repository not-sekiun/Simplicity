"""Central configuration: paths, labels, generator taxonomy, environment.

Split into four modules by what a value *is*, re-exported here so that
``from aigc_detect.config import TRAIN_MANIFEST`` keeps working unchanged:

    settings.py    environment and credentials (.env), the only reader of os.environ
    paths.py       every directory and manifest, rooted at $AIGC_DATA_ROOT
    labels.py      label encoding, split seed, fallback preprocessing constants
    generators.py  generator -> architecture family, and the training-pool set

Import this package, not the submodules, unless you specifically want
``get_settings`` / ``resolve_device`` / ``hf_token_kwargs`` -- those are exported
here too but read more clearly qualified as ``settings.resolve_device()``.

Cheap to import by design: no torch, no network, no filesystem scan. Device
resolution is a function, not a constant, for exactly that reason.
"""

from __future__ import annotations

from aigc_detect.config.generators import GENERATOR_FAMILY, TRAIN_GENERATORS
from aigc_detect.config.labels import (
    IMAGE_SIZE,
    LABEL_AIGC,
    LABEL_NAMES,
    LABEL_REAL,
    NORM_MEAN,
    NORM_STD,
    RANDOM_SEED,
    VAL_FRACTION,
)
from aigc_detect.config.paths import (
    AIGC_EXT_DIR,
    AIGC_MODERN_MANIFEST,
    CORPORA_DIR,
    DALLE3_HOLDOUT_MANIFEST,
    DATA_DIR,
    DEMO_VAL_DIR,
    DEMO_VAL_MANIFEST,
    EMBEDDINGS_DIR,
    HELDOUT_DIR,
    HELDOUT_MANIFEST,
    MANIFESTS_DIR,
    MANIFESTS_RESOLVED_DIR,
    MIDJOURNEY_V6_MANIFEST,
    NANO_BANANA_MANIFEST,
    OOD_DIR,
    OOD_MANIFEST,
    PEXELS_REAL_MANIFEST,
    PHOTO_REAL_MANIFEST,
    PROCESSED_DIR,
    QUARANTINE_DIR,
    RAW_DIR,
    REAL_EXT_DIR,
    ROOT_DIR,
    SID_REAL_MANIFEST,
    TRAIN_EXT_MANIFEST,
    TRAIN_MANIFEST,
    UNSPLASH_REAL_MANIFEST,
    VAL_MANIFEST,
    WILDRF_DIR,
    WILDRF_REAL_MANIFEST,
    WILDRF_TEST_MANIFEST,
)
from aigc_detect.config.settings import (
    Settings,
    get_settings,
    hf_token_kwargs,
    resolve_device,
)

__all__ = [
    "AIGC_EXT_DIR",
    "AIGC_MODERN_MANIFEST",
    "CORPORA_DIR",
    "DALLE3_HOLDOUT_MANIFEST",
    "DATA_DIR",
    "DEMO_VAL_DIR",
    "DEMO_VAL_MANIFEST",
    "EMBEDDINGS_DIR",
    "GENERATOR_FAMILY",
    "HELDOUT_DIR",
    "HELDOUT_MANIFEST",
    "IMAGE_SIZE",
    "LABEL_AIGC",
    "LABEL_NAMES",
    "LABEL_REAL",
    "MANIFESTS_DIR",
    "MANIFESTS_RESOLVED_DIR",
    "MIDJOURNEY_V6_MANIFEST",
    "NANO_BANANA_MANIFEST",
    "NORM_MEAN",
    "NORM_STD",
    "OOD_DIR",
    "OOD_MANIFEST",
    "PEXELS_REAL_MANIFEST",
    "PHOTO_REAL_MANIFEST",
    "PROCESSED_DIR",
    "QUARANTINE_DIR",
    "RANDOM_SEED",
    "RAW_DIR",
    "REAL_EXT_DIR",
    "ROOT_DIR",
    "SID_REAL_MANIFEST",
    "TRAIN_EXT_MANIFEST",
    "TRAIN_GENERATORS",
    "TRAIN_MANIFEST",
    "UNSPLASH_REAL_MANIFEST",
    "VAL_FRACTION",
    "VAL_MANIFEST",
    "WILDRF_DIR",
    "WILDRF_REAL_MANIFEST",
    "WILDRF_TEST_MANIFEST",
    "Settings",
    "get_settings",
    "hf_token_kwargs",
    "resolve_device",
]
