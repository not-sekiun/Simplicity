"""Label encoding, split parameters, and the fallback preprocessing constants.

These are the values that must agree across every stage of the pipeline. A
manifest written under one label convention and scored under another produces
an inverted AUC, which looks like a broken model rather than a broken constant.
"""

from __future__ import annotations

# Canonical labels, used by every manifest and every metric in the project.
# CSV manifests always carry columns image_path,label,source[,generator].
LABEL_REAL = 0
LABEL_AIGC = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_AIGC: "aigc"}

# Train/val split. The seed is pinned because the split defines which images a
# cached embedding covers.
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# FALLBACK preprocessing only -- do not reach for these when embedding.
#
# Each backbone declares its own native resolution (224 / 256 / 336 / 378 / 518)
# and its own normalisation statistics, and load_backbone() attaches them to the
# returned module. Using these ImageNet defaults instead evaluates a model under
# normalisation it never saw, which degrades quietly rather than failing. They
# remain here for the augmentation-preview command and as the documented
# fallback when a checkpoint ships no image processor.
IMAGE_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

__all__ = [
    "IMAGE_SIZE",
    "LABEL_AIGC",
    "LABEL_NAMES",
    "LABEL_REAL",
    "NORM_MEAN",
    "NORM_STD",
    "RANDOM_SEED",
    "VAL_FRACTION",
]
