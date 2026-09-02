"""Manifest-driven image dataset, and the one rule for reading an image_path.

A "manifest" is a CSV with columns: image_path, label, source (+ generator).
  - image_path: absolute, or relative to $AIGC_DATA_ROOT -- see
                :func:`resolve_image_path`, which is the only place that rule
                is written down.
  - label:      0 = real, 1 = AIGC (see aigc_detect.config.LABEL_*)
  - source:     which corpus it came from (e.g. "tiny_genimage", "wildrf_real")

Manifests are built by recipes under `data/manifests/` -- see
:mod:`aigc_detect.data.manifest`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from aigc_detect.config import DATA_DIR

MANIFEST_COLUMNS = ["image_path", "label", "source"]


def resolve_image_path(raw: str | Path) -> Path:
    """A manifest row's image_path as a file on this machine.

    Absolute stays absolute; relative hangs off ``$AIGC_DATA_ROOT``.

    RELATIVE TO THE DATA ROOT, NOT THE REPO ROOT. The two coincide today, since
    `data/` sits inside the repo -- they diverge exactly when someone sets
    AIGC_DATA_ROOT to move 26 GB of images off the system drive, which is the
    case worth being right for. It is also the root the cache's hash memo keys
    on, so a manifest and a cached embedding relocate together or not at all.

    THIS FUNCTION HAD THREE COPIES: the dataset, the multi-view embedder and the
    error analysis each carried their own two-line version. That is survivable
    while every committed path is absolute and the rule never fires; it stops
    being survivable in tier 5, where relative paths become the norm and a
    disagreement between two of those copies would mean two components reading
    different files for the same row.
    """
    p = Path(str(raw))
    return p if p.is_absolute() else DATA_DIR / p


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform=None):
        self.manifest_path = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_path)
        missing = set(MANIFEST_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns: {missing}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(resolve_image_path(row["image_path"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    def class_counts(self) -> dict[int, int]:
        return self.df["label"].value_counts().to_dict()
