"""Manifest-driven image dataset.

A "manifest" is a CSV with columns: image_path, label, source.
  - image_path: path to the image (absolute, or relative to the repo root)
  - label:      0 = real, 1 = AIGC (see aigc_detect.config.LABEL_*)
  - source:     which raw dataset it came from (e.g. "cifake", "sid_set")

scripts/make_splits.py builds data/processed/{train,val}.csv from whatever
raw datasets have been indexed under data/raw/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from aigc_detect.config import ROOT_DIR

MANIFEST_COLUMNS = ["image_path", "label", "source"]


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

    def _resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (ROOT_DIR / p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self._resolve(row["image_path"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    def class_counts(self) -> dict[int, int]:
        return self.df["label"].value_counts().to_dict()
