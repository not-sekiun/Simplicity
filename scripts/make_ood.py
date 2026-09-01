"""Merge data/ood/*_index.csv into data/ood/ood.csv (no split).

This tier is EVALUATION ONLY. Like data/heldout/ and data/demo_val/, it lives
outside data/raw/, which is the only directory scripts/make_splits.py globs, so
it cannot leak into training through any code path.

Also runs a leakage guard against the training manifests. The guard is on
filename, which is weak here (this tier's filenames are freshly generated
`ood_NNNNNNN.jpg`, so a collision is near-impossible by construction) -- but
the two datasets do share upstream sources: Tiny-GenImage and
AIGC-Detection-Benchmark both draw on GenImage, so the same underlying image
could in principle appear in both. The guard cannot detect that. Treat the
seen-generator rows (ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM, Wukong) as
possibly-overlapping with training content and read the UNSEEN-generator rows
as the trustworthy signal -- which is what this tier is for anyway.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aigc_detect.config import (
    OOD_DIR,
    OOD_MANIFEST,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)


def main() -> Path:
    index_files = sorted(OOD_DIR.glob("*_index.csv"))
    if not index_files:
        raise SystemExit(
            f"No *_index.csv under {OOD_DIR}. Run `uv run main.py download-ood` first."
        )

    df = pd.concat([pd.read_csv(f) for f in index_files], ignore_index=True)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    OOD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OOD_MANIFEST, index=False)

    print(f"[ood] {len(df)} rows -> {OOD_MANIFEST}")
    print(f"[ood] labels: {df['label'].value_counts().to_dict()}")
    print(f"[ood] generators: {df['generator'].value_counts().to_dict()}")

    # Leakage guard (filename-level; see module docstring for its limits).
    names = set(df["image_path"].map(lambda p: Path(str(p)).name))
    for manifest, tag in [(TRAIN_MANIFEST, "train"), (VAL_MANIFEST, "val")]:
        if not manifest.exists():
            continue
        other = set(pd.read_csv(manifest, usecols=["image_path"])["image_path"].map(lambda p: Path(str(p)).name))
        overlap = names & other
        if overlap:
            print(f"[ood] WARNING: {len(overlap)} filename(s) also appear in {tag}.csv")
        else:
            print(f"[ood] no filename overlap with {tag}.csv")
    return OOD_MANIFEST


if __name__ == "__main__":
    main()
