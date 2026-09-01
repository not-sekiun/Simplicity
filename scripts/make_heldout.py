"""Merge data/heldout/*_index.csv into one heldout.csv manifest.

data/heldout/ is the cross-generator test tier (see src/aigc_detect/config.py):
images from a HF dataset's own "validation" split (e.g. Tiny-GenImage) that
were never part of our training pool (data/raw/) and so give an honest
unseen-generator readout. This set is NEVER split into train/val and NEVER
merged into data/processed/{train,val}.csv -- it lives in a directory
scripts/make_splits.py structurally never globs.

Also runs a leakage guard, mirroring scripts/make_demo_val.py: warns (does
not fail) if any heldout image filename collides with a filename already in
data/processed/{train,val}.csv, since accidental overlap would invalidate
the "held-out" claim.

Usage:
    uv run main.py build-heldout
    uv run python scripts/make_heldout.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aigc_detect.config import HELDOUT_DIR, HELDOUT_MANIFEST, TRAIN_MANIFEST, VAL_MANIFEST


def load_heldout_indexes() -> pd.DataFrame:
    index_files = sorted(HELDOUT_DIR.glob("*_index.csv"))
    if not index_files:
        raise FileNotFoundError(
            f"No *_index.csv under {HELDOUT_DIR}. Run `main.py download tiny-genimage` first."
        )
    frames = [pd.read_csv(f) for f in index_files]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_path").reset_index(drop=True)


def check_leakage(heldout_df: pd.DataFrame) -> None:
    if not (TRAIN_MANIFEST.exists() and VAL_MANIFEST.exists()):
        return
    train_names = set(pd.read_csv(TRAIN_MANIFEST)["image_path"].apply(lambda p: Path(p).name))
    val_names = set(pd.read_csv(VAL_MANIFEST)["image_path"].apply(lambda p: Path(p).name))
    heldout_names = set(heldout_df["image_path"].apply(lambda p: Path(p).name))
    collisions = heldout_names & (train_names | val_names)
    if collisions:
        print(f"[build-heldout] WARNING: {len(collisions)} filename(s) appear in BOTH "
              f"heldout and train/val -- inspect for accidental leakage: {list(collisions)[:5]}...")
    else:
        print("[build-heldout] leakage check OK: no filename overlap with train/val manifests.")


def main():
    heldout_df = load_heldout_indexes()
    counts = heldout_df.groupby(["source", "label"]).size()
    print(f"[build-heldout] loaded {len(heldout_df)} heldout images:\n{counts}")

    check_leakage(heldout_df)

    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    heldout_df.to_csv(HELDOUT_MANIFEST, index=False)
    print(f"[build-heldout] wrote {len(heldout_df)} rows -> {HELDOUT_MANIFEST}")


if __name__ == "__main__":
    main()
