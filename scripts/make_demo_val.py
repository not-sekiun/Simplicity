"""Merge data/demo_val/*_index.csv into one demo_val.csv manifest.

This set is NEVER split into train/val and NEVER merged into
data/processed/{train,val}.csv — per the challenge brief (5.4) it's a
self-reported, out-of-training benchmark ("Do not use the following data
during training"). Run this after download_demo_val.py has indexed at least
one of its two halves.

Also runs a leakage guard: warns (does not fail) if any demo_val image
filename collides with a filename already in data/processed/{train,val}.csv,
since accidental overlap would invalidate the "held-out" claim.

Usage:
    uv run main.py build-demo-val
    uv run python scripts/make_demo_val.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from aigc_detect.config import DEMO_VAL_DIR, DEMO_VAL_MANIFEST, TRAIN_MANIFEST, VAL_MANIFEST


def load_demo_indexes() -> pd.DataFrame:
    index_files = sorted(DEMO_VAL_DIR.glob("*_index.csv"))
    if not index_files:
        raise FileNotFoundError(
            f"No *_index.csv under {DEMO_VAL_DIR}. Run `main.py download-demo coco-val2017` "
            "and/or `main.py download-demo wildfake-dalle-advanced` first."
        )
    frames = [pd.read_csv(f) for f in index_files]
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="image_path").reset_index(drop=True)


def check_leakage(demo_df: pd.DataFrame) -> None:
    if not (TRAIN_MANIFEST.exists() and VAL_MANIFEST.exists()):
        return
    train_names = set(pd.read_csv(TRAIN_MANIFEST)["image_path"].apply(lambda p: Path(p).name))
    val_names = set(pd.read_csv(VAL_MANIFEST)["image_path"].apply(lambda p: Path(p).name))
    demo_names = set(demo_df["image_path"].apply(lambda p: Path(p).name))
    collisions = demo_names & (train_names | val_names)
    if collisions:
        print(f"[build-demo-val] WARNING: {len(collisions)} filename(s) appear in BOTH "
              f"demo_val and train/val — inspect for accidental leakage: {list(collisions)[:5]}...")
    else:
        print("[build-demo-val] leakage check OK: no filename overlap with train/val manifests.")


def main():
    demo_df = load_demo_indexes()
    counts = demo_df.groupby(["source", "label"]).size()
    print(f"[build-demo-val] loaded {len(demo_df)} demo-val images:\n{counts}")

    check_leakage(demo_df)

    DEMO_VAL_DIR.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(DEMO_VAL_MANIFEST, index=False)
    print(f"[build-demo-val] wrote {len(demo_df)} rows -> {DEMO_VAL_MANIFEST}")

    if demo_df["source"].nunique() < 2:  # noqa: PLR2004
        have = set(demo_df["source"].unique())
        missing = {"coco_val2017", "wildfake_dalle_advanced"} - have
        print(f"[build-demo-val] NOTE: only {have} indexed so far — still missing {missing}.")


if __name__ == "__main__":
    main()
