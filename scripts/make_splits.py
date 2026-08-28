"""Build stratified train/val manifests from every indexed raw dataset.

Reads all data/raw/*_index.csv (written by download_data.py), concatenates
them, and writes data/processed/{train,val}.csv — stratified by (source,
label) so each dataset's class balance is preserved in both splits.

Usage:
    uv run main.py split --val-fraction 0.15 --seed 42
    uv run python scripts/make_splits.py --val-fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from aigc_detect.config import PROCESSED_DIR, RAW_DIR, RANDOM_SEED, VAL_FRACTION


def load_indexes() -> pd.DataFrame:
    index_files = sorted(RAW_DIR.glob("*_index.csv"))
    if not index_files:
        raise FileNotFoundError(
            f"No *_index.csv files found under {RAW_DIR}. Run `download_data.py` "
            "(or `main.py download ...`) first."
        )
    frames = [pd.read_csv(f) for f in index_files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="image_path").reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = df["source"] + "_" + df["label"].astype(str)
    # Any stratum with <2 members can't be split with stratify; fall back to
    # putting it entirely in train and warn, rather than crashing.
    counts = strata.value_counts()
    tiny = counts[counts < 2].index
    if len(tiny):
        print(f"[split] warning: strata with <2 samples go entirely to train: {list(tiny)}")
    df_ok, df_tiny = df[~strata.isin(tiny)], df[strata.isin(tiny)]
    strata_ok = strata[~strata.isin(tiny)]

    train_df, val_df = train_test_split(
        df_ok, test_size=val_fraction, random_state=seed, stratify=strata_ok
    )
    train_df = pd.concat([train_df, df_tiny], ignore_index=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    df = load_indexes()
    print(f"[split] loaded {len(df)} indexed images across sources: "
          f"{df['source'].value_counts().to_dict()}")

    train_df, val_df = stratified_split(df, args.val_fraction, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[split] train: {len(train_df)} rows -> {train_path} "
          f"(label counts: {train_df['label'].value_counts().to_dict()})")
    print(f"[split] val:   {len(val_df)} rows -> {val_path} "
          f"(label counts: {val_df['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
