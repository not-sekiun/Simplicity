"""Build stratified train/val manifests from every indexed raw dataset.

Reads all data/raw/*_index.csv (written by download_data.py /
download_tiny_genimage.py), concatenates them, and writes
data/processed/{train,val}.csv -- stratified by (source, label) so each
dataset's class balance is preserved in both splits. Only ever globs
data/raw/ -- data/heldout/ (cross-generator test set) and data/demo_val/
(the brief's external benchmark) are structurally unreachable from here,
see src/aigc_detect/config.py.

Every row carries a `generator` column. Sources that were indexed before
Tiny-GenImage introduced it (cifake, sid_set) are backfilled here so the
column is uniform across the merged manifest:
    cifake   label 0 (real) -> CIFAR-10   label 1 (aigc) -> SD14
    sid_set  label 0 (real) -> OpenImages label 1 (aigc) -> FLUX
    (anything else unlabelled)          -> unknown

We deliberately do NOT stratify by generator -- that would spread every
generator across both train and val, defeating held-out-generator
evaluation. Use --holdout-generators to move specific generators entirely
into val instead.

Usage:
    uv run main.py split --val-fraction 0.15 --seed 42
    uv run main.py split --exclude-source sid_set --max-per-source 5000
    uv run main.py split --holdout-generators Midjourney VQDM
    uv run python scripts/make_splits.py --val-fraction 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

from aigc_detect.config import LABEL_AIGC, LABEL_REAL, PROCESSED_DIR, RAW_DIR, RANDOM_SEED, VAL_FRACTION

# Backfill for sources indexed before the `generator` column existed.
_GENERATOR_BACKFILL = {
    "cifake": {LABEL_REAL: "CIFAR-10", LABEL_AIGC: "SD14"},
    "sid_set": {LABEL_REAL: "OpenImages", LABEL_AIGC: "FLUX"},
}


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

    if "generator" not in df.columns:
        df["generator"] = pd.NA
    missing_generator = df["generator"].isna()
    if missing_generator.any():
        def _backfill(row):
            table = _GENERATOR_BACKFILL.get(row["source"])
            if table is None:
                return "unknown"
            return table.get(row["label"], "unknown")

        df.loc[missing_generator, "generator"] = df.loc[missing_generator].apply(_backfill, axis=1)

    return df


def filter_sources(df: pd.DataFrame, exclude_sources: list[str] | None) -> pd.DataFrame:
    if not exclude_sources:
        return df
    excluded = set(exclude_sources)
    kept = df[~df["source"].isin(excluded)].reset_index(drop=True)
    dropped = len(df) - len(kept)
    print(f"[split] --exclude-source {sorted(excluded)}: dropped {dropped} rows")
    return kept


def cap_per_source(df: pd.DataFrame, max_per_source: int | None, seed: int) -> pd.DataFrame:
    if max_per_source is None:
        return df
    parts = []
    for source, group in df.groupby("source", sort=False):
        if len(group) > max_per_source:
            group = group.sample(n=max_per_source, random_state=seed)
            print(f"[split] --max-per-source {max_per_source}: subsampled {source} "
                  f"{len(df[df['source'] == source])} -> {max_per_source}")
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


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


def holdout_generators(
    train_df: pd.DataFrame, val_df: pd.DataFrame, generators: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move every row whose generator is in `generators` out of train and into
    val, so val measures performance on generators unseen during training."""
    if not generators:
        return train_df, val_df
    wanted = set(generators)
    moved_mask = train_df["generator"].isin(wanted)
    moved = train_df[moved_mask]
    if moved.empty:
        print(f"[split] --holdout-generators {sorted(wanted)}: no matching rows found in train")
        return train_df, val_df

    train_df = train_df[~moved_mask].reset_index(drop=True)
    val_df = pd.concat([val_df, moved], ignore_index=True)
    print(f"[split] --holdout-generators {sorted(wanted)}: moved {len(moved)} rows from train -> val "
          f"(by generator: {moved['generator'].value_counts().to_dict()})")
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--exclude-source", action="append", default=None, metavar="SOURCE",
        help="Drop a source entirely before splitting (repeatable), e.g. --exclude-source sid_set.",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=None,
        help="Cap rows per source by random subsample (seeded), before splitting.",
    )
    parser.add_argument(
        "--holdout-generators", nargs="+", default=None, metavar="GENERATOR",
        help="Move all rows for these generators out of train and into val (unseen-generator eval).",
    )
    args = parser.parse_args()

    df = load_indexes()
    print(f"[split] loaded {len(df)} indexed images across sources: "
          f"{df['source'].value_counts().to_dict()}")

    df = filter_sources(df, args.exclude_source)
    df = cap_per_source(df, args.max_per_source, args.seed)

    train_df, val_df = stratified_split(df, args.val_fraction, args.seed)
    train_df, val_df = holdout_generators(train_df, val_df, args.holdout_generators)

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
