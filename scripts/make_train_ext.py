"""Build data/processed/train_ext.csv = train.csv + the data/train_ext/ slice.

WHY A NEW MANIFEST RATHER THAN data/raw/ + `main.py split`. Dropping the new
images into data/raw/ and re-splitting would rewrite train.csv and val.csv IN
PLACE. Both files keep their names, so every cached embedding keyed on their
fingerprint is silently invalidated -- the full-pool embedding, every val cache,
and the whole comparability epoch (NARRATIVE "Comparability epochs"). Writing a
new manifest under a new name means a new cache stem, so nothing already
computed goes stale and the existing benchmark stays reproducible.

LEAKAGE. The slice is provably disjoint from data/ood/ by construction (see
download_ood_benchmark: deterministic stream order, filenames encode stream
position, the eval tier occupies 1..8400, the slice starts after it). It is NOT
provably disjoint from val.csv: both AIGC-Detection-Benchmark and Tiny-GenImage
draw on GenImage upstream, so the same source image could in principle appear in
both under different filenames. Two mitigations:

  1. The slice keeps only generators ABSENT from data/raw/, so the fake half
     cannot collide with Tiny-GenImage's fakes at all.
  2. The reals can still collide. That would inflate `val` and NOT `ood` (ood is
     disjoint by construction), which is a detectable signature: if val improves
     much more than ood after adding this data, suspect real-image leakage
     rather than genuine gains. This is stated in the handoff too, because it is
     the kind of thing that otherwise gets read as a win.

Usage:
    uv run python scripts/make_train_ext.py [--balance]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_detect.config import (  # noqa: E402
    DATA_DIR,
    LABEL_AIGC,
    LABEL_REAL,
    PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)

EXT_INDEX = DATA_DIR / "train_ext" / "train_ext_index.csv"
OUT_MANIFEST = PROCESSED_DIR / "train_ext.csv"


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--balance", action="store_true",
                    help="Drop surplus reals from the new slice so it contributes 50/50. The real "
                         "quota was sized for 16 fake generators but only 9 are kept, so the slice "
                         "is real-heavy on its own.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    if not EXT_INDEX.exists():
        raise SystemExit(f"No slice index at {EXT_INDEX}. Run the train_ext pull first.")

    base = pd.read_csv(TRAIN_MANIFEST)
    ext = pd.read_csv(EXT_INDEX)
    print(f"[train-ext] base train.csv : {len(base):,} rows, {base['generator'].nunique()} generators")
    print(f"[train-ext] new slice      : {len(ext):,} rows, {ext['generator'].nunique()} generators "
          f"{dict(sorted(ext['generator'].value_counts().items()))}")

    if a.balance:
        n_fake = int((ext["label"] == LABEL_AIGC).sum())
        reals = ext[ext["label"] == LABEL_REAL]
        if len(reals) > n_fake:
            keep = reals.sample(n=n_fake, random_state=a.seed)
            ext = pd.concat([ext[ext["label"] == LABEL_AIGC], keep])
            print(f"[train-ext] --balance: trimmed reals {len(reals):,} -> {n_fake:,}")

    merged = pd.concat([base, ext], ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    # Filename-level leakage guard against val. Weak (see module docstring) but
    # it costs nothing and would catch an outright mistake.
    val_names = set(pd.read_csv(VAL_MANIFEST, usecols=["image_path"])["image_path"].map(lambda p: Path(str(p)).name))
    overlap = {Path(str(p)).name for p in ext["image_path"]} & val_names
    if overlap:
        print(f"[train-ext] WARNING: {len(overlap)} slice filename(s) also appear in val.csv")
    else:
        print("[train-ext] no filename overlap between the new slice and val.csv")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_MANIFEST, index=False)
    print(f"[train-ext] {len(merged):,} rows -> {OUT_MANIFEST}")
    print(f"[train-ext] labels: {merged['label'].value_counts().to_dict()}")
    print(f"[train-ext] generators: {merged['generator'].nunique()} "
          f"({sorted(merged['generator'].unique())})")
    print("\n[train-ext] next:")
    print("  uv run main.py embed-views --backbone pe-core-l --manifest train-ext --train-chains")
    print("  uv run main.py train-head-views --backbone pe-core-l --with-chains --val-sample-rows 2000 \\")
    print("      --train-manifest-name train-ext --out models/pe-core-l__linear__trainext.pt")
    return OUT_MANIFEST


if __name__ == "__main__":
    main()
