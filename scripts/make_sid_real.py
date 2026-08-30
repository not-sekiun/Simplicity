"""Build data/processed/sid_real.csv -- SID_Set's REAL half only, no fakes.

WHY REALS ONLY. SID_Set was dropped from training because of two shortcuts, and
both live in the relationship between its halves, not in the images themselves
(FINDINGS 1). Its AIGC half is FLUX 1024x1024 composed like stock imagery; its
real half is OpenImages V7 -- cluttered, unrestricted-subject photographs. A
probe reading an 8x8 greyscale thumbnail separates those two piles at 0.935
balanced accuracy with geometry fully controlled, and a SID-trained head
transfers to CIFAKE at 0.5047 bacc, i.e. chance.

A composition shortcut is only exploitable when composition correlates with the
LABEL. Taking the reals alone and discarding all 4,000 fakes removes the
correlation by construction: nothing in this manifest carries label 1, so
"composed like stock imagery" has no AIGC side to point at.

WHAT IT IS FOR. FINDINGS 2h: every one of ood's 110 WhichFaceIsReal portraits is
called AI-generated with probability 1.000, and the diagnosis is domain coverage
of the REAL class -- our entire real pool is ImageNet. OpenImages is a different
real domain that is already on disk, so this is the cheapest available test of
that hypothesis (~4,000 images x 11 training views, versus hours of downloading
CelebA-HQ/Unsplash).

WHAT IT IS NOT. OpenImages contains people but not FFHQ-style close-up
portraits, so this is a partial probe of the portrait failure, not a fix for it.
If the canary stays at 1.000 while the ImageNet-real FPR falls, that is the
informative outcome: general diversity helps, faces need faces.

SEPARATE MANIFEST ON PURPOSE. Merging these rows into a union CSV would give the
combined file a new fingerprint and force re-embedding all 23,800 train images.
Its own manifest means its own cache stem; `train-head-views
--extra-train-manifest sid-real` concatenates the two caches at training time,
each still verified against its own manifest.

Usage:
    uv run python scripts/make_sid_real.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_detect.config import (  # noqa: E402
    DEMO_VAL_MANIFEST,
    HELDOUT_MANIFEST,
    LABEL_REAL,
    OOD_MANIFEST,
    RANDOM_SEED,
    SID_REAL_MANIFEST,
    VAL_MANIFEST,
)

SID_INDEX = Path(__file__).resolve().parents[1] / "data" / "raw" / "sid_set_index.csv"

# Distinct from train.csv's "Real" (ImageNet) so that per-source reporting --
# train_head's per-source val AUC and eval_grid's FPR-per-REAL-source table --
# can tell the two real domains apart instead of pooling them.
SOURCE = "sid_set_real"
GENERATOR = "Real_OpenImages"


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="Keep only the first N reals (seeded sample). Default: all of them.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    if not SID_INDEX.exists():
        raise SystemExit(f"No SID_Set index at {SID_INDEX}. Run `main.py download sid-set` first.")

    idx = pd.read_csv(SID_INDEX)
    n_all = len(idx)
    reals = idx[idx["label"] == LABEL_REAL].copy()
    print(f"[sid-real] index: {n_all:,} rows -> {len(reals):,} reals kept, "
          f"{n_all - len(reals):,} AIGC rows DISCARDED (FINDINGS 1b)")

    # The index was written in a previous session; data/ is gitignored and may
    # have been pruned since. Never emit a manifest row whose image is gone --
    # embed would fail thousands of rows in.
    exists = reals["image_path"].map(lambda p: Path(str(p)).is_file())
    missing = int((~exists).sum())
    if missing:
        print(f"[sid-real] WARNING: {missing:,} indexed real(s) are no longer on disk -- dropped")
    reals = reals[exists]
    if reals.empty:
        raise SystemExit("[sid-real] no SID_Set real images found on disk; nothing to build.")

    if a.limit is not None and a.limit < len(reals):
        reals = reals.sample(n=a.limit, random_state=a.seed)
        print(f"[sid-real] --limit: sampled {len(reals):,} of the available reals")

    out = pd.DataFrame({
        "image_path": reals["image_path"].astype(str),
        "label": LABEL_REAL,
        "source": SOURCE,
        "generator": GENERATOR,
    }).drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    # Leakage guard. These images live under data/raw/, which make_splits.py
    # globs -- they were excluded by --exclude-source sid_set, but check rather
    # than trust: a real in both train and an eval tier inflates that tier
    # silently, and this manifest exists precisely to move a real-image metric.
    names = set(out["image_path"].map(lambda p: Path(str(p)).name))
    for label, manifest in (("val", VAL_MANIFEST), ("heldout", HELDOUT_MANIFEST),
                            ("ood", OOD_MANIFEST), ("demo-val", DEMO_VAL_MANIFEST)):
        if not manifest.exists():
            print(f"[sid-real] {label}: manifest absent, skipped")
            continue
        other = pd.read_csv(manifest, usecols=["image_path"])["image_path"]
        overlap = names & {Path(str(p)).name for p in other}
        if overlap:
            raise SystemExit(
                f"[sid-real] LEAK: {len(overlap)} filename(s) appear in both this manifest and "
                f"{manifest}. Refusing to write -- training on them would inflate that tier."
            )
        print(f"[sid-real] no filename overlap with {label}")

    SID_REAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SID_REAL_MANIFEST, index=False)
    print(f"[sid-real] {len(out):,} rows -> {SID_REAL_MANIFEST}")
    print(f"[sid-real] labels: {out['label'].value_counts().to_dict()} (real-only by design)")
    print("\n[sid-real] next:")
    print("  uv run python scripts/worker.py --job embed:sid-real     # 11 training views only")
    print("  uv run main.py train-head-views --backbone pe-core-l --with-chains --val-sample-rows 2000 \\")
    print("      --extra-train-manifest sid-real --out models/pe-core-l__linear__sidreal.pt")
    return SID_REAL_MANIFEST


if __name__ == "__main__":
    main()
