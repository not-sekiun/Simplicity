"""Build manifests from the extracted WildRF dataset (data/real_ext/WildRF).

WildRF (Cavia et al., "Real-Time Deepfake Detection in the Real-World",
arXiv:2406.09398) is 5,613 full-resolution images scraped from Reddit, X and
Facebook -- real photographs and AI images as they actually circulate, already
carrying platform re-encoding. Its premise is that detectors trained under
standard protocols fail on real-world social content, which is the same gap
observed here.

TWO MANIFESTS, AND THE REASON THEY ARE SEPARATE:

  wildrf_real.csv   train/0_real + val/0_real (1,555 reals)  -- TRAINING source
  wildrf_test.csv   test/{reddit,twitter,facebook}, both     -- EVAL tier only
                    classes (2,503 images)

The split is WildRF's own, so training reals and test reals are disjoint images.
But they come from the same scrape, so a low FPR on wildrf_test after training
on wildrf_real is an IN-DOMAIN reading -- it says the head can fit this
distribution, not that it generalizes to unseen real domains. The honest
generalization number comes from the arm trained on Unsplash/Pexels only and
evaluated here. Both arms are worth having; only one answers the question.

Per-platform `generator` labels (Real_WildRF_reddit, ...) so eval_grid's
per-REAL-source FPR table breaks the eval tier down by platform rather than
pooling it -- the point of that table being that an aggregate hides a
subpopulation failure (FINDINGS 2h).

The AIGC half of the test tier is kept: those are real-world social-media fakes
of unknown provenance, which is much closer to the competition's "unknown
generator" condition than our GenImage-era training fakes. It is filed under
family "social" so it never enters the in-scope diffusion mean.

Usage:
    uv run python scripts/make_wildrf.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aigc_detect.config import DATA_DIR, LABEL_AIGC, LABEL_REAL, PROCESSED_DIR

WILDRF_DIR = DATA_DIR / "real_ext" / "WildRF"
TRAIN_OUT = PROCESSED_DIR / "wildrf_real.csv"
TEST_OUT = DATA_DIR / "wildrf" / "wildrf_test.csv"

# .jfif is in here deliberately: 262 WildRF files carry it (197 of them Twitter
# reals). It is ordinary JPEG under a Windows-era extension, and omitting it
# silently drops 58% of the Twitter real cell -- the kind of quiet loss that
# would have shown up only as a suspiciously small n.
EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"}


def _rows(directory: Path, label: int, source: str, generator: str) -> list[dict]:
    if not directory.is_dir():
        return []
    return [
        {"image_path": str(p.resolve()), "label": label, "source": source, "generator": generator}
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in EXTS
    ]


def main() -> tuple[Path, Path]:
    if not WILDRF_DIR.is_dir():
        raise SystemExit(f"No WildRF at {WILDRF_DIR}. Download and extract it first.")

    # TRAINING half: reals only. Adding the fakes at the same time would change
    # two variables at once, and the question this run answers is whether real
    # domain coverage moves FPR.
    train_rows: list[dict] = []
    for split in ("train", "val"):
        train_rows += _rows(WILDRF_DIR / split / "0_real", LABEL_REAL,
                            "wildrf_real", "Real_WildRF_train")
    if not train_rows:
        raise SystemExit("[wildrf] no training reals found -- check the extracted layout.")

    # EVAL half: every platform, both classes, per-platform labels.
    test_rows: list[dict] = []
    for platform in ("reddit", "twitter", "facebook"):
        base = WILDRF_DIR / "test" / platform
        test_rows += _rows(base / "0_real", LABEL_REAL,
                           f"wildrf_{platform}", f"Real_WildRF_{platform}")
        test_rows += _rows(base / "1_fake", LABEL_AIGC,
                           f"wildrf_{platform}", f"WildRF_{platform}")
    if not test_rows:
        raise SystemExit("[wildrf] no test images found -- check the extracted layout.")

    train_df = pd.DataFrame(train_rows).drop_duplicates(subset=["image_path"])
    test_df = pd.DataFrame(test_rows).drop_duplicates(subset=["image_path"])

    # The two halves must not share an image: the eval tier is worthless if the
    # head was trained on it. WildRF's own split guarantees this, so a hit here
    # means the extraction is not what we think it is.
    overlap = set(train_df["image_path"]) & set(test_df["image_path"])
    if overlap:
        raise SystemExit(f"[wildrf] LEAK: {len(overlap)} image(s) in both the train and test manifests.")

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"[wildrf] TRAIN reals : {len(train_df):,} -> {TRAIN_OUT}")
    print(f"[wildrf] EVAL tier   : {len(test_df):,} -> {TEST_OUT}")
    print(f"[wildrf] eval labels : {test_df['label'].value_counts().to_dict()}")
    print("[wildrf] eval per-platform:")
    for src, grp in test_df.groupby("source"):
        n_real = int((grp["label"] == LABEL_REAL).sum())
        print(f"[wildrf]   {src:<18} {len(grp):>5} ({n_real} real / {len(grp) - n_real} fake)")
    return TRAIN_OUT, TEST_OUT


if __name__ == "__main__":
    main()
