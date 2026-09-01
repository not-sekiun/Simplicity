"""Download Tiny-GenImage into data/raw/ (train) and data/heldout/ (validation),
and write per-tier index CSVs.

TheKernel01/Tiny-GenImage (HuggingFace, ungated, parquet, ~8.4GB total):
  splits:     train (28,000), validation (7,000)
  features:   image (PIL), label ClassLabel ['real','fake'] (0=real, 1=fake --
              already matches our convention), generator ClassLabel
              ['Real','ADM','BigGAN','GLIDE','Midjourney','SD14','SD15','VQDM',
              'Wukong']
  license:    cc-by-nc-sa-4.0

Why this dataset replaces SID_Set as our AIGC training source: see FINDINGS.md
section 1 -- SID_Set's real half (OpenImages photos) and AI half (FLUX) are
separable at 0.93 balanced accuracy from an 8x8 greyscale thumbnail alone (a
composition shortcut), because the two halves are unrelated image piles.
Tiny-GenImage's reals and fakes are both ImageNet-class-prompted, so no such
shortcut can form, and it carries 8 distinct generators.

Every image is re-encoded as JPEG quality 95 regardless of its source format,
converting to RGB first. This is deliberate: leaving source formats alone
would reopen a *different* shortcut, since real photos are usually native
JPEG and AI images are usually native PNG -- a detector can hit ~99% just by
reading compression history, learning nothing about content. Uniform
re-encoding closes that off.

The HF `train` split is our training pool -> data/raw/tiny_genimage/ +
data/raw/tiny_genimage_index.csv (source=tiny_genimage). The HF `validation`
split is a held-out, cross-generator test set that must NEVER be trained on
and must NEVER be globbed by scripts/make_splits.py -> data/heldout/
tiny_genimage/ + data/heldout/tiny_genimage_heldout_index.csv
(source=tiny_genimage_heldout). This is a third tier distinct from
data/demo_val/ (the brief's external self-reported benchmark, 5.4) -- see
src/aigc_detect/config.py for the three-tier layout.

Usage (via the entry script, preferred):
    uv run main.py download tiny-genimage
    uv run main.py download tiny-genimage --limit-per-split 40 --force

Or directly:
    uv run python scripts/download_tiny_genimage.py
    uv run python scripts/download_tiny_genimage.py --limit-per-split 40 --force

Streaming note: `load_dataset(..., split=...)` here is NOT the streaming=True
mode used for SID_Set -- Tiny-GenImage is small enough (~8.4GB) to pull as a
regular (non-streaming) dataset, which gives HF's own resumable download
cache for free. A `Server disconnected` mid-download has been observed once;
re-running the same command resumes from HF's cache rather than restarting.
Writing to disk is separately resumable via the --force flag / on-disk
existence check below, independent of that HF-level cache.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aigc_detect.config import HELDOUT_DIR, LABEL_AIGC, LABEL_REAL, RAW_DIR

TINY_GENIMAGE_HF_HANDLE = "TheKernel01/Tiny-GenImage"

# (output_dir_for_images, index_csv_path, source_name) per HF split.
_TIER_CONFIG = {
    "train": (RAW_DIR / "tiny_genimage", RAW_DIR / "tiny_genimage_index.csv", "tiny_genimage"),
    "validation": (
        HELDOUT_DIR / "tiny_genimage",
        HELDOUT_DIR / "tiny_genimage_heldout_index.csv",
        "tiny_genimage_heldout",
    ),
}


def _write_index(index_path: Path, source: str, records: list[tuple[str, int, str]]) -> Path:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source", "generator"])
        for image_path, label, generator in records:
            writer.writerow([image_path, label, source, generator])
    print(f"[tiny_genimage] wrote {len(records)} rows -> {index_path}")
    return index_path


def _download_split(split: str, limit: int | None, force: bool) -> Path:
    from datasets import load_dataset
    from tqdm import tqdm

    out_dir, index_path, source = _TIER_CONFIG[split]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tiny_genimage] loading '{TINY_GENIMAGE_HF_HANDLE}' split='{split}'...")
    ds = load_dataset(TINY_GENIMAGE_HF_HANDLE, split=split)
    label_names = ds.features["label"].names  # ['real', 'fake']
    generator_names = ds.features["generator"].names  # ['Real','ADM',...,'Wukong']

    n = len(ds) if limit is None else min(limit, len(ds))
    records: list[tuple[str, int, str]] = []
    label_map = {0: LABEL_REAL, 1: LABEL_AIGC}

    for idx in tqdm(range(n), desc=f"[tiny_genimage:{split}]"):
        example = ds[idx]
        raw_label = example["label"]
        norm_label = label_map[raw_label]
        generator = generator_names[example["generator"]]
        gen_dir = out_dir / generator
        gen_dir.mkdir(parents=True, exist_ok=True)
        img_path = gen_dir / f"{split}_{idx:06d}.jpg"

        if force or not img_path.exists():
            img = example["image"]
            img.convert("RGB").save(img_path, format="JPEG", quality=95)

        records.append((str(img_path.resolve()), norm_label, generator))

    print(f"[tiny_genimage:{split}] label names={label_names} generator names={generator_names}")
    if not records:
        raise RuntimeError(
            f"No Tiny-GenImage '{split}' images were saved. The HF dataset schema may have "
            "changed -- inspect the download and adjust download_tiny_genimage.py."
        )
    return _write_index(index_path, source, records)


def download_tiny_genimage(
    limit_per_split: int | None = None,
    splits: tuple[str, ...] = ("train", "validation"),
    force: bool = False,
) -> dict[str, Path]:
    return {split: _download_split(split, limit_per_split, force) for split in splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--limit-per-split", type=int, default=None,
        help="Only download the first N images of each split (default: all).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"],
        help="Which HF splits to download (default: both).",
    )
    parser.add_argument("--force", action="store_true", help="Re-encode images even if already written.")
    args = parser.parse_args()

    download_tiny_genimage(limit_per_split=args.limit_per_split, splits=tuple(args.splits), force=args.force)


if __name__ == "__main__":
    main()
