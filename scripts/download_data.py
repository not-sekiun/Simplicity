"""Download raw datasets into data/raw/<source>/ and write a per-source index CSV.

Each index CSV (data/raw/<source>_index.csv) has columns: image_path, label, source
— the same schema make_splits.py expects, just not yet train/val split.

Usage (via the entry script, preferred):
    uv run main.py download cifake
    uv run main.py download sid-set --limit-per-class 4000

Or directly:
    uv run python scripts/download_data.py cifake
    uv run python scripts/download_data.py sid-set --limit-per-class 4000

Credentials needed:
  - CIFAKE (Kaggle): a free Kaggle API token at ~/.kaggle/kaggle.json, or the
    KAGGLE_USERNAME / KAGGLE_KEY env vars. Get one from
    https://www.kaggle.com/settings -> "Create New Token".
  - SID_Set (Hugging Face): public, no login required unless HF changes access.
    If you hit a 401, run `uv run huggingface-cli login` first.

SID_Set is ~140GB total across its splits, so by default we *stream* it and
stop once --limit-per-class images per label have been saved, rather than
downloading the full dataset — see 5.3 "hackathon-scale, limited compute".
SID_Set label 2 ("tampered") is out of scope for this binary real-vs-AIGC
task (see problem statement 5.2) and is skipped by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aigc_detect.config import LABEL_AIGC, LABEL_REAL, RAW_DIR

CIFAKE_KAGGLE_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
SID_SET_HF_HANDLE = "saberzl/SID_Set"


def _write_index(source: str, records: list[tuple[str, int]]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index_path = RAW_DIR / f"{source}_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source"])
        for image_path, label in records:
            writer.writerow([image_path, label, source])
    print(f"[{source}] wrote {len(records)} rows -> {index_path}")
    return index_path


def download_cifake() -> Path:
    import kagglehub

    print(f"[cifake] downloading '{CIFAKE_KAGGLE_HANDLE}' via kagglehub "
          f"(requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)...")
    dataset_root = Path(kagglehub.dataset_download(CIFAKE_KAGGLE_HANDLE))
    print(f"[cifake] downloaded to {dataset_root}")

    # Known CIFAKE layout: {train,test}/{REAL,FAKE}/*.jpg
    label_dirs = {"REAL": LABEL_REAL, "FAKE": LABEL_AIGC}
    records: list[tuple[str, int]] = []
    for split_dir in dataset_root.rglob("*"):
        if split_dir.is_dir() and split_dir.name in label_dirs:
            label = label_dirs[split_dir.name]
            for img_path in split_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    records.append((str(img_path.resolve()), label))

    if not records:
        raise RuntimeError(
            f"No images found under {dataset_root} — CIFAKE's folder layout may have "
            "changed; inspect the download and adjust download_cifake()."
        )
    return _write_index("cifake", records)


def download_sid_set(limit_per_class: int = 4000, include_tampered: bool = False, split: str = "train") -> Path:
    from datasets import load_dataset

    out_dir = RAW_DIR / "sid_set"
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "aigc").mkdir(parents=True, exist_ok=True)

    label_map = {0: (LABEL_REAL, "real"), 1: (LABEL_AIGC, "aigc")}
    if include_tampered:
        label_map[2] = (LABEL_AIGC, "tampered")  # tampered treated as non-authentic if included

    print(f"[sid_set] streaming '{SID_SET_HF_HANDLE}' split='{split}', "
          f"capping at {limit_per_class} images per class...")
    ds = load_dataset(SID_SET_HF_HANDLE, split=split, streaming=True)

    counts = {label: 0 for label, _ in label_map.values()}
    records: list[tuple[str, int]] = []

    for example in ds:
        raw_label = example.get("label")
        if raw_label not in label_map:
            continue
        norm_label, subdir_hint = label_map[raw_label]
        if counts[norm_label] >= limit_per_class:
            if all(c >= limit_per_class for c in counts.values()):
                break
            continue

        img = example["image"]  # PIL.Image (datasets decodes automatically)
        img_id = example.get("img_id", f"{subdir_hint}_{counts[norm_label]:06d}")
        subdir = "real" if norm_label == LABEL_REAL else "aigc"
        out_path = out_dir / subdir / f"{img_id}.jpg"
        if not out_path.exists():
            img.convert("RGB").save(out_path, format="JPEG", quality=95)

        records.append((str(out_path.resolve()), norm_label))
        counts[norm_label] += 1

    print(f"[sid_set] saved counts: {counts}")
    if not records:
        raise RuntimeError(
            "No SID_Set images were saved. The HF dataset schema may have changed, or "
            "it may require `huggingface-cli login`. See module docstring."
        )
    return _write_index("sid_set", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)

    sub.add_parser("cifake", help="Download CIFAKE (Kaggle) in full (~100MB, 60k train + 20k test images).")

    sid = sub.add_parser("sid-set", help="Stream a capped subset of SID_Set (HuggingFace) real/AIGC images.")
    sid.add_argument("--limit-per-class", type=int, default=4000,
                      help="Max images to save per class (default: 4000).")
    sid.add_argument("--split", default="train", choices=["train", "validation"],
                      help="Which SID_Set split to stream from (default: train).")
    sid.add_argument("--include-tampered", action="store_true",
                      help="Also include label=2 (tampered) images, mapped to the AIGC class.")

    args = parser.parse_args()
    if args.dataset == "cifake":
        download_cifake()
    elif args.dataset == "sid-set":
        download_sid_set(limit_per_class=args.limit_per_class, include_tampered=args.include_tampered, split=args.split)


if __name__ == "__main__":
    main()
