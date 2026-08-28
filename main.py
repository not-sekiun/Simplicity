#!/usr/bin/env python
"""Entry script for the AIGC-detection project (TikTok TechJam 2026, Track 5).

Subcommands:
    check-env                        Verify PyTorch/CUDA setup and report dataset status.
    download cifake                  Download CIFAKE in full (Kaggle).
    download sid-set [--limit-per-class N]
                                      Stream a capped subset of SID_Set (HuggingFace).
    split [--val-fraction F] [--seed S]
                                      Build stratified data/processed/{train,val}.csv.
    preview-augment [--n N] [--out PATH]
                                      Save a grid image sanity-checking the augmentation
                                      pipeline (requires a train split to exist).

Examples:
    uv run main.py check-env
    uv run main.py download cifake
    uv run main.py download sid-set --limit-per-class 4000
    uv run main.py split
    uv run main.py preview-augment --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aigc_detect.config import PROCESSED_DIR, RAW_DIR, RANDOM_SEED, TRAIN_MANIFEST, VAL_FRACTION, VAL_MANIFEST  # noqa: E402


def cmd_check_env(_args):
    import torch

    print(f"torch:          {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device:         {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"vram:           {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA device visible to PyTorch. Training will run on CPU.")

    print()
    index_files = sorted(RAW_DIR.glob("*_index.csv"))
    if index_files:
        for f in index_files:
            print(f"raw index:      {f.name}")
    else:
        print("raw index:      none yet -- run `main.py download cifake` / `download sid-set`")

    print(f"train manifest: {'OK - ' + str(TRAIN_MANIFEST) if TRAIN_MANIFEST.exists() else 'missing -- run `main.py split`'}")
    print(f"val manifest:   {'OK - ' + str(VAL_MANIFEST) if VAL_MANIFEST.exists() else 'missing -- run `main.py split`'}")


def cmd_download(args):
    from scripts.download_data import download_cifake, download_sid_set

    if args.dataset == "cifake":
        download_cifake()
    elif args.dataset == "sid-set":
        download_sid_set(limit_per_class=args.limit_per_class, include_tampered=args.include_tampered, split=args.split)


def cmd_split(args):
    from scripts.make_splits import load_indexes, stratified_split

    df = load_indexes()
    print(f"[split] loaded {len(df)} indexed images across sources: {df['source'].value_counts().to_dict()}")
    train_df, val_df = stratified_split(df, args.val_fraction, args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_MANIFEST, index=False)
    val_df.to_csv(VAL_MANIFEST, index=False)
    print(f"[split] train: {len(train_df)} rows -> {TRAIN_MANIFEST}")
    print(f"[split] val:   {len(val_df)} rows -> {VAL_MANIFEST}")


def cmd_preview_augment(args):
    from aigc_detect.dataset import ManifestImageDataset
    from aigc_detect.transforms import build_train_transform
    from PIL import Image
    import torchvision.transforms.v2.functional as F

    if not TRAIN_MANIFEST.exists():
        print(f"No train manifest at {TRAIN_MANIFEST}. Run `main.py download ...` then `main.py split` first.")
        sys.exit(1)

    ds = ManifestImageDataset(TRAIN_MANIFEST, transform=build_train_transform())
    n = min(args.n, len(ds))
    tiles = []
    for i in range(n):
        tensor, _label = ds[i]
        # Undo normalization for viewing.
        img = F.to_pil_image((tensor * 0.229 + 0.485).clamp(0, 1))
        tiles.append(img)

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * cols, h * rows), "white")
    for idx, tile in enumerate(tiles):
        grid.paste(tile, ((idx % cols) * w, (idx // cols) * h))
    grid.save(args.out)
    print(f"Saved augmentation preview grid ({n} samples) -> {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-env", help="Verify PyTorch/CUDA setup and dataset status.").set_defaults(func=cmd_check_env)

    p_download = sub.add_parser("download", help="Download a raw dataset.")
    dsub = p_download.add_subparsers(dest="dataset", required=True)
    dsub.add_parser("cifake")
    sid = dsub.add_parser("sid-set")
    sid.add_argument("--limit-per-class", type=int, default=4000)
    sid.add_argument("--split", default="train", choices=["train", "validation"])
    sid.add_argument("--include-tampered", action="store_true")
    p_download.set_defaults(func=cmd_download)

    p_split = sub.add_parser("split", help="Build stratified train/val manifests.")
    p_split.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p_split.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_split.set_defaults(func=cmd_split)

    p_preview = sub.add_parser("preview-augment", help="Save a grid image sanity-checking the aug pipeline.")
    p_preview.add_argument("--n", type=int, default=8)
    p_preview.add_argument("--out", default="augment_preview.png")
    p_preview.set_defaults(func=cmd_preview_augment)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
