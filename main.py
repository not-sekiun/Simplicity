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
    download-demo coco-val2017       Download the self-reported demo-val "real" half.
    download-demo wildfake-dalle-advanced
                                      Index the demo-val "AIGC" half (manual fetch required
                                      first - see scripts/download_demo_val.py docstring).
    build-demo-val                   Merge demo-val indexes into data/demo_val/demo_val.csv.
                                      NEVER used for training - see 5.4 in the brief.
    audit-data [--sample N] [--transform]
                                      Shortcut audit of data/raw/*_index.csv: per-source
                                      stats + a blind-probe canary for label shortcuts
                                      (e.g. aspect ratio). --transform runs the probe on
                                      build_eval_transform() tensors instead of raw images.
    list-backbones                   List registered frozen-backbone keys (see
                                      src/aigc_detect/backbones.py).
    embed --backbone KEY --manifest {train,val,demo-val} [--force] [--limit N]
                                      Precompute + cache pooled embeddings for a manifest
                                      under data/embeddings/. Implements the "Simplicity
                                      Prevails" (arXiv:2602.01738) preprocessing recipe.
                                      demo-val is EVALUATION ONLY (see 5.4).
    train-head --backbone KEY [--head linear|mlp] [--epochs E] [--lr LR]
               [--batch-size B]
                                      Train a classifier head on cached embeddings for
                                      KEY (run `embed` for both train and val first).

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

from aigc_detect.config import (  # noqa: E402
    DEMO_VAL_DIR,
    DEMO_VAL_MANIFEST,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TRAIN_MANIFEST,
    VAL_FRACTION,
    VAL_MANIFEST,
)


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

    print()
    demo_index_files = sorted(DEMO_VAL_DIR.glob("*_index.csv")) if DEMO_VAL_DIR.exists() else []
    if demo_index_files:
        for f in demo_index_files:
            print(f"demo-val index: {f.name}")
    else:
        print("demo-val index: none yet -- run `main.py download-demo coco-val2017`")
    print(f"demo-val manifest (self-reported ONLY, never trained on): "
          f"{'OK - ' + str(DEMO_VAL_MANIFEST) if DEMO_VAL_MANIFEST.exists() else 'missing -- run `main.py build-demo-val`'}")


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


def cmd_download_demo(args):
    from scripts.download_demo_val import download_coco_val2017, index_wildfake_dalle_advanced

    if args.which == "coco-val2017":
        download_coco_val2017()
    else:
        index_wildfake_dalle_advanced()


def cmd_build_demo_val(_args):
    from scripts.make_demo_val import main as build_demo_val_main

    build_demo_val_main()


def cmd_audit_data(args):
    from scripts.audit_data import run_audit

    run_audit(sample=args.sample, use_transform=args.transform, seed=args.seed)


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


def cmd_list_backbones(_args):
    from aigc_detect.backbones import BACKBONE_REGISTRY, list_backbones

    for key in list_backbones():
        entry = BACKBONE_REGISTRY[key]
        print(
            f"{key:12s} checkpoint={entry['checkpoint']:55s} loader={entry['loader']:12s} "
            f"pooled_dim={entry['pooled_dim']:5d} native_res={entry['native_res']}"
        )


def cmd_embed(args):
    from aigc_detect.embed import precompute_embeddings

    # demo-val is embeddable for EVALUATION ONLY (brief 5.4 forbids training on
    # it). train_head never looks at it -- it hardcodes TRAIN_MANIFEST/VAL_MANIFEST.
    manifests = {"train": TRAIN_MANIFEST, "val": VAL_MANIFEST, "demo-val": DEMO_VAL_MANIFEST}
    manifest = manifests[args.manifest]
    if not manifest.exists():
        hint = "build-demo-val" if args.manifest == "demo-val" else "split"
        print(f"No {args.manifest} manifest at {manifest}. Run `main.py {hint}` first.")
        sys.exit(1)

    precompute_embeddings(
        manifest_path=manifest,
        backbone_key=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
        limit=args.limit,
    )


def cmd_train_head(args):
    from aigc_detect.embed import embeddings_path
    from aigc_detect.train_head import train_head

    train_npz = embeddings_path(args.backbone, TRAIN_MANIFEST)
    val_npz = embeddings_path(args.backbone, VAL_MANIFEST)
    for p, name in [(train_npz, "train"), (val_npz, "val")]:
        if not p.exists():
            print(f"No cached {name} embeddings at {p}. Run `main.py embed --backbone {args.backbone} "
                  f"--manifest {name}` first.")
            sys.exit(1)

    train_head(
        train_npz=train_npz,
        val_npz=val_npz,
        backbone_key=args.backbone,
        head_kind=args.head,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )


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

    p_download_demo = sub.add_parser(
        "download-demo", help="Fetch the self-reported demo-val set (5.4) - never used for training."
    )
    ddsub = p_download_demo.add_subparsers(dest="which", required=True)
    ddsub.add_parser("coco-val2017")
    ddsub.add_parser("wildfake-dalle-advanced")
    p_download_demo.set_defaults(func=cmd_download_demo)

    sub.add_parser(
        "build-demo-val", help="Merge demo-val indexes into data/demo_val/demo_val.csv (no split)."
    ).set_defaults(func=cmd_build_demo_val)

    p_audit = sub.add_parser(
        "audit-data", help="Shortcut audit of data/raw/*_index.csv (aspect ratio, blind probe canary)."
    )
    p_audit.add_argument("--sample", type=int, default=600, help="Max images sampled per (source, label) group.")
    p_audit.add_argument(
        "--transform",
        action="store_true",
        help="Run the blind probe on build_eval_transform() tensors instead of raw images.",
    )
    p_audit.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_audit.set_defaults(func=cmd_audit_data)

    sub.add_parser(
        "list-backbones", help="List registered frozen-backbone keys (src/aigc_detect/backbones.py)."
    ).set_defaults(func=cmd_list_backbones)

    p_embed = sub.add_parser(
        "embed", help='Precompute + cache pooled embeddings for a manifest under data/embeddings/.'
    )
    p_embed.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_embed.add_argument("--manifest", required=True, choices=["train", "val", "demo-val"])
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--num-workers", type=int, default=4)
    p_embed.add_argument("--force", action="store_true", help="Recompute even if the cached .npz already exists.")
    p_embed.add_argument(
        "--limit", type=int, default=None, help="Only embed the first N rows of the manifest (for quick trials)."
    )
    p_embed.set_defaults(func=cmd_embed)

    p_train_head = sub.add_parser(
        "train-head", help="Train a classifier head on cached embeddings (run `embed` for train+val first)."
    )
    p_train_head.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_train_head.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p_train_head.add_argument("--epochs", type=int, default=2)
    p_train_head.add_argument("--lr", type=float, default=1e-3)
    p_train_head.add_argument("--batch-size", type=int, default=128)
    p_train_head.add_argument("--weight-decay", type=float, default=0.0)
    p_train_head.set_defaults(func=cmd_train_head)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
