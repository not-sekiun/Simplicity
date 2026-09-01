#!/usr/bin/env python
"""Entry script for the AIGC-detection project (TikTok TechJam 2026, Track 5).

Subcommands:
    check-env                        Verify PyTorch/CUDA setup and report dataset status.
    download cifake                  Download CIFAKE in full (Kaggle).
    download sid-set [--limit-per-class N]
                                      Stream a capped subset of SID_Set (HuggingFace).
    download tiny-genimage [--limit-per-split N] [--force]
                                      Download Tiny-GenImage (HuggingFace): HF "train" ->
                                      data/raw/ (training pool), HF "validation" ->
                                      data/heldout/ (cross-generator test set, never
                                      trained on). Re-encodes every image as JPEG q95.
    split [--val-fraction F] [--seed S] [--exclude-source SOURCE ...]
          [--max-per-source N] [--holdout-generators G [G ...]]
                                      Build stratified data/processed/{train,val}.csv from
                                      data/raw/*_index.csv only (never data/heldout/ or
                                      data/demo_val/).
    preview-augment [--n N] [--out PATH]
                                      Save a grid image sanity-checking the augmentation
                                      pipeline (requires a train split to exist).
    download-demo coco-val2017       Download the self-reported demo-val "real" half.
    download-demo wildfake-dalle-advanced
                                      Index the demo-val "AIGC" half (manual fetch required
                                      first - see scripts/download_demo_val.py docstring).
    build-demo-val                   Merge demo-val indexes into data/demo_val/demo_val.csv.
                                      NEVER used for training - see 5.4 in the brief.
    download-ood [--per-generator N] [--max-scan N] [--min-scan N] [--seed S]
                 [--force]
                                      Stream a capped, generator-balanced slice of
                                      TheKernel01/AIGC-Detection-Benchmark into data/ood/ --
                                      a deliberately HARD out-of-distribution tier
                                      (10 of 18 generators unseen in training). Evaluation
                                      only, never trained on.
    build-ood                        Merge data/ood/*_index.csv into data/ood/ood.csv.
                                      Evaluation only, never trained on.
    build-heldout                    Merge data/heldout/*_index.csv into
                                      data/heldout/heldout.csv. Cross-generator test set,
                                      NEVER used for training.
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
    embed-views --backbone KEY --manifest {train,val,heldout,demo-val}
                [--views V ...] [--force] [--limit N] [--sample-rows N]
                [--train-chains]
                                      Same, but for all 18 robustness views: clean,
                                      the 14 single-transform rows of the 5.2 table,
                                      and 3 chained rows. One decode per image feeds
                                      every view. Caches to
                                      <backbone>__<stem>__<view>.npz. This is the
                                      instrument for the AUC_robust half of the score.
                                      --sample-rows draws a label-balanced,
                                      source-proportional subsample (seeded, so every
                                      backbone faces the identical subset) and tags
                                      the cache stem with it -- the intended path for
                                      racing backbones cheaply. --train-chains also
                                      computes 4 extra trainchain_* views (train
                                      manifest only) -- augmentation material for
                                      train-head-views, never scored by eval-grid.
    train-head-views --backbone KEY [--train-sample-rows N] [--val-sample-rows N]
                      [--train-views V ...] [--with-chains] [--clean-only]
                      [--head linear|mlp] [--epochs E] [--lr LR]
                      [--batch-size B] [--weight-decay WD] [--out PATH]
                                      Train a head on cached CLEAN + DEGRADED
                                      embeddings (run `embed-views` for train and val
                                      first). The augmentation ablation: default trains
                                      on one severity per family and holds the rest,
                                      including all 3 scored chains, out for
                                      evaluation only. --with-chains additionally
                                      trains on the 4 trainchain_* composition views
                                      (--sample-rows must match what embed-views used).
    eval-grid --backbone KEY --manifest {train,val,heldout,demo-val} [--head PATH]
              [--sample-rows N] [--limit N]
                                      Score a trained head over every cached view:
                                      per-view AUC/balanced accuracy at one fixed
                                      threshold, AUC_robust three ways, the
                                      robustness gap, and single-vs-chained means.
                                      Deliverable 5.5.4. No GPU work.
    error-analysis --backbone KEY --manifest {train,val,heldout,demo-val,ood}
                    [--head PATH] [--sample-rows N] [--limit N] [--top-k N]
                                      Concrete false positives/negatives (most
                                      confident first) + a per-generator collapse
                                      ranking, at eval-grid's fixed threshold.
                                      Writes CSV + a markdown report + copied
                                      example images under reports/error_analysis/.
                                      Deliverable 5.5.5. No GPU work; needs
                                      embed-views cached for the same manifest first.
    train-head --backbone KEY [--head linear|mlp] [--epochs E] [--lr LR]
               [--batch-size B]
                                      Train a classifier head on cached embeddings for
                                      KEY (run `embed` for both train and val first).
    predict --input_dir DIR --output preds.json [--head PATH]
                                      Run inference on a directory of images, emit
                                      JSON [{"image_path": str, "pred": float}, ...]
                                      where pred = P(AIGC). Deliverable 5.5.2. Also
                                      runnable standalone as `uv run python predict.py`.

Examples:
    uv run main.py check-env
    uv run main.py download cifake
    uv run main.py download sid-set --limit-per-class 4000
    uv run main.py download tiny-genimage --limit-per-split 40
    uv run main.py split
    uv run main.py preview-augment --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aigc_detect.config import (  # noqa: E402
    AIGC_MODERN_MANIFEST,
    DALLE3_HOLDOUT_MANIFEST,
    DEMO_VAL_DIR,
    DEMO_VAL_MANIFEST,
    HELDOUT_DIR,
    HELDOUT_MANIFEST,
    MIDJOURNEY_V6_MANIFEST,
    NANO_BANANA_MANIFEST,
    OOD_MANIFEST,
    PEXELS_REAL_MANIFEST,
    PHOTO_REAL_MANIFEST,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    ROOT_DIR,
    SID_REAL_MANIFEST,
    TRAIN_EXT_MANIFEST,
    TRAIN_MANIFEST,
    UNSPLASH_REAL_MANIFEST,
    VAL_FRACTION,
    VAL_MANIFEST,
    WILDRF_REAL_MANIFEST,
    WILDRF_TEST_MANIFEST,
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
    heldout_index_files = sorted(HELDOUT_DIR.glob("*_index.csv")) if HELDOUT_DIR.exists() else []
    if heldout_index_files:
        for f in heldout_index_files:
            print(f"heldout index:  {f.name}")
    else:
        print("heldout index:  none yet -- run `main.py download tiny-genimage`")
    print(f"heldout manifest (cross-generator test, never trained on): "
          f"{'OK - ' + str(HELDOUT_MANIFEST) if HELDOUT_MANIFEST.exists() else 'missing -- run `main.py build-heldout`'}")

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
    from scripts.download_tiny_genimage import download_tiny_genimage

    if args.dataset == "cifake":
        download_cifake()
    elif args.dataset == "sid-set":
        download_sid_set(limit_per_class=args.limit_per_class, include_tampered=args.include_tampered, split=args.split)
    elif args.dataset == "tiny-genimage":
        download_tiny_genimage(limit_per_split=args.limit_per_split, force=args.force)


def cmd_split(args):
    from scripts.make_splits import (
        cap_per_source,
        filter_sources,
        holdout_generators,
        load_indexes,
        stratified_split,
    )

    df = load_indexes()
    print(f"[split] loaded {len(df)} indexed images across sources: {df['source'].value_counts().to_dict()}")

    df = filter_sources(df, args.exclude_source)
    df = cap_per_source(df, args.max_per_source, args.seed)

    train_df, val_df = stratified_split(df, args.val_fraction, args.seed)
    train_df, val_df = holdout_generators(train_df, val_df, args.holdout_generators)

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


def cmd_download_ood(args):
    from scripts.download_ood_benchmark import download_ood_benchmark

    download_ood_benchmark(
        per_generator=args.per_generator,
        max_scan=args.max_scan,
        min_scan=args.min_scan,
        seed=args.seed,
        force=args.force,
    )


def cmd_build_ood(_args):
    from scripts.make_ood import main as build_ood_main

    build_ood_main()


def cmd_build_heldout(_args):
    from scripts.make_heldout import main as build_heldout_main

    build_heldout_main()


def cmd_audit_data(args):
    from scripts.audit_data import run_audit

    run_audit(sample=args.sample, use_transform=args.transform, seed=args.seed)


def cmd_preview_augment(args):
    import torchvision.transforms.v2.functional as F
    from PIL import Image

    from aigc_detect.dataset import ManifestImageDataset
    from aigc_detect.transforms import build_train_transform

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


# Single source of truth for --manifest. Four argparse choices lists used to
# carry their own hardcoded copy of these names, and they drifted the moment new
# corpora were added: _resolve_manifest knew about them, embed-views did not, so
# a valid manifest was rejected at the CLI boundary with a confusing error.
MANIFESTS = {
    "train": TRAIN_MANIFEST,
    "train-ext": TRAIN_EXT_MANIFEST,
    "sid-real": SID_REAL_MANIFEST,
    "photo-real": PHOTO_REAL_MANIFEST,
    "unsplash-real": UNSPLASH_REAL_MANIFEST,
    "pexels-real": PEXELS_REAL_MANIFEST,
    "wildrf-real": WILDRF_REAL_MANIFEST,
    "wildrf-test": WILDRF_TEST_MANIFEST,
    "nano-banana": NANO_BANANA_MANIFEST,
    "midjourney-v6": MIDJOURNEY_V6_MANIFEST,
    "dalle3-holdout": DALLE3_HOLDOUT_MANIFEST,
    "aigc-modern": AIGC_MODERN_MANIFEST,
    "val": VAL_MANIFEST,
    "heldout": HELDOUT_MANIFEST,
    "demo-val": DEMO_VAL_MANIFEST,
    "ood": OOD_MANIFEST,
}
MANIFEST_CHOICES = list(MANIFESTS)


def _resolve_manifest(name: str):
    """Map a --manifest choice to its path, exiting with a hint if it's missing.

    demo-val is embeddable for EVALUATION ONLY (brief 5.4 forbids training on
    it). train_head never looks at it -- it hardcodes TRAIN_MANIFEST/VAL_MANIFEST.
    """
    manifests = MANIFESTS
    manifest = manifests[name]
    if not manifest.exists():
        hint = {"demo-val": "build-demo-val", "heldout": "build-heldout",
                "ood": "download-ood` then `main.py build-ood",
                "train-ext": "python scripts/make_train_ext.py",
                "sid-real": "python scripts/make_sid_real.py",
                "photo-real": "python scripts/download_real_domains.py --merge",
                "wildrf-real": "python scripts/make_wildrf.py",
                "wildrf-test": "python scripts/make_wildrf.py",
                "nano-banana": "python scripts/download_aigc_modern.py --source nano-banana",
                "midjourney-v6": "python scripts/download_aigc_modern.py --source midjourney-v6",
                "dalle3-holdout": "python scripts/download_aigc_modern.py --source dalle3-holdout",
                "aigc-modern": "python scripts/download_aigc_modern.py --merge"}.get(name, "split")
        print(f"No {name} manifest at {manifest}. Run `main.py {hint}` first.")
        sys.exit(1)
    return manifest


def cmd_embed(args):
    from aigc_detect.embed import precompute_embeddings

    manifest = _resolve_manifest(args.manifest)

    precompute_embeddings(
        manifest_path=manifest,
        backbone_key=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
        limit=args.limit,
    )


def cmd_embed_views(args):
    from aigc_detect.embed_views import precompute_view_embeddings

    manifest = _resolve_manifest(args.manifest)
    precompute_view_embeddings(
        manifest_path=manifest,
        backbone_key=args.backbone,
        views=args.views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
        limit=args.limit,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
        include_train_chains=args.train_chains,
        dtype=args.dtype,
    )


def cmd_train_head_views(args):
    from aigc_detect.embed_views import cache_stem
    from aigc_detect.train_head import (
        TRAIN_VIEWS_ALL_SEVERITIES,
        TRAIN_VIEWS_DEFAULT,
        TRAIN_VIEWS_WITH_CHAINS,
        train_head_on_views,
    )

    train_manifest = TRAIN_EXT_MANIFEST if args.train_manifest == 'train-ext' else TRAIN_MANIFEST
    train_stem = cache_stem(train_manifest, sample_rows=args.train_sample_rows)
    val_stem = cache_stem(VAL_MANIFEST, sample_rows=args.val_sample_rows)

    # Extra manifests are concatenated onto the training rows, each keeping its
    # own cache stem and its own fingerprint check. The point is that adding
    # images does NOT invalidate the stem already computed for the base pool.
    extra = []
    for name in args.extra_train_manifest or ():
        m = _resolve_manifest(name)
        extra.append((cache_stem(m), m))
    views = tuple(args.train_views) if args.train_views else TRAIN_VIEWS_DEFAULT
    if args.with_chains:
        views = TRAIN_VIEWS_WITH_CHAINS
    if args.all_severities:
        # Checked after --with-chains because it is a superset of it: the ship
        # recipe is "every severity AND the training chains".
        views = TRAIN_VIEWS_ALL_SEVERITIES
    if args.clean_only:
        views = ("clean",)

    train_head_on_views(
        backbone_key=args.backbone,
        train_stem=train_stem,
        val_stem=val_stem,
        train_views=views,
        head_kind=args.head,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        out_path=args.out,
        train_manifest=train_manifest,
        train_sample_rows=args.train_sample_rows,
        val_manifest=VAL_MANIFEST,
        val_sample_rows=args.val_sample_rows,
        seed=args.seed,
        extra_train=extra,
        balance_classes=args.balance,
        exclude_generators=args.exclude_generators or (),
    )


def cmd_eval_grid(args):
    from aigc_detect.eval_grid import evaluate_grid

    manifest = _resolve_manifest(args.manifest)
    head_path = Path(args.head) if args.head else ROOT_DIR / "models" / f"{args.backbone}__{args.head_kind}.pt"
    if not head_path.exists():
        print(f"No head checkpoint at {head_path}. Run `main.py train-head --backbone {args.backbone}` first.")
        sys.exit(1)

    evaluate_grid(
        backbone_key=args.backbone,
        manifest_path=manifest,
        head_path=head_path,
        limit=args.limit,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
        by_generator=args.by_generator,
        out_csv=args.out,
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


def cmd_error_analysis(args):
    from aigc_detect.error_analysis import run_error_analysis

    manifest = _resolve_manifest(args.manifest)
    head_path = Path(args.head) if args.head else ROOT_DIR / "models" / f"{args.backbone}__{args.head_kind}.pt"
    if not head_path.exists():
        print(f"No head checkpoint at {head_path}. Run `main.py train-head --backbone {args.backbone}` first.")
        sys.exit(1)

    run_error_analysis(
        backbone_key=args.backbone,
        manifest_path=manifest,
        head_path=head_path,
        limit=args.limit,
        sample_rows=args.sample_rows,
        sample_seed=args.sample_seed,
        top_k=args.top_k,
        extra_views=tuple(args.extra_views or ()),
        out_dir=args.out_dir,
        copy_images=not args.no_copy_images,
    )


def cmd_predict(args):
    from aigc_detect.predict import run_inference

    head_path = Path(args.head) if args.head else ROOT_DIR / "models" / "pe-core-l__linear__allsev_e1.pt"
    if not head_path.exists():
        print(f"No head checkpoint at {head_path}. Pass --head <path> or train one first.")
        sys.exit(1)

    run_inference(
        input_dir=args.input_dir,
        head_path=head_path,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
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
    tgi = dsub.add_parser("tiny-genimage")
    tgi.add_argument("--limit-per-split", type=int, default=None,
                      help="Only download the first N images of each HF split (default: all).")
    tgi.add_argument("--force", action="store_true", help="Re-encode images even if already written.")
    p_download.set_defaults(func=cmd_download)

    p_split = sub.add_parser("split", help="Build stratified train/val manifests.")
    p_split.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    p_split.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_split.add_argument(
        "--exclude-source", action="append", default=None, metavar="SOURCE",
        help="Drop a source entirely before splitting (repeatable), e.g. --exclude-source sid_set.",
    )
    p_split.add_argument(
        "--max-per-source", type=int, default=None,
        help="Cap rows per source by random subsample (seeded), before splitting.",
    )
    p_split.add_argument(
        "--holdout-generators", nargs="+", default=None, metavar="GENERATOR",
        help="Move all rows for these generators out of train and into val (unseen-generator eval).",
    )
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

    p_dl_ood = sub.add_parser(
        "download-ood",
        help="Stream a capped, generator-balanced slice of AIGC-Detection-Benchmark into data/ood/.",
    )
    p_dl_ood.add_argument("--per-generator", type=int, default=250)
    p_dl_ood.add_argument("--max-scan", type=int, default=60_000)
    p_dl_ood.add_argument("--min-scan", type=int, default=2_000)
    p_dl_ood.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_dl_ood.add_argument("--force", action="store_true")
    p_dl_ood.set_defaults(func=cmd_download_ood)

    sub.add_parser(
        "build-ood", help="Merge data/ood/*_index.csv into data/ood/ood.csv (evaluation only)."
    ).set_defaults(func=cmd_build_ood)

    sub.add_parser(
        "build-heldout", help="Merge data/heldout/*_index.csv into data/heldout/heldout.csv (no split)."
    ).set_defaults(func=cmd_build_heldout)

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
    p_embed.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES)
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--num-workers", type=int, default=4)
    p_embed.add_argument("--force", action="store_true", help="Recompute even if the cached .npz already exists.")
    p_embed.add_argument(
        "--limit", type=int, default=None, help="Only embed the first N rows of the manifest (for quick trials)."
    )
    p_embed.set_defaults(func=cmd_embed)

    p_embed_views = sub.add_parser(
        "embed-views",
        help="Precompute cached embeddings for every robustness view (5.2 table) of a manifest.",
    )
    p_embed_views.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_embed_views.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES)
    p_embed_views.add_argument(
        "--views", nargs="+", default=None, metavar="VIEW",
        help="Only compute these views (default: all 18). E.g. --views clean blur_sigma2.0 chain_heavy",
    )
    p_embed_views.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size in IMAGES; the effective GPU batch is this x n_views (default 8).",
    )
    p_embed_views.add_argument("--num-workers", type=int, default=4)
    p_embed_views.add_argument("--force", action="store_true", help="Recompute even if a current cache exists.")
    p_embed_views.add_argument(
        "--limit", type=int, default=None,
        help="Only embed the first N rows. Prefer --sample-rows: a manifest prefix is not label-balanced.",
    )
    p_embed_views.add_argument(
        "--sample-rows", type=int, default=None, metavar="N",
        help="Embed a label-balanced, source-proportional subsample of N rows (seeded). "
             "Caches under a '-sN' stem so it coexists with the full run.",
    )
    p_embed_views.add_argument("--sample-seed", type=int, default=RANDOM_SEED)
    p_embed_views.add_argument(
        "--train-chains", action="store_true",
        help="Also compute the 4 trainchain_* augmentation views. Train manifest only -- they are "
             "never scored, so caching them for val/demo-val is waste.",
    )
    p_embed_views.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p_embed_views.set_defaults(func=cmd_embed_views)

    p_thv = sub.add_parser(
        "train-head-views",
        help="Train a head on cached CLEAN + DEGRADED embeddings (the augmentation ablation).",
    )
    p_thv.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_thv.add_argument("--train-sample-rows", type=int, default=None,
                       help="Match the --sample-rows used for `embed-views --manifest train`.")
    p_thv.add_argument("--val-sample-rows", type=int, default=None,
                       help="Match the --sample-rows used for `embed-views --manifest val`.")
    p_thv.add_argument("--train-views", nargs="+", default=None, metavar="VIEW",
                       help="Views to TRAIN on (default: one severity per family; the rest, "
                            "including all chains, stay held out and are only evaluated).")
    p_thv.add_argument("--train-manifest", default="train", choices=["train", "train-ext"],
                       help="Which training manifest to read cached views for (default: train).")
    p_thv.add_argument("--extra-train-manifest", nargs="+", default=None, metavar="NAME",
                       choices=["train-ext", "sid-real", "photo-real", "unsplash-real", "pexels-real", "wildrf-real",
                                "nano-banana", "midjourney-v6", "aigc-modern"],
                       help="Additional manifest(s) whose cached views are CONCATENATED onto the "
                            "training rows. Each keeps its own stem and fingerprint check, so the "
                            "base pool's cache is reused rather than recomputed. Embed them first "
                            "with the same --train-views/--with-chains selection.")
    p_thv.add_argument("--exclude-generators", nargs="+", default=None, metavar="NAME",
                       help="Drop training rows whose manifest `generator` matches (case-insensitive), "
                            "e.g. --exclude-generators BigGAN CycleGAN GauGAN StarGAN StyleGAN StyleGAN2. "
                            "Filters the cached rows in place -- no re-embedding -- and leaves the scaler "
                            "computed on the FULL clean view so the only variable is which rows train.")
    p_thv.add_argument("--balance", action="store_true",
                       help="Weight the loss so the two classes contribute equally (pos_weight = "
                            "n_real/n_aigc). Use when an extra manifest skews the label prior -- "
                            "otherwise a drop in FPR cannot be told apart from a shifted boundary.")
    p_thv.add_argument("--with-chains", action="store_true",
                       help="Also train on the 4 trainchain_* views (composition, built only from "
                            "severities already in the default set). The 3 SCORED chains stay held out.")
    p_thv.add_argument("--all-severities", action="store_true",
                       help="THE SHIPPING RECIPE. Train on all 19 views: every severity of "
                            "every degradation family plus the 4 trainchain_* views. Only the 3 "
                            "SCORED chains stay held out. Supersedes --with-chains. Embed the "
                            "manifests with the same view selection first.")
    p_thv.add_argument("--clean-only", action="store_true",
                       help="Control arm: train on the clean view alone, same images, same scaler.")
    p_thv.add_argument("--head", default="linear", choices=["linear", "mlp", "mlp2"])
    p_thv.add_argument("--epochs", type=int, default=2)
    p_thv.add_argument("--lr", type=float, default=1e-3)
    p_thv.add_argument("--batch-size", type=int, default=128)
    p_thv.add_argument("--weight-decay", type=float, default=0.0)
    p_thv.add_argument("--seed", type=int, default=RANDOM_SEED,
                       help="Seeds head init and DataLoader shuffle. Unseeded runs vary ~+/-0.0005 AUC.")
    p_thv.add_argument("--out", default=None, help="Checkpoint path (default: models/<backbone>__<kind>__<tag>.pt).")
    p_thv.set_defaults(func=cmd_train_head_views)

    p_eval_grid = sub.add_parser(
        "eval-grid", help="Score a trained head across every cached robustness view (5.5.4)."
    )
    p_eval_grid.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_eval_grid.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES)
    p_eval_grid.add_argument("--head", default=None, help="Head checkpoint (default: models/<backbone>__<kind>.pt).")
    p_eval_grid.add_argument("--head-kind", default="linear", choices=["linear", "mlp"],
                             help="Only used to locate the default checkpoint path.")
    p_eval_grid.add_argument("--limit", type=int, default=None, help="Match the --limit used for embed-views.")
    p_eval_grid.add_argument("--sample-rows", type=int, default=None,
                             help="Match the --sample-rows used for embed-views.")
    p_eval_grid.add_argument("--sample-seed", type=int, default=RANDOM_SEED)
    p_eval_grid.add_argument(
        "--by-generator", action="store_true",
        help="Break AUC down per generator, grouped by architecture family (diffusion vs GAN) "
             "and by trained/UNSEEN. Needs a manifest with a generator column.",
    )
    p_eval_grid.add_argument("--out", default=None, help="Per-view CSV path (default: reports/grid__*.csv).")
    p_eval_grid.set_defaults(func=cmd_eval_grid)

    p_err = sub.add_parser(
        "error-analysis",
        help="Concrete false positives/negatives + per-generator collapse ranking for a trained head (5.5.5). "
             "Needs embed-views cached for the same (backbone, manifest, sample-rows) first.",
    )
    p_err.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_err.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES)
    p_err.add_argument("--head", default=None, help="Head checkpoint (default: models/<backbone>__<kind>.pt).")
    p_err.add_argument("--head-kind", default="linear", choices=["linear", "mlp"],
                        help="Only used to locate the default checkpoint path.")
    p_err.add_argument("--limit", type=int, default=None, help="Match the --limit used for embed-views.")
    p_err.add_argument("--sample-rows", type=int, default=None, help="Match the --sample-rows used for embed-views.")
    p_err.add_argument("--sample-seed", type=int, default=RANDOM_SEED)
    p_err.add_argument("--top-k", type=int, default=8, help="Most-confident false positives/negatives per view.")
    p_err.add_argument("--extra-views", nargs="*", default=None,
                        help="Additional cached view names to dump examples for, beyond clean + the worst view.")
    p_err.add_argument("--out-dir", default=None, help="Output dir (default: reports/error_analysis/).")
    p_err.add_argument("--no-copy-images", action="store_true",
                        help="Skip copying example image files alongside the CSV/markdown report.")
    p_err.set_defaults(func=cmd_error_analysis)

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

    p_predict = sub.add_parser(
        "predict",
        help="Run inference on a directory of images, emit JSON [{image_path, pred}] (deliverable 5.5.2).",
    )
    p_predict.add_argument("--input_dir", required=True, help="Directory to recurse for images.")
    p_predict.add_argument("--output", required=True, help="Path to write the JSON predictions array to.")
    p_predict.add_argument(
        "--head", default=None,
        help="Head checkpoint path (default: models/pe-core-l__linear__allsev_e1.pt).",
    )
    p_predict.add_argument("--threshold", type=float, default=None,
                           help="Decision boundary for the flagged/not-flagged summary line "
                                "(default: predict.DECISION_THRESHOLD = 0.980, chosen on a held-out "
                                "WildRF split). Does NOT change the JSON, which always carries the "
                                "raw probability in 'pred' as the brief requires.")
    p_predict.add_argument("--batch-size", type=int, default=32)
    p_predict.add_argument("--num-workers", type=int, default=4)
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
