from __future__ import annotations

import sys

from aigc_detect.cli._manifests import _resolve_manifest
from aigc_detect.config import RANDOM_SEED, TRAIN_EXT_MANIFEST, TRAIN_MANIFEST, VAL_MANIFEST


def cmd_train_head_views(args):
    from aigc_detect.embed.views import cache_stem
    from aigc_detect.train.probe import (
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


def cmd_train_head(args):
    from aigc_detect.embed.embeddings import embeddings_path
    from aigc_detect.train.probe import train_head

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


def register_train_head_views(sub):
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


def register_train_head(sub):
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
