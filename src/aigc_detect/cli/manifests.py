from __future__ import annotations

from aigc_detect.config import PROCESSED_DIR, RANDOM_SEED, TRAIN_MANIFEST, VAL_FRACTION, VAL_MANIFEST


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


def cmd_build_demo_val(_args):
    from scripts.make_demo_val import main as build_demo_val_main

    build_demo_val_main()


def cmd_build_ood(_args):
    from scripts.make_ood import main as build_ood_main

    build_ood_main()


def cmd_build_heldout(_args):
    from scripts.make_heldout import main as build_heldout_main

    build_heldout_main()


def register_split(sub):
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


def register_build_demo_val(sub):
    sub.add_parser(
        "build-demo-val", help="Merge demo-val indexes into data/demo_val/demo_val.csv (no split)."
    ).set_defaults(func=cmd_build_demo_val)


def register_build_ood(sub):
    sub.add_parser(
        "build-ood", help="Merge data/ood/*_index.csv into data/ood/ood.csv (evaluation only)."
    ).set_defaults(func=cmd_build_ood)


def register_build_heldout(sub):
    sub.add_parser(
        "build-heldout", help="Merge data/heldout/*_index.csv into data/heldout/heldout.csv (no split)."
    ).set_defaults(func=cmd_build_heldout)
