from __future__ import annotations

from aigc_detect.cli._manifests import MANIFEST_CHOICES, _resolve_manifest
from aigc_detect.config import RANDOM_SEED


def cmd_embed(args):
    from aigc_detect.embed.embeddings import precompute_embeddings

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
    from aigc_detect.embed.views import precompute_view_embeddings

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


def register_embed(sub):
    p_embed = sub.add_parser(
        "embed", help='Precompute + cache pooled embeddings for a manifest under data/embeddings/.'
    )
    p_embed.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_embed.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES or None)
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--num-workers", type=int, default=4)
    p_embed.add_argument("--force", action="store_true", help="Recompute even if the cached .npz already exists.")
    p_embed.add_argument(
        "--limit", type=int, default=None, help="Only embed the first N rows of the manifest (for quick trials)."
    )
    p_embed.set_defaults(func=cmd_embed)


def register_embed_views(sub):
    p_embed_views = sub.add_parser(
        "embed-views",
        help="Precompute cached embeddings for every robustness view (5.2 table) of a manifest.",
    )
    p_embed_views.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_embed_views.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES or None)
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
