from __future__ import annotations

import sys
from pathlib import Path

from aigc_detect.cli._manifests import MANIFEST_CHOICES, _resolve_manifest
from aigc_detect.config import RANDOM_SEED, ROOT_DIR


def cmd_eval_grid(args):
    from aigc_detect.evaluation.grid import evaluate_grid

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


def cmd_error_analysis(args):
    from aigc_detect.evaluation.error_analysis import run_error_analysis

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


def register_eval_grid(sub):
    p_eval_grid = sub.add_parser(
        "eval-grid", help="Score a trained head across every cached robustness view (5.5.4)."
    )
    p_eval_grid.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_eval_grid.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES or None)
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


def register_error_analysis(sub):
    p_err = sub.add_parser(
        "error-analysis",
        help="Concrete false positives/negatives + per-generator collapse ranking for a trained head (5.5.5). "
             "Needs embed-views cached for the same (backbone, manifest, sample-rows) first.",
    )
    p_err.add_argument("--backbone", required=True, help="Backbone registry key, e.g. pe-core-l.")
    p_err.add_argument("--manifest", required=True, choices=MANIFEST_CHOICES or None)
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
