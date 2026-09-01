from __future__ import annotations

from aigc_detect.config import RANDOM_SEED


def cmd_audit_data(args):
    from scripts.audit_data import run_audit

    run_audit(sample=args.sample, use_transform=args.transform, seed=args.seed)


def register_audit_data(sub):
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
