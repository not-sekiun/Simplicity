"""`aigc audit-data` -- the shortcut audit, run over the whole corpus registry.

Used to shell out to `scripts/audit_data.py`, which globbed `data/raw/`,
`data/aigc_ext/` and `data/real_ext/` -- three directories Tier 5 retired.
`aigc_detect.data.audit.run_registry_audit` is the same report (per-source
descriptive stats, a blind probe per source, one pooled probe) sourced from
`registry/corpora.yaml` instead, which is also how a single-label source
(most of the modern-generator pulls) gets a probe at all -- see that
module's docstring.
"""

from __future__ import annotations

from aigc_detect.config import RANDOM_SEED


def cmd_audit_data(args):
    from aigc_detect.data.audit import run_registry_audit

    n_suspect = run_registry_audit(sample=args.sample, use_transform=args.transform, seed=args.seed)
    if n_suspect:
        raise SystemExit(1)


def register_audit_data(sub):
    p_audit = sub.add_parser(
        "audit-data", help="Shortcut audit of the corpus registry (aspect ratio, blind probe canary)."
    )
    p_audit.add_argument("--sample", type=int, default=600, help="Max images sampled per (source, label) group.")
    p_audit.add_argument(
        "--transform",
        action="store_true",
        help="Run the blind probe on build_eval_transform() tensors instead of raw images.",
    )
    p_audit.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_audit.set_defaults(func=cmd_audit_data)
