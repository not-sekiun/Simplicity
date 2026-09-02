"""Thin shim -- the real audit now lives in `aigc_detect.data.audit`.

Kept so `uv run python scripts/audit_data.py` still works from muscle memory;
`uv run aigc audit-data` is the supported entry point (see `cli/audit.py`).
Tier 6 moved the logic off `data/raw` / `data/aigc_ext` / `data/real_ext`
(none of which exist since the Tier 5 corpus move) onto the corpus registry --
see `aigc_detect.data.audit`'s module docstring for why, and for the SD3 /
depth-map-pexels incidents this check exists to catch before training, not
after.
"""

from __future__ import annotations

import argparse

from aigc_detect.config import RANDOM_SEED
from aigc_detect.data.audit import run_registry_audit


def run_audit(sample: int, use_transform: bool, seed: int = RANDOM_SEED) -> None:
    run_registry_audit(sample=sample, use_transform=use_transform, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=600, help="Max images sampled per (source, label) group.")
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Run the blind probe on build_eval_transform() tensors instead of raw images.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_audit(args.sample, args.transform, args.seed)


if __name__ == "__main__":
    main()
