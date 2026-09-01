#!/usr/bin/env python
"""Standalone inference entry point (deliverable 5.5.2 -- required CLI contract):

    uv run python predict.py --input_dir <dir> --output preds.json

Thin CLI wrapper only. All real logic lives in
src/aigc_detect/predict.py::run_inference, which this module shares with
`main.py predict` -- see that module's docstring for the preprocessing-parity
requirements this satisfies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aigc_detect.config import ROOT_DIR  # noqa: E402
from aigc_detect.predict import run_inference  # noqa: E402

# Keep in sync with main.py and demo/server.py -- this is the graded 5.5.2
# entry point, so a stale default here scores the submission with the wrong head.
DEFAULT_HEAD = ROOT_DIR / "models" / "pe-core-l__linear__allsev_e1.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Directory to recurse for images (jpg/jpeg/png/webp/bmp).")
    parser.add_argument("--output", required=True, help="Path to write the JSON predictions array to.")
    parser.add_argument(
        "--head", default=str(DEFAULT_HEAD),
        help=f"Head checkpoint path (default: {DEFAULT_HEAD.name}).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    run_inference(
        input_dir=args.input_dir,
        head_path=args.head,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
