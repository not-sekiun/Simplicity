from __future__ import annotations

import sys
from pathlib import Path

from aigc_detect.config import ROOT_DIR


def cmd_predict(args):
    from aigc_detect.inference.predict import run_inference

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


def register_predict(sub):
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
