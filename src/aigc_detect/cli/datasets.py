from __future__ import annotations

from aigc_detect.config import RANDOM_SEED


def cmd_download(args):
    from scripts.download_data import download_cifake, download_sid_set
    from scripts.download_tiny_genimage import download_tiny_genimage

    if args.dataset == "cifake":
        download_cifake()
    elif args.dataset == "sid-set":
        download_sid_set(limit_per_class=args.limit_per_class, include_tampered=args.include_tampered, split=args.split)
    elif args.dataset == "tiny-genimage":
        download_tiny_genimage(limit_per_split=args.limit_per_split, force=args.force)


def cmd_download_demo(args):
    from scripts.download_demo_val import download_coco_val2017, index_wildfake_dalle_advanced

    if args.which == "coco-val2017":
        download_coco_val2017()
    else:
        index_wildfake_dalle_advanced()


def cmd_download_ood(args):
    from scripts.download_ood_benchmark import download_ood_benchmark

    download_ood_benchmark(
        per_generator=args.per_generator,
        max_scan=args.max_scan,
        min_scan=args.min_scan,
        seed=args.seed,
        force=args.force,
    )


def register_download(sub):
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


def register_download_demo(sub):
    p_download_demo = sub.add_parser(
        "download-demo", help="Fetch the self-reported demo-val set (5.4) - never used for training."
    )
    ddsub = p_download_demo.add_subparsers(dest="which", required=True)
    ddsub.add_parser("coco-val2017")
    ddsub.add_parser("wildfake-dalle-advanced")
    p_download_demo.set_defaults(func=cmd_download_demo)


def register_download_ood(sub):
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
