"""`download` / `download-demo` / `download-ood` -- kept working, re-pointed at the fetchers.

These three command groups used to shell out to `scripts/download_data.py`,
`scripts/download_tiny_genimage.py`, `scripts/download_demo_val.py` and
`scripts/download_ood_benchmark.py` directly. Tier 6 replaced all four with
declared sources in `registry/sources.yaml` and resumable backends under
`aigc_detect.data.fetchers` -- `aigc pull run <id>` is the real interface now,
and everything below is a thin, CLI-contract-preserving bridge onto it.

WHAT DELIBERATELY NARROWED. `download sid-set` used to take `--split` and
`--include-tampered`; the registered `sid_set` source is REALS ONLY, fixed
(docs/findings.md 1 -- the AIGC half is a labelled-composition shortcut, and
the fetcher-level `label_map: {0: real}` filter is what keeps a re-pull from
ever reopening it). Asking for anything else now raises rather than silently
downloading something this project no longer trains on.
"""

from __future__ import annotations

import dataclasses

from aigc_detect.config import RANDOM_SEED


def _run_pull(source_id: str, *, force: bool = False, limit: int | None = None):
    from aigc_detect.data.fetchers import CorpusPaths, get_fetcher, open_state
    from aigc_detect.data.sources import get_source

    source = get_source(source_id)
    if limit is not None:
        source = dataclasses.replace(source, config={**source.config, "cap": limit})
    dest = CorpusPaths.for_source(source_id)
    fetcher = get_fetcher(source.fetcher)
    state, _resumed = open_state(source, dest, force=force)
    result = fetcher.pull(source, dest, state)
    print(f"[download] '{source_id}': {result.rows_written:,} rows written "
          f"({'resumed' if result.resumed else 'fresh'}, completed={result.completed})")
    return result


def cmd_download(args):
    if args.dataset == "cifake":
        _run_pull("cifake")
    elif args.dataset == "sid-set":
        if args.split != "train" or args.include_tampered:
            raise SystemExit(
                "[download] sid-set is REALS-ONLY now (docs/findings.md 1) -- `--split validation` "
                "and `--include-tampered` are no longer supported. See registry/sources.yaml's "
                "sid_set entry."
            )
        _run_pull("sid_set", limit=args.limit_per_class)
    elif args.dataset == "tiny-genimage":
        _run_pull("tiny_genimage", force=args.force, limit=args.limit_per_split)
        _run_pull("tiny_genimage_heldout", force=args.force, limit=args.limit_per_split)


def cmd_download_demo(args):
    if args.which == "coco-val2017":
        _run_pull("coco_val2017")
    else:
        _run_pull("wildfake_dalle_advanced")


def cmd_download_ood(args):
    from aigc_detect.data.fetchers import CorpusPaths, get_fetcher, open_state
    from aigc_detect.data.sources import get_source

    source_id = "aigc_detect_bench"
    source = get_source(source_id)
    overrides = {
        "per_generator": args.per_generator,
        "max_scan": args.max_scan,
        "min_scan": args.min_scan,
        "seed": args.seed,
    }
    source = dataclasses.replace(source, config={**source.config, **overrides})
    dest = CorpusPaths.for_source(source_id)
    fetcher = get_fetcher(source.fetcher)
    state, _resumed = open_state(source, dest, force=args.force)
    result = fetcher.pull(source, dest, state)
    print(f"[download-ood] {result.rows_written:,} rows written "
          f"({'resumed' if result.resumed else 'fresh'}, completed={result.completed})")


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
        help="Stream a capped, generator-balanced slice of AIGC-Detection-Benchmark into data/corpora/.",
    )
    p_dl_ood.add_argument("--per-generator", type=int, default=250)
    p_dl_ood.add_argument("--max-scan", type=int, default=60_000)
    p_dl_ood.add_argument("--min-scan", type=int, default=2_000)
    p_dl_ood.add_argument("--seed", type=int, default=RANDOM_SEED)
    p_dl_ood.add_argument("--force", action="store_true")
    p_dl_ood.set_defaults(func=cmd_download_ood)
