"""The fetcher dispatch table: `sources.yaml`'s `fetcher:` string -> a backend instance.

WHY A REGISTRY AND NOT AN IF/ELIF IN `pull.py`. `sources.py`'s docstring
promises "add a new HuggingFace source with eight lines of YAML and no
Python" -- true only if the CLI never has to learn a fetcher's name. This is
the one place that maps a string to code, so `cli/pull.py` stays generic over
every source and adding a *new kind* of backend (not just a new source) is
one entry here plus one new module, not a change to every place a fetcher
gets dispatched.

NO `http_archive` BACKEND. It was scoped for a source that would need a raw
HTTP/zip fetch with no HF or Kaggle mirror available. Nothing in
`registry/sources.yaml` currently declares it -- `coco_val2017`'s official-S3
fallback is documented, not automated (see `kaggle.py`'s module docstring for
why) -- so it is not written here. A backend with no caller is just an
untested code path; add it, and its test, the day a source actually needs it.
"""

from __future__ import annotations

from aigc_detect.data.fetchers.base import (
    CorpusPaths,
    Fetcher,
    IncrementalIndexWriter,
    PullResult,
    PullState,
    config_hash,
    existing_index_paths,
    open_state,
    verify_and_repair,
    wipe,
)
from aigc_detect.data.fetchers.hf import HFParquetFetcher, HFStreamingFetcher
from aigc_detect.data.fetchers.kaggle import KaggleFetcher
from aigc_detect.data.fetchers.manual import ManualFetcher

#: `sources.yaml`'s `fetcher:` value -> the backend that implements it.
FETCHERS: dict[str, Fetcher] = {
    "hf_streaming": HFStreamingFetcher(),
    "hf_parquet": HFParquetFetcher(),
    "kagglehub": KaggleFetcher(),
    "manual": ManualFetcher(),
}


def get_fetcher(name: str) -> Fetcher:
    try:
        return FETCHERS[name]
    except KeyError:
        raise SystemExit(
            f"[fetchers] unknown fetcher '{name}'. Registered: {sorted(FETCHERS)}\n"
            f"        A source's `fetcher:` in sources.yaml must name one of these."
        ) from None


__all__ = [
    "FETCHERS",
    "CorpusPaths",
    "Fetcher",
    "HFParquetFetcher",
    "HFStreamingFetcher",
    "IncrementalIndexWriter",
    "KaggleFetcher",
    "ManualFetcher",
    "PullResult",
    "PullState",
    "config_hash",
    "existing_index_paths",
    "get_fetcher",
    "open_state",
    "verify_and_repair",
    "wipe",
]
