"""The `manual` backend: index what is already on disk, or refuse with the exact steps.

WHY A FETCHER FOR SOMETHING THAT CANNOT BE FETCHED. `wildrf` is a paper's own
release with no API (arXiv:2406.09398); `wildfake_dalle_advanced` sits behind
ModelScope, which this network cannot reach at the API or SDK level even
though the page itself loads -- confirmed via both `curl` and the `modelscope`
Python SDK hanging indefinitely (see AGENTS.md). Leaving either out of
`sources.yaml` would make `aigc pull list` lie about what reproducing this
project's training pool actually requires, and would leave their `corpus.yaml`
records the only place recording provenance, with nothing to verify it against
the way `verify_and_repair` does for every other source. So `manual` fetches
nothing and indexes everything: it walks whatever already sits under
`data/corpora/<id>/images/`, in the shape the source's own `expected_layout`
declares, and if that directory is empty or absent it exits printing the
source's own `instructions:` text verbatim rather than inventing a message --
the same "refuse to guess" instinct as `verify_and_repair` on a missing file,
or `open_state`'s refusal to resume across a changed config.

TWO LAYOUT SHAPES.
  scan_dirs       a fixed, ordered list of label-and-source-carrying
                  subdirectories (`wildrf`). The order is the one
                  `corpus.py`'s `_scan_rows` walked this same corpus in before
                  its Tier 5 move -- preserved here because manifests built
                  from it were committed in that order, and re-deriving it in
                  a different order would silently reshuffle what a `--limit`
                  prefix selects downstream.
  flat_recursive  every file anywhere under the tree, sorted, one fixed label
                  (`wildfake_dalle_advanced`) -- there is no internal
                  structure worth preserving for a single-label pull.

RESUME IS THE SAME DISK-DIFF AS `kaggle.py`, and for the same reason: neither
source has a cursor to resume from, and a `.pull_state.json`
`rows_scanned`/`cursor` pair means nothing for a fetcher whose whole job is
walking a directory that was already fully present before the pull started.
`existing_index_paths()` is what makes re-running this idempotent instead of
appending duplicate rows.
"""

from __future__ import annotations

from aigc_detect.config import LABEL_AIGC, LABEL_REAL
from aigc_detect.data.fetchers.base import (
    CorpusPaths,
    IncrementalIndexWriter,
    PullResult,
    PullState,
    existing_index_paths,
)
from aigc_detect.data.relocate import _rel_posix
from aigc_detect.log import get_logger

logger = get_logger(__name__)

_LABEL_NAME_TO_INT = {"real": LABEL_REAL, "aigc": LABEL_AIGC}


def _is_present(dest: CorpusPaths) -> bool:
    return dest.images.is_dir() and any(dest.images.rglob("*"))


def _refuse(source, dest: CorpusPaths) -> None:
    cfg = source.config
    lines = [
        f"[pull] '{source.id}' is a manual source -- nothing under {dest.images} to index yet.",
        "",
        (cfg.get("instructions") or "(no `instructions:` recorded in sources.yaml)").strip(),
        "",
        f"Expected images under: {dest.images}",
    ]
    if cfg.get("source_url"):
        lines.append(f"Source: {cfg['source_url']}")
    lines.append(f"\nThen run: uv run aigc pull run {source.id}")
    raise SystemExit("\n".join(lines))


def _scan_dirs_rows(dest: CorpusPaths, layout: dict):
    exts = {e.lower() for e in layout["exts"]}
    for spec in layout["dirs"]:
        directory = dest.images / spec["dir"]
        if not directory.is_dir():
            logger.warning("expected_layout dir absent, skipped: %s", directory)
            continue
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                yield p, spec["label"], spec["source"], spec.get("generator") or ""


def _flat_recursive_rows(dest: CorpusPaths, layout: dict, cfg: dict):
    exts = {e.lower() for e in layout["exts"]}
    label_name = cfg["label"]
    source_name = cfg.get("source_name", "")
    generator = cfg.get("generator") or ""
    for p in sorted(dest.images.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p, label_name, source_name, generator


class ManualFetcher:
    def pull(self, source, dest: CorpusPaths, state: PullState) -> PullResult:
        if not _is_present(dest):
            _refuse(source, dest)

        cfg = source.config
        layout = cfg["expected_layout"]
        resumed = state.rows_written > 0
        writer = IncrementalIndexWriter(dest, state)
        already = existing_index_paths(dest)

        if layout["kind"] == "scan_dirs":
            rows = _scan_dirs_rows(dest, layout)
        elif layout["kind"] == "flat_recursive":
            rows = _flat_recursive_rows(dest, layout, cfg)
        else:
            raise SystemExit(f"[manual] unknown expected_layout.kind '{layout['kind']}'")

        scanned = 0
        for path, label_name, src_name, generator in rows:
            scanned += 1
            rel_str = _rel_posix(path)
            if rel_str in already:
                continue
            writer.add({
                "image_path": rel_str,
                "label": _LABEL_NAME_TO_INT[label_name],
                "source": src_name,
                "generator": generator,
            })
            already.add(rel_str)
            writer.maybe_checkpoint(rows_scanned=scanned, cursor={})

        writer.finish(rows_scanned=scanned, cursor={}, completed=True)
        return PullResult(
            rows_written=state.rows_written,
            rows_scanned=state.rows_scanned,
            resumed=resumed,
            completed=True,
            note="indexed from an already-present directory -- nothing was downloaded",
        )
