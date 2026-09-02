"""The `kagglehub` backend: a Kaggle mirror preferred over a slow official host.

WHY KAGGLE AT ALL. `download_demo_val.py` measured the official COCO val2017
S3 bucket throttled to ~12kB/s on this network (an 18+ hour ETA for 815MB)
against a Kaggle mirror's ~20MB/s -- the same throughput CIFAKE's Kaggle pull
already relied on. `coco_val2017` and `cifake` both go through kagglehub for
that reason; the official-host fallback each script carried is deliberately
NOT reproduced here (see `sources.yaml`'s comment on `coco_val2017`) -- one
fetcher per source id keeps `.pull_state.json`'s config hash meaning one
thing, and an S3 zip-download path would be a second, untested code path for
a fallback that has fired exactly once so far. Fetch the zip by hand and use
`manual` if the mirror is ever actually down.

RESUME STRATEGY: DISK-DIFF, NOT A CURSOR. `kagglehub.dataset_download()` is
already a cache -- calling it again after a kill re-verifies rather than
re-downloading. What this module resumes is the copy-into-`images/`-and-index
step on top of that, and a Kaggle dataset handle carries no natural stream
position to check pointer against, which is exactly the case
`existing_index_paths()`'s docstring in `base.py` names: diff what committed
rows the index already carries against what the layout would produce, and
skip the ones already there. A file that made it to disk (bytes copied) but
never reached the index -- the crash window `base.py`'s module docstring
describes -- is recopied for free (`shutil.copy2` over an identical file is
a no-op cost) and simply indexed on the next pass.

TWO LAYOUTS, BOTH MEASURED FROM THE MIRROR ITSELF. `flat` (coco_val2017) is
every file under the mirror matching a glob, flattened by filename. `label_dirs`
(cifake) is `download_data.py`'s known `{train,test}/{REAL,FAKE}/*.jpg` shape;
`preserve_subpath: true` keeps that structure under `images/` rather than
flattening, because REAL and FAKE filenames repeat across the two splits and
flattening would silently drop rows to a name collision -- the same
preserve_subpath comment `sources.yaml` carries on the `cifake` entry.
"""

from __future__ import annotations

import shutil
from pathlib import Path

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


def _iter_layout(mirror_root: Path, layout: dict):
    """Yield (src_file, dest_relpath_under_images, label_name_or_None), sorted.

    Sorted order matters twice over: it is what makes the disk-diff resume
    deterministic across runs, and (for `label_dirs`) it is the order a
    committed index/manifest built from this corpus carries.
    """
    kind = layout["kind"]
    if kind == "flat":
        pattern = layout.get("pattern", "*")
        for src in sorted(mirror_root.rglob(pattern)):
            if src.is_file():
                yield src, src.name, None
    elif kind == "label_dirs":
        label_dirs = layout["label_dirs"]
        preserve = layout.get("preserve_subpath", False)
        for src in sorted(mirror_root.rglob("*")):
            if not src.is_file():
                continue
            matched = next((part for part in src.relative_to(mirror_root).parts if part in label_dirs), None)
            if matched is None:
                continue
            dest_rel = str(src.relative_to(mirror_root)) if preserve else src.name
            yield src, dest_rel, label_dirs[matched]
    else:
        raise SystemExit(f"[kaggle] unknown layout.kind '{kind}'")


class KaggleFetcher:
    def pull(self, source, dest: CorpusPaths, state: PullState) -> PullResult:
        import kagglehub

        cfg = source.config
        resumed = state.rows_written > 0
        writer = IncrementalIndexWriter(dest, state)
        already = existing_index_paths(dest)

        logger.info("downloading/verifying kagglehub mirror '%s'...", cfg["handle"])
        mirror_root = Path(kagglehub.dataset_download(cfg["handle"]))

        layout = cfg["layout"]
        fixed_label = cfg.get("label")
        source_name = cfg.get("source_name", source.id)
        generator = cfg.get("generator") or ""

        scanned = 0
        for src, dest_rel, layout_label in _iter_layout(mirror_root, layout):
            scanned += 1
            dest_file = dest.images / dest_rel
            rel_str = _rel_posix(dest_file)
            if rel_str in already:
                continue

            if fixed_label is not None:
                label = _LABEL_NAME_TO_INT[fixed_label]
            elif layout_label is not None:
                label = _LABEL_NAME_TO_INT[layout_label]
            else:
                continue  # this file matched no label_dir -- not part of the corpus

            if not dest_file.exists():
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_file)

            writer.add({
                "image_path": rel_str,
                "label": label,
                "source": source_name,
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
        )
