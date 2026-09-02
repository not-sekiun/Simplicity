"""Dump the store to files anything can read.

The store's two virtues -- a transactional index and 256 append-only shards --
are exactly what make it hard to eyeball. `sqlite3` and a hex editor are not the
tools you want when the question is "did the OOD tier actually get embedded
under the pinned revision, and how many rows".

So: `index.csv`, one row per stored vector with the backbone key, the view name
and the content id spelled out, plus (with ``--vectors``) a plain ``.npy``
matrix and a matching ``ids.txt`` per (backbone, view) group. Both formats are
readable by pandas, numpy, R, Excel and `grep`, and neither is a format this
project has to keep working -- the store stays authoritative, and an export is a
snapshot you throw away.

This is the answer to "SQLite is opaque" recorded in the refactor plan's
decision list, and the reason that objection did not have to change the storage
design.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aigc_detect.cache.store import EmbeddingStore
from aigc_detect.log import get_logger

logger = get_logger(__name__)


def export(
    store: EmbeddingStore,
    out_dir: str | Path,
    *,
    vectors: bool = False,
    backbone: str | None = None,
    view: str | None = None,
) -> dict:
    """Write `index.csv` (and optionally vectors) for the selected groups.

    `backbone`/`view` filter by the human-readable names, not ids -- this
    command exists to be typed by a person looking at `cache status`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = [
        g for g in store.groups()
        if (backbone is None or g[1] == backbone) and (view is None or g[3] == view)
    ]
    if not groups:
        raise SystemExit(
            f"[export] no stored rows match backbone={backbone!r} view={view!r}. "
            f"Run `uv run aigc cache status` to see what the store holds."
        )

    index_path = out / "index.csv"
    written_rows = 0
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["backbone", "view", "bb_id", "view_id", "img_id", "shard", "offset"])
        for bb_id, key, vid, name, _n in groups:
            for iid, shard, off in store._conn.execute(
                "SELECT img_id, shard, offset_ FROM rows_ WHERE bb_id=? AND view_id=? "
                "ORDER BY img_id",
                (bb_id, vid),
            ):
                writer.writerow([key, name, bb_id, vid, iid, shard, off])
                written_rows += 1

    files = [index_path]
    if vectors:
        for bb_id, key, vid, name, _n in groups:
            ids = [r[0] for r in store._conn.execute(
                "SELECT img_id FROM rows_ WHERE bb_id=? AND view_id=? ORDER BY img_id", (bb_id, vid)
            )]
            matrix, missing = store.gather(bb_id, vid, ids)
            if missing:
                # Indexed but unreadable: the shard file is gone or truncated.
                raise SystemExit(
                    f"[export] {key}/{name}: {len(missing)} indexed rows have no vector on disk. "
                    f"The store is damaged; do not trust this export."
                )
            stem = f"{key}__{name}"
            np.save(out / f"{stem}.npy", matrix.astype(np.float16))
            (out / f"{stem}.ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            files += [out / f"{stem}.npy", out / f"{stem}.ids.txt"]
            logger.info("exported %s: %d x %d", stem, matrix.shape[0], matrix.shape[1])

    return {"groups": len(groups), "rows": written_rows, "files": [str(p) for p in files],
            "out_dir": str(out)}
