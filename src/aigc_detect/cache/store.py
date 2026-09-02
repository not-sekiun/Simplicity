"""Content-addressed embedding store: sharded float16 vectors, SQLite index.

An embedding is identified by exactly three things -- which image (content id),
which backbone (including its resolved checkpoint revision), and which view
(the transform's canonical spec). Nothing about a manifest, a row order, or a
filesystem path takes part. A manifest therefore stops being a cache key and
becomes a *query*: hand `gather` a list of image ids and get back a matrix plus
the ids that are missing.

That single change is what buys the three properties the old .npz layout could
not express:

  resumable          every committed batch is durable, so a killed run resumes
                     by asking which ids are absent rather than starting over
  incremental        change one image and exactly one row is recomputed; the
                     old layout held N x dim floats under one digest, so there
                     was no way to say "these 5,999 rows are still valid"
  relocatable        renaming the repo or moving AIGC_DATA_ROOT changes nothing,
                     because no identity is derived from a path

DURABILITY ORDER IS LOAD-BEARING. Vectors are appended to their shard and
fsync'd BEFORE the index transaction commits. Crash between the two and the
result is orphaned bytes at the tail of a shard -- wasted space, reclaimed by
`compact`, never incorrect. Commit the index first and a crash would leave rows
pointing at bytes that were never written, which is silent corruption. The
cheap failure is the one we choose.

WHY SQLITE. Durability under `kill -9` is the actual requirement, and a
transactional commit is what provides it. A directory of Parquet files would be
easier to eyeball, which is what `export` is for.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

# float16 throughout. Not an approximation: the forward pass runs under AMP, so
# the values already carry fp16 precision -- verified on the previous cache at
# max |float32 - float16| == 0. Halving the file size is free.
DTYPE = np.float16
ITEMSIZE = 2

# 256 shards per (backbone, view), keyed on the first byte of the image id.
# Content ids are uniformly distributed, so this spreads rows evenly with no
# bookkeeping, and keeps any single file to a few thousand vectors.
SHARD_HEX = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS backbones (
    bb_id       TEXT PRIMARY KEY,
    key         TEXT NOT NULL,
    checkpoint  TEXT NOT NULL,
    revision    TEXT,
    pinned      INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    native_res  INTEGER NOT NULL,
    norm        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS views (
    view_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    spec        TEXT NOT NULL,
    seed_scheme TEXT,
    stochastic  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rows_ (
    bb_id   TEXT NOT NULL,
    view_id TEXT NOT NULL,
    img_id  TEXT NOT NULL,
    shard   TEXT NOT NULL,
    offset_ INTEGER NOT NULL,
    PRIMARY KEY (bb_id, view_id, img_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_rows_bv ON rows_(bb_id, view_id);
"""


def backbone_id(
    key: str,
    checkpoint: str,
    revision: str | None,
    dim: int,
    native_res: int,
    norm_mean,
    norm_std,
) -> str:
    """Stable id for "which model produced this vector".

    Includes the resolved revision, so a silent upstream re-upload yields a
    different id and a cache miss rather than a quiet mix of two models'
    embeddings in one matrix. The old .npz recorded the checkpoint but never
    compared it, which made the pe-core-l revision pin decorative.
    """
    payload = json.dumps(
        {
            "key": key,
            "checkpoint": checkpoint,
            "revision": revision,
            "dim": dim,
            "native_res": native_res,
            "norm_mean": [round(float(x), 6) for x in norm_mean],
            "norm_std": [round(float(x), 6) for x in norm_std],
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def view_id(name: str, spec: str, seed_scheme: str | None) -> str:
    """Stable id for "what was done to the image".

    The spec string is the canonical description built by
    `build_robustness_views`, so editing a severity changes the id and
    invalidates only the views that actually changed -- the one part of the
    previous design that was already right, kept.

    `seed_scheme` participates only for stochastic views, where the seed is part
    of what the numbers mean. Folding it into deterministic views would
    needlessly invalidate `clean`, `jpeg_*` and friends whenever the scheme
    changes.
    """
    payload = spec if seed_scheme is None else f"{spec}|seed={seed_scheme}"
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def _next_generation(out_dir: Path, shard: str) -> str:
    """Next unused generation name for a shard being compacted.

    `000` -> `000.g1` -> `000.g2`. The index stores the shard name explicitly,
    so a rewritten shard does not have to keep the name its first byte implies;
    that indirection is what lets the new file exist alongside the old one until
    the index commit chooses between them.
    """
    base = shard.split(".")[0]
    used = {p.stem for p in out_dir.glob(f"{base}*.f16")}
    gen = 1
    while f"{base}.g{gen}" in used:
        gen += 1
    return f"{base}.g{gen}"


class EmbeddingStore:
    """Sharded vector store with a transactional index.

    Not thread-safe and single-writer by design; use `merge` to fold in a store
    built on another machine.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.vec_dir = self.root / "vec"
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.root / "index.sqlite")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")  # the durability we are paying for
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> EmbeddingStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- registration ---------------------------------------------------------

    def register_backbone(self, bb_id: str, *, key: str, checkpoint: str, revision: str | None,
                          dim: int, native_res: int, norm_mean, norm_std) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO backbones(bb_id, key, checkpoint, revision, pinned, dim, "
            "native_res, norm) VALUES(?,?,?,?,?,?,?,?)",
            (bb_id, key, checkpoint, revision, int(revision is not None), dim, native_res,
             json.dumps({"mean": list(map(float, norm_mean)), "std": list(map(float, norm_std))})),
        )
        self._conn.commit()

    def register_view(self, vid: str, *, name: str, spec: str, seed_scheme: str | None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO views(view_id, name, spec, seed_scheme, stochastic) VALUES(?,?,?,?,?)",
            (vid, name, spec, seed_scheme, int(seed_scheme is not None)),
        )
        self._conn.commit()

    def backbone_dim(self, bb_id: str) -> int | None:
        row = self._conn.execute("SELECT dim FROM backbones WHERE bb_id=?", (bb_id,)).fetchone()
        return row[0] if row else None

    # -- writing --------------------------------------------------------------

    def missing(self, bb_id: str, vid: str, img_ids: list[str]) -> list[str]:
        """Which of these ids have no vector yet, de-duplicated, order preserved.

        This is the whole of "resume": a killed run re-asks the question and
        computes only the gap.
        """
        have: set[str] = set()
        unique = list(dict.fromkeys(img_ids))
        for start in range(0, len(unique), 800):
            batch = unique[start : start + 800]
            q = ",".join("?" * len(batch))
            have.update(
                r[0] for r in self._conn.execute(
                    f"SELECT img_id FROM rows_ WHERE bb_id=? AND view_id=? AND img_id IN ({q})",
                    [bb_id, vid, *batch],
                )
            )
        return [i for i in unique if i not in have]

    def put_batch(self, bb_id: str, vid: str, img_ids: list[str], vectors: np.ndarray) -> int:
        """Append vectors and index them, durably.

        Bytes are written and fsync'd before the index commits -- see the module
        docstring on why that order is not interchangeable. Returns the number
        of rows newly written; ids already present are skipped, which makes this
        idempotent and therefore safe to re-run after an interrupted batch.
        """
        if len(img_ids) != len(vectors):
            raise ValueError(f"[store] {len(img_ids)} ids but {len(vectors)} vectors")
        if not img_ids:
            return 0

        todo = set(self.missing(bb_id, vid, img_ids))
        by_shard: dict[str, list[tuple[str, np.ndarray]]] = {}
        seen: set[str] = set()
        for iid, vec in zip(img_ids, vectors, strict=True):
            if iid not in todo or iid in seen:
                continue
            seen.add(iid)
            by_shard.setdefault(iid[:SHARD_HEX], []).append((iid, vec))

        if not by_shard:
            return 0

        out_dir = self.vec_dir / bb_id / vid
        out_dir.mkdir(parents=True, exist_ok=True)
        index_rows: list[tuple[str, str, str, str, int]] = []

        for shard, items in by_shard.items():
            shard_path = out_dir / f"{shard}.f16"
            dim = int(vectors.shape[1])
            base = shard_path.stat().st_size // (dim * ITEMSIZE) if shard_path.exists() else 0
            block = np.ascontiguousarray(
                np.stack([v for _, v in items]).astype(DTYPE, copy=False)
            )
            with open(shard_path, "ab") as fh:
                fh.write(block.tobytes())
                fh.flush()
                os.fsync(fh.fileno())
            for k, (iid, _) in enumerate(items):
                index_rows.append((bb_id, vid, iid, shard, base + k))

        with self._conn:  # one transaction; rolls back whole on failure
            self._conn.executemany(
                "INSERT OR IGNORE INTO rows_(bb_id, view_id, img_id, shard, offset_) VALUES(?,?,?,?,?)",
                index_rows,
            )
        return len(index_rows)

    def drop(self, bb_id: str, vid: str, img_ids: list[str] | None = None) -> int:
        """Forget rows, so the next run recomputes them. Returns rows removed.

        Only the index entry is deleted; the vector's bytes stay in their shard
        until `compact` runs. That is deliberate and it is the same trade the
        write path makes -- deleting from an append-only shard would mean
        rewriting it, and a crash mid-rewrite is the one failure this store
        refuses to risk. `--force` therefore costs disk until compaction, never
        correctness.
        """
        if img_ids is None:
            cur = self._conn.execute("DELETE FROM rows_ WHERE bb_id=? AND view_id=?", (bb_id, vid))
            removed = cur.rowcount
        else:
            removed = 0
            unique = list(dict.fromkeys(img_ids))
            for start in range(0, len(unique), 800):
                batch = unique[start : start + 800]
                q = ",".join("?" * len(batch))
                cur = self._conn.execute(
                    f"DELETE FROM rows_ WHERE bb_id=? AND view_id=? AND img_id IN ({q})",
                    [bb_id, vid, *batch],
                )
                removed += cur.rowcount
        self._conn.commit()
        return removed

    # -- reading --------------------------------------------------------------

    def gather(self, bb_id: str, vid: str, img_ids: list[str]) -> tuple[np.ndarray, list[str]]:
        """Return vectors for `img_ids` in the given order, plus what is missing.

        Rows for missing ids come back as NaN, so a caller that ignores the
        missing list gets obviously-broken numbers rather than plausible ones.
        """
        dim = self.backbone_dim(bb_id)
        if dim is None:
            return np.empty((0, 0), dtype=np.float32), list(dict.fromkeys(img_ids))

        located: dict[str, tuple[str, int]] = {}
        unique = list(dict.fromkeys(img_ids))
        for start in range(0, len(unique), 800):
            batch = unique[start : start + 800]
            q = ",".join("?" * len(batch))
            for iid, shard, off in self._conn.execute(
                f"SELECT img_id, shard, offset_ FROM rows_ WHERE bb_id=? AND view_id=? "
                f"AND img_id IN ({q})",
                [bb_id, vid, *batch],
            ):
                located[iid] = (shard, off)

        out = np.full((len(img_ids), dim), np.nan, dtype=np.float32)
        # Group reads by shard so each file is opened once.
        wanted: dict[str, list[tuple[int, int]]] = {}
        for row_i, iid in enumerate(img_ids):
            hit = located.get(iid)
            if hit is not None:
                wanted.setdefault(hit[0], []).append((row_i, hit[1]))

        out_dir = self.vec_dir / bb_id / vid
        for shard, pairs in wanted.items():
            data = np.fromfile(out_dir / f"{shard}.f16", dtype=DTYPE).reshape(-1, dim)
            for row_i, off in pairs:
                out[row_i] = data[off]

        missing = [i for i in unique if i not in located]
        return out, missing

    # -- housekeeping ---------------------------------------------------------

    def stats(self) -> dict:
        (n_rows,) = self._conn.execute("SELECT COUNT(*) FROM rows_").fetchone()
        per = self._conn.execute(
            "SELECT b.key, v.name, COUNT(*) FROM rows_ r "
            "JOIN backbones b ON b.bb_id = r.bb_id JOIN views v ON v.view_id = r.view_id "
            "GROUP BY b.key, v.name ORDER BY 3 DESC"
        ).fetchall()
        bytes_on_disk = sum(p.stat().st_size for p in self.vec_dir.rglob("*.f16"))
        return {"rows": n_rows, "bytes": bytes_on_disk, "per_backbone_view": per}

    def groups(self) -> list[tuple[str, str, str, str, int]]:
        """(bb_id, backbone key, view_id, view name, row count) for what is stored.

        The unit `export`, `verify` and `compact` all iterate over -- a
        (backbone, view) pair is the only grouping the store has, since nothing
        else about a vector is recorded.
        """
        return self._conn.execute(
            "SELECT r.bb_id, b.key, r.view_id, v.name, COUNT(*) FROM rows_ r "
            "JOIN backbones b ON b.bb_id = r.bb_id JOIN views v ON v.view_id = r.view_id "
            "GROUP BY r.bb_id, r.view_id ORDER BY b.key, v.name"
        ).fetchall()

    def orphan_bytes(self) -> int:
        """Bytes sitting in shards that no index row points at.

        These accumulate exactly one way: a crash after a shard was fsync'd and
        before its index transaction committed (see the module docstring on why
        that is the order we chose). Counting them is cheap -- file size against
        the indexed row count per shard -- so `status` can report the number
        rather than leaving it to be guessed at.
        """
        total = 0
        for bb_id, _key, vid, _name, _n in self.groups():
            dim = self.backbone_dim(bb_id)
            if not dim:
                continue
            used = dict(self._conn.execute(
                "SELECT shard, COUNT(*) FROM rows_ WHERE bb_id=? AND view_id=? GROUP BY shard",
                (bb_id, vid),
            ))
            out_dir = self.vec_dir / bb_id / vid
            for shard_path in out_dir.glob("*.f16"):
                on_disk = shard_path.stat().st_size // (dim * ITEMSIZE)
                total += max(0, on_disk - used.get(shard_path.stem, 0)) * dim * ITEMSIZE
        return total

    def compact(self, *, dry_run: bool = False) -> dict:
        """Rewrite shards carrying orphaned bytes, reclaiming the space.

        SAFE UNDER kill -9, by the same argument as `put_batch` and in the same
        direction. Each shard is rewritten under a NEW generation name, the
        index is re-pointed at it in one transaction, and only then is the old
        file unlinked. A crash before the commit leaves the new file orphaned
        and the old one still authoritative; a crash after it leaves the old
        file orphaned and the new one authoritative. Neither state has an index
        row pointing at bytes that were never written, which is the one outcome
        that would be silently wrong.

        Rewriting in place would be less code and would corrupt the store on any
        crash mid-write, since the offsets already committed in the index would
        no longer describe the file.
        """
        reclaimed = 0
        rewritten = 0
        for bb_id, _key, vid, _name, _n in self.groups():
            dim = self.backbone_dim(bb_id)
            if not dim:
                continue
            out_dir = self.vec_dir / bb_id / vid
            by_shard: dict[str, list[tuple[str, int]]] = {}
            for iid, shard, off in self._conn.execute(
                "SELECT img_id, shard, offset_ FROM rows_ WHERE bb_id=? AND view_id=?", (bb_id, vid)
            ):
                by_shard.setdefault(shard, []).append((iid, off))

            # Driven by the FILES, not by the index: a shard every one of whose
            # rows was dropped has no index entry left to lead us to it, and is
            # exactly the shard with the most to reclaim.
            for shard_path in sorted(out_dir.glob("*.f16")):
                shard = shard_path.stem
                items = by_shard.get(shard, [])
                on_disk = shard_path.stat().st_size // (dim * ITEMSIZE)
                if on_disk <= len(items):
                    continue  # nothing orphaned in this shard
                reclaimed += (on_disk - len(items)) * dim * ITEMSIZE
                rewritten += 1
                if dry_run:
                    continue
                if not items:
                    shard_path.unlink()  # wholly orphaned; no index row to re-point
                    continue

                items.sort(key=lambda t: t[1])
                data = np.fromfile(shard_path, dtype=DTYPE).reshape(-1, dim)
                kept = np.ascontiguousarray(data[[off for _iid, off in items]])
                new_shard = _next_generation(out_dir, shard)
                new_path = out_dir / f"{new_shard}.f16"
                with open(new_path, "wb") as fh:
                    fh.write(kept.tobytes())
                    fh.flush()
                    os.fsync(fh.fileno())
                with self._conn:
                    self._conn.executemany(
                        "UPDATE rows_ SET shard=?, offset_=? WHERE bb_id=? AND view_id=? AND img_id=?",
                        [(new_shard, k, bb_id, vid, iid) for k, (iid, _off) in enumerate(items)],
                    )
                shard_path.unlink()
        return {"shards_rewritten": rewritten, "bytes_reclaimed": reclaimed, "dry_run": dry_run}

    def merge(self, other_root: str | Path) -> int:
        """Fold another store into this one (see scripts/worker.py).

        Distributed embedding used to require both machines to check the repo
        out at an identical absolute path, because the cache key was a hash of
        that path. Content addressing removes the constraint entirely: two
        stores built anywhere can be merged by copying rows neither side has.
        """
        other = EmbeddingStore(other_root)
        try:
            copied = 0
            for bb_id, key, ckpt, rev, _pin, dim, res, norm in other._conn.execute(
                "SELECT bb_id, key, checkpoint, revision, pinned, dim, native_res, norm FROM backbones"
            ):
                n = json.loads(norm)
                self.register_backbone(bb_id, key=key, checkpoint=ckpt, revision=rev, dim=dim,
                                       native_res=res, norm_mean=n["mean"], norm_std=n["std"])
            for vid, name, spec, seed, _sto in other._conn.execute(
                "SELECT view_id, name, spec, seed_scheme, stochastic FROM views"
            ):
                self.register_view(vid, name=name, spec=spec, seed_scheme=seed)
            pairs = other._conn.execute("SELECT DISTINCT bb_id, view_id FROM rows_").fetchall()
            for bb_id, vid in pairs:
                ids = [r[0] for r in other._conn.execute(
                    "SELECT img_id FROM rows_ WHERE bb_id=? AND view_id=?", (bb_id, vid)
                )]
                need = self.missing(bb_id, vid, ids)
                if not need:
                    continue
                vecs, absent = other.gather(bb_id, vid, need)
                if absent:
                    raise SystemExit(f"[store] source store is inconsistent: {len(absent)} indexed "
                                     f"rows have no vectors ({bb_id}/{vid})")
                copied += self.put_batch(bb_id, vid, need, vecs)
            return copied
        finally:
            other.close()
