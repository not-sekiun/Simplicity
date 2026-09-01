"""Content addressing for images, and a memo table so it stays cheap.

An embedding's identity should be a property of the *bytes that were embedded*,
not of where those bytes happened to live. The previous cache keyed on a SHA1
of absolute image paths, which had two consequences worth stating plainly
because they pull in opposite directions:

  - Renaming the repository invalidated every cache, though not one pixel had
    changed. `scripts/worker.py` documents this as a rule its users must obey
    ("THE ONE HARD CONSTRAINT: THE REPO PATH MUST MATCH").
  - Re-encoding an image in place did NOT invalidate its cache, because the
    path was unchanged. That is the dangerous direction: a stale embedding is
    served with no error.

Hashing content fixes both at once, and the fix is the same fix.

WHY FILE BYTES AND NOT DECODED PIXELS. Decoding is the expensive half of
embedding; hashing decoded pixels would cost about what recomputing the
embedding costs, which defeats the purpose. Byte hashing does mean a lossless
re-container (same pixels, new file) reads as a new image and gets re-embedded.
That is the conservative direction -- it wastes a forward pass rather than
serving a wrong vector -- and it does not arise in this project, whose corpora
are all re-encoded to JPEG q95 at pull time, changing bytes and pixels together.

THE MEMO TABLE. Hashing 24 GB on every run would be its own problem, so
(path, mtime, size) -> id is memoised in SQLite. The memo is an optimisation
and never an identity: a mismatch on mtime or size just means re-hash. Paths
are stored RELATIVE to the data root where possible, so relocating the repo or
pointing AIGC_DATA_ROOT elsewhere keeps the memo valid too -- otherwise a
rename would still force a full re-hash pass (minutes of disk, not hours of
GPU, but avoidable).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from aigc_detect.config import DATA_DIR

# 16 bytes. The store may hold a few million rows; at 128 bits the probability
# of any collision across even 10^9 images is on the order of 10^-21. Truncating
# further to save index space would be false economy -- a collision here means
# one image silently wearing another's embedding.
DIGEST_BYTES = 16

# Read in 1 MiB blocks: large enough that syscall overhead vanishes, small
# enough that hashing a directory of 4K images does not balloon RSS.
_CHUNK = 1 << 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_ids (
    path      TEXT PRIMARY KEY,   -- relative to DATA_DIR when possible
    mtime_ns  INTEGER NOT NULL,
    size      INTEGER NOT NULL,
    img_id    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_file_ids_img ON file_ids(img_id);
"""


def content_id(path: str | Path) -> str:
    """blake2b-128 of a file's bytes, as 32 lowercase hex characters."""
    import hashlib

    h = hashlib.blake2b(digest_size=DIGEST_BYTES)
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _key_for(path: Path) -> str:
    """Memo key: relative to the data root when the file lives under it.

    Keeping it relative is what lets the memo survive a repo move or an
    AIGC_DATA_ROOT change, which is the whole point of not keying on absolute
    paths anywhere in this system.
    """
    try:
        return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


class HashCache:
    """Memoised path -> content id, backed by SQLite.

    Not thread-safe; one instance per process. Used as a context manager or
    closed explicitly.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        # WAL so a reader (an eval run) and the writer (an embed run) can
        # coexist instead of one erroring with "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> HashCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def id_for(self, path: str | Path) -> str:
        return self.ids_for([path])[0]

    def ids_for(self, paths: Iterable[str | Path], *, progress: bool = False) -> list[str]:
        """Content ids for many paths, hashing only what the memo cannot answer.

        Returns ids positionally aligned with ``paths``. Raises FileNotFoundError
        naming the first missing file -- a manifest pointing at an absent image
        is a data fault worth surfacing loudly rather than skipping.
        """
        resolved = [Path(p) for p in paths]
        keys = [_key_for(p) for p in resolved]

        known: dict[str, tuple[int, int, str]] = {}
        # SQLite caps host parameters (999 on older builds); chunk the lookup.
        for start in range(0, len(keys), 800):
            batch = keys[start : start + 800]
            placeholders = ",".join("?" * len(batch))
            for k, mt, sz, iid in self._conn.execute(
                f"SELECT path, mtime_ns, size, img_id FROM file_ids WHERE path IN ({placeholders})",
                batch,
            ):
                known[k] = (mt, sz, iid)

        out: list[str] = []
        writes: list[tuple[str, int, int, str]] = []
        iterator: Sequence[int] = range(len(resolved))
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="[hash] content ids", unit="img")

        for i in iterator:
            path, key = resolved[i], keys[i]
            try:
                st = path.stat()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"[hash] manifest references a file that is not on disk: {path}"
                ) from exc
            cached = known.get(key)
            if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                out.append(cached[2])
                continue
            iid = content_id(path)
            out.append(iid)
            writes.append((key, st.st_mtime_ns, st.st_size, iid))

        if writes:
            self._conn.executemany(
                "INSERT INTO file_ids(path, mtime_ns, size, img_id) VALUES(?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET mtime_ns=excluded.mtime_ns, "
                "size=excluded.size, img_id=excluded.img_id",
                writes,
            )
            self._conn.commit()
        return out

    def stats(self) -> dict:
        (n_files,) = self._conn.execute("SELECT COUNT(*) FROM file_ids").fetchone()
        (n_ids,) = self._conn.execute("SELECT COUNT(DISTINCT img_id) FROM file_ids").fetchone()
        return {"paths": n_files, "distinct_ids": n_ids}
