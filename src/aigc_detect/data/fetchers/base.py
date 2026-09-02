"""The `Fetcher` protocol, resumable pull state, and the incremental index writer.

Every backend under this package (`hf.py`, `kaggle.py`, `http.py`, `manual.py`)
answers to the same contract:

    class Fetcher(Protocol):
        def pull(self, source: Source, dest: CorpusPaths, state: PullState) -> PullResult: ...

and every one of them is built on :class:`IncrementalIndexWriter`, which is
where the actual durability guarantee lives.

WHY A STATE FILE PER CORPUS. `download_ood_benchmark.py` streams for up to an
hour and used to write `index.csv` only after the loop ended -- so a kill at
minute 50 discarded every image already saved, and it grew a `--reindex-only`
flag whose whole job was recovering from that after the fact (see that
script's `reindex_from_disk` docstring, and the label-recovery bug it goes on
to describe: a rebuilt index cannot always recover the label from the
directory name alone). `.pull_state.json` is that recovery made unnecessary in
the first place: every fetcher commits its progress as it goes, so a killed
pull resumes from the last committed batch instead of needing a forensic pass
over the directory tree afterward.

DURABILITY ORDER IS LOAD-BEARING, the same way it is in the embedding store
(`aigc_detect.cache.store`): an image's bytes are written to
`images/` by the caller BEFORE the row reaches :meth:`IncrementalIndexWriter.add`,
and `index.csv` is appended and fsync'd BEFORE `.pull_state.json` is
overwritten. Crash between "bytes on disk" and "row in the index" leaves an
orphan image file -- wasted space, harmless, and indistinguishable from what
`aigc corpus orphans` already reports for other reasons. Crash between "index
written" and "state committed" leaves the index slightly ahead of the state
file's `rows_written` count, which `--verify` reconciles by recounting the
index rather than trusting the counter. The one order never chosen is
committing the state file first: that would let a resumed pull believe rows
exist that the index does not carry, and silently produce a corpus with holes.

CONFIG_HASH FORCES A CLEAN RESTART. `.pull_state.json` fingerprints the
`Source.config` that produced it. If someone edits `sources.yaml` -- raises a
cap, changes a repo -- and then runs `aigc pull <id>` (resume is the default),
resuming against the OLD partial index under the NEW config would silently
interleave two different pull configurations under one `source` label, the
same failure `download_aigc_modern.py` and `download_real_domains.py` guard
against with their own "already holds N images -- pass --overwrite" checks
(now generalized here instead of re-implemented per script). A hash mismatch
therefore refuses to resume and asks for `--force` instead of guessing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from aigc_detect.config import CORPORA_DIR
from aigc_detect.data.corpus import COLUMNS as INDEX_COLUMNS
from aigc_detect.data.dataset import resolve_image_path

STATE_FILENAME = ".pull_state.json"

#: Rows queued before a checkpoint is forced even if the caller never asks --
#: keeps a slow backend (manual's directory walk) from holding an unbounded
#: pending list in memory on a huge corpus.
DEFAULT_BATCH_SIZE = 200


def config_hash(config: dict) -> str:
    """A short, stable fingerprint of a source's pull config.

    Canonical JSON (sorted keys) rather than `hash()` or `repr()`: both of
    those vary across processes or Python versions for the same logical dict,
    which would make `.pull_state.json` written by one run unreadable by the
    next for no reason connected to the config actually changing.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class PullState:
    """`.pull_state.json`'s contents: what has been committed, and where to
    resume from. `cursor` is deliberately untyped -- a byte offset means
    something different to `http.py` than a stream position means to
    `hf.py`, and only the fetcher that wrote it needs to read it back.
    """

    source_id: str
    config_hash: str
    rows_scanned: int = 0
    rows_written: int = 0
    cursor: dict = field(default_factory=dict)
    completed: bool = False

    @classmethod
    def fresh(cls, source_id: str, config: dict) -> PullState:
        return cls(source_id=source_id, config_hash=config_hash(config))

    @classmethod
    def load(cls, path: Path) -> PullState | None:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-replace: an atomic rename on both POSIX and NTFS, so a
        # crash mid-write leaves the PREVIOUS state file intact rather than a
        # truncated, unparseable one.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(path)


@dataclass
class PullResult:
    rows_written: int
    rows_scanned: int
    resumed: bool
    completed: bool
    note: str = ""


@dataclass(frozen=True)
class CorpusPaths:
    """Where a pull writes: ``data/corpora/<id>/{images/, index.csv, .pull_state.json}``.

    The same three-file shape :mod:`aigc_detect.data.relocate` moved every
    corpus into -- a fetcher and a relocation both converge on one layout,
    which is what lets a corpus be re-pulled, moved, or hand-verified
    interchangeably.
    """

    root: Path

    @classmethod
    def for_source(cls, source_id: str) -> CorpusPaths:
        return cls(root=CORPORA_DIR / source_id)

    @property
    def images(self) -> Path:
        return self.root / "images"

    @property
    def index_csv(self) -> Path:
        return self.root / "index.csv"

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILENAME


class Fetcher(Protocol):
    def pull(self, source, dest: CorpusPaths, state: PullState) -> PullResult: ...


def wipe(dest: CorpusPaths) -> None:
    """Discard a partial (or complete) pull: images, index, and state, gone.

    Used by `--force` and by a config_hash mismatch. Never called on a
    resume -- see the module docstring for why a mismatch does not silently
    keep going instead.
    """
    if dest.images.is_dir():
        shutil.rmtree(dest.images)
    dest.index_csv.unlink(missing_ok=True)
    dest.state_path.unlink(missing_ok=True)


def open_state(source, dest: CorpusPaths, *, force: bool) -> tuple[PullState, bool]:
    """Load (or start) this source's pull state, honoring `--force` and the
    config-hash check. Returns ``(state, resumed)``.
    """
    existing = None if force else PullState.load(dest.state_path)
    new_hash = config_hash(source.config)

    if force:
        wipe(dest)
        return PullState.fresh(source.id, source.config), False

    if existing is None:
        return PullState.fresh(source.id, source.config), False

    if existing.config_hash != new_hash:
        raise SystemExit(
            f"[pull] '{source.id}': the pull config has changed since the partial pull at "
            f"{dest.root} was started ({existing.rows_written:,} rows committed under the old "
            f"config).\n"
            f"       Resuming would silently mix two different configurations under one corpus. "
            f"Pass --force to discard the partial pull and restart clean."
        )

    if existing.completed:
        return existing, True

    return existing, True


class IncrementalIndexWriter:
    """Appends index rows and commits `.pull_state.json` together, every batch.

    THIS is the piece that makes the resumability guarantee real rather than
    aspirational: a fetcher backend calls :meth:`add` for every row it has
    already written image bytes for, and :meth:`checkpoint` (directly, or via
    :meth:`maybe_checkpoint` every `batch_size` rows) flushes the queued rows
    to `index.csv` -- fsync'd -- and only then overwrites `.pull_state.json`.
    A process killed between two checkpoints loses at most one batch's worth
    of already-fetched images as orphans on disk; it never loses a row that
    was already reported committed, and it never leaves the state file ahead
    of the index it describes.
    """

    def __init__(self, dest: CorpusPaths, state: PullState, *, batch_size: int = DEFAULT_BATCH_SIZE):
        self.dest = dest
        self.state = state
        self.batch_size = batch_size
        self._pending: list[dict] = []
        dest.images.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict) -> None:
        """Queue one row. The caller must have already written its image
        bytes under `dest.images` -- this only ever records that a row
        exists, never fetches anything itself.
        """
        self._pending.append(row)

    def maybe_checkpoint(self, *, rows_scanned: int, cursor: dict) -> bool:
        """Checkpoint if a full batch is queued. Returns whether it did."""
        if len(self._pending) < self.batch_size:
            return False
        self.checkpoint(rows_scanned=rows_scanned, cursor=cursor)
        return True

    def checkpoint(self, *, rows_scanned: int, cursor: dict) -> None:
        if self._pending:
            new_file = not self.dest.index_csv.exists()
            with open(self.dest.index_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
                if new_file:
                    w.writeheader()
                for row in self._pending:
                    w.writerow({col: row.get(col, "") for col in INDEX_COLUMNS})
                f.flush()
                os.fsync(f.fileno())
            self.state.rows_written += len(self._pending)
            self._pending.clear()
        self.state.rows_scanned = rows_scanned
        self.state.cursor = cursor
        self.state.save(self.dest.state_path)

    def finish(self, *, rows_scanned: int, cursor: dict, completed: bool) -> None:
        self.checkpoint(rows_scanned=rows_scanned, cursor=cursor)
        self.state.completed = completed
        self.state.save(self.dest.state_path)


def existing_index_paths(dest: CorpusPaths) -> set[str]:
    """`image_path` values already committed -- for a backend whose resume
    strategy is "diff what's on disk against what's indexed" rather than a
    linear cursor (kagglehub, manual).
    """
    if not dest.index_csv.is_file():
        return set()
    with open(dest.index_csv, newline="", encoding="utf-8") as f:
        return {row["image_path"] for row in csv.DictReader(f)}


def verify_and_repair(dest: CorpusPaths) -> dict:
    """Reconcile `index.csv` against what's actually on disk. Reports and
    repairs, in the same spirit as `aigc corpus orphans` (report) and
    `aigc cache compact` (repair bytes, never rows): an index row whose file
    is gone is dropped (the file is gone; keeping the row would just make a
    later manifest resolve fail against nothing), and `.pull_state.json`'s
    counters are reconciled to match the index actually on disk rather than
    trusted blindly -- a checkpoint crash can leave the counter and the file
    disagreeing by at most one batch, per the module docstring.
    """
    if not dest.index_csv.is_file():
        return {"rows": 0, "dropped_missing": 0, "orphan_files": 0}

    with open(dest.index_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept, dropped = [], 0
    referenced: set[Path] = set()
    for row in rows:
        p = resolve_image_path(row["image_path"])
        if p.is_file():
            kept.append(row)
            referenced.add(p.resolve())
        else:
            dropped += 1

    if dropped:
        with open(dest.index_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
            w.writeheader()
            for row in kept:
                w.writerow({col: row.get(col, "") for col in INDEX_COLUMNS})

    orphans = 0
    if dest.images.is_dir():
        orphans = sum(1 for p in dest.images.rglob("*") if p.is_file() and p.resolve() not in referenced)

    state = PullState.load(dest.state_path)
    if state is not None and state.rows_written != len(kept):
        state.rows_written = len(kept)
        state.save(dest.state_path)

    return {"rows": len(kept), "dropped_missing": dropped, "orphan_files": orphans}
