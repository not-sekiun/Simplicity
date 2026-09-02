"""Move every corpus into one hierarchy, and make manifests portable.

Today `data/` has eleven top-level directories that mix PROVENANCE (`wildrf`,
`aigc_ext`, `raw`) with EVALUATION TIER (`heldout`, `ood`, `demo_val`), so where
an image lives tells you either where it came from or what it is used for,
depending on which directory you happen to be in. Afterwards there is one rule:

    data/corpora/<corpus_id>/
      images/       the files
      index.csv     image_path relative to $AIGC_DATA_ROOT, POSIX separators
      corpus.yaml   what this is, where it came from, when it was pulled

WHY THIS IS SAFE TO DO NOW AND WAS NOT BEFORE. Every cached embedding used to be
keyed on a hash of absolute image paths, so moving a single corpus invalidated
gigabytes of GPU-hours. Tier 4 keyed them on image CONTENT instead. Moving a
file now changes nothing about its embedding's identity: the hash memo misses on
the new path, re-reads the bytes, computes the same id, and the store hits. That
costs minutes of disk and no GPU at all -- and demonstrating exactly that is the
acceptance test for this tier.

RELATIVE PATHS, AND WHICH ROOT. Indexes are rewritten relative to
``$AIGC_DATA_ROOT`` with forward slashes, so a committed manifest stops being a
machine-specific artifact. It is the same root the hash memo keys on, so a
manifest and its cached vectors relocate together or not at all.

ORDER WITHIN A CORPUS IS PRESERVED. A rewritten index keeps its rows in their
original order, because thirteen manifests are defined as selections over these
rows and the tier's acceptance test compares them row for row.

MOVES, NEVER COPIES -- except COCO, which is not ours to move: those 5,000
images live in `~/.cache/kagglehub`, which `kagglehub` may evict at any time,
and a committed manifest pointing into a transient cache is the defect being
fixed. They are copied in.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from aigc_detect.config import DATA_DIR
from aigc_detect.data.corpus import Corpus, all_corpora
from aigc_detect.data.dataset import resolve_image_path
from aigc_detect.log import get_logger

logger = get_logger(__name__)

CORPORA_DIR = DATA_DIR / "corpora"

#: Corpora whose images are not ours to move -- they live in someone else's
#: cache directory, so they are copied in rather than relocated.
COPY_IN = frozenset({"coco_val2017"})

#: Roles that keep their own top-level directory rather than joining `corpora/`.
#: A rejected corpus is evidence, and evidence filed next to the training data
#: is evidence waiting to be trained on by accident.
KEEP_IN_PLACE = frozenset({"quarantine"})


@dataclass
class Move:
    corpus_id: str
    kind: str  # "move" | "copy" | "index-only" | "skip"
    src: Path | None
    dst: Path | None
    index_src: Path | None
    index_dst: Path | None
    rows: int
    bytes: int
    note: str = ""


def target_dir(corpus_id: str) -> Path:
    return CORPORA_DIR / corpus_id


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _rel_posix(path: Path) -> str:
    """A path as an index stores it: relative to the data root, forward slashes."""
    return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()


def plan(corpora: dict[str, Corpus] | None = None) -> list[Move]:
    """What relocation would do. Touches nothing."""
    moves: list[Move] = []
    for cid, corpus in sorted((corpora or all_corpora()).items()):
        dst_root = target_dir(cid)
        images_dst = dst_root / "images"
        index_dst = dst_root / "index.csv"

        if corpus.role in KEEP_IN_PLACE:
            moves.append(Move(cid, "skip", corpus.images, None, corpus.index, None, 0, 0,
                              f"role={corpus.role}: stays outside corpora/"))
            continue

        # Already relocated: the registry points at the target. Idempotent.
        if corpus.images is not None and corpus.images.resolve() == images_dst.resolve():
            moves.append(Move(cid, "skip", corpus.images, images_dst, corpus.index, index_dst,
                              0, 0, "already in place"))
            continue

        rows = 0
        if corpus.index is not None and corpus.index.exists():
            rows = len(pd.read_csv(corpus.index, usecols=["image_path"]))
        elif corpus.scan is not None and corpus.images is not None and corpus.images.is_dir():
            rows = len(corpus.rows())

        if corpus.images is None and cid in COPY_IN and corpus.index is not None:
            # No images root of its own because the images are in someone
            # else's cache. Ingest exactly the rows the index names -- not the
            # whole cache directory, which holds more than this corpus.
            files = _external_files(corpus)
            moves.append(Move(cid, "ingest", None, images_dst, corpus.index, index_dst, rows,
                              sum(f.stat().st_size for f in files),
                              f"{len(files):,} files copied out of a transient cache"))
            continue

        if corpus.images is None or not corpus.images.is_dir():
            # No images to move; the index is still worth filing under the
            # corpus it describes, so `data/raw` stops being a junk drawer.
            kind = "index-only" if corpus.index and corpus.index.exists() else "skip"
            note = "images already gone; index preserved as a record" if kind == "index-only" else "nothing on disk"
            moves.append(Move(cid, kind, None, None, corpus.index, index_dst, rows, 0, note))
            continue

        kind = "copy" if cid in COPY_IN else "move"
        moves.append(Move(cid, kind, corpus.images, images_dst, corpus.index, index_dst,
                          rows, _dir_bytes(corpus.images),
                          "copied out of a transient cache" if kind == "copy" else ""))
    return moves


def _external_files(corpus: Corpus) -> list[Path]:
    """The files an externally-cached corpus's index names, checked to exist.

    Flattened into one directory on ingest, so their basenames must be unique.
    COCO's are; asserting it means a future corpus that is not gets an error
    instead of quietly losing rows to overwrites.
    """
    df = pd.read_csv(corpus.index, usecols=["image_path"])
    files = [resolve_image_path(p) for p in df["image_path"]]
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise SystemExit(
            f"[relocate] {corpus.id}: {len(missing):,} of {len(files):,} indexed images are not "
            f"on disk, so they cannot be ingested.\n"
            f"        first missing: {missing[0]}\n"
            f"        This corpus lives in a cache that may have been evicted; re-pull it first."
        )
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise SystemExit(
            f"[relocate] {corpus.id}: index has duplicate basenames, so flattening them into one "
            f"images/ directory would lose rows."
        )
    return files


def _rewrite_index(corpus: Corpus, move: Move) -> pd.DataFrame:
    """Map an index's absolute paths onto the new layout, relative and POSIX.

    Every row must fall under the corpus's old images root. A row that does not
    is a fault worth stopping on: it means the registry's `images` is wrong, and
    silently keeping that row would leave one manifest pointing at a path this
    move is about to invalidate.
    """
    df = pd.read_csv(corpus.index)
    if corpus.images is None:
        return df

    old_root = corpus.images.resolve()
    new_root = move.dst if move.dst is not None else target_dir(corpus.id) / "images"

    out: list[str] = []
    strays: list[str] = []
    for raw in df["image_path"].astype(str):
        p = resolve_image_path(raw)
        try:
            rel = p.resolve().relative_to(old_root)
        except ValueError:
            strays.append(raw)
            continue
        out.append((new_root / rel).resolve().relative_to(DATA_DIR.resolve()).as_posix())

    if strays:
        raise SystemExit(
            f"[relocate] {corpus.id}: {len(strays)} index row(s) point outside the corpus's "
            f"images root {old_root}. The registry is wrong, or the index mixes corpora.\n"
            f"        first: {strays[0]}"
        )
    df["image_path"] = out
    return df


def _write_corpus_record(corpus: Corpus, rows: int) -> Path:
    """The per-corpus `corpus.yaml`: what this is and where it came from.

    Provenance travels WITH the images rather than only in the code registry,
    so a corpus directory handed to someone else still says what it is. Tier 6
    extends this with the pull's audit verdict, which is the thing that turns it
    from documentation into a gate.
    """
    record = {
        "id": corpus.id,
        "role": corpus.role,
        "rows": rows,
        "provenance": corpus.provenance,
        "relocated": date.today().isoformat(),
    }
    if corpus.notes:
        record["notes"] = corpus.notes
    out = target_dir(corpus.id) / "corpus.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def apply(moves: list[Move]) -> dict:
    """Execute the plan, one corpus at a time, index written straight after.

    Deliberately not parallel and deliberately not batched: a partial relocation
    should leave every corpus either fully moved with a correct index or
    untouched, never a directory in one place and an index describing another.
    """
    corpora = all_corpora()
    moved = copied = indexed = 0
    moved_bytes = 0

    for m in moves:
        if m.kind == "skip":
            continue
        corpus = corpora[m.corpus_id]
        dst_root = target_dir(m.corpus_id)
        dst_root.mkdir(parents=True, exist_ok=True)

        if m.kind == "ingest":
            assert m.dst is not None
            if m.dst.exists():
                raise SystemExit(f"[relocate] {m.corpus_id}: destination already exists: {m.dst}")
            files = _external_files(corpus)
            m.dst.mkdir(parents=True)
            logger.info("%s: ingesting %d files -> %s", m.corpus_id, len(files), m.dst)
            for f in files:
                shutil.copy2(f, m.dst / f.name)
            copied += 1
            moved_bytes += m.bytes
            df = pd.read_csv(corpus.index)
            df["image_path"] = [_rel_posix(m.dst / Path(str(p)).name) for p in df["image_path"]]
            m.index_dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(m.index_dst, index=False)
            indexed += 1
            _write_corpus_record(corpus, m.rows)
            continue

        # The index is rewritten from the OLD paths, so compute it before moving.
        new_index = None
        if corpus.index is not None and corpus.index.exists():
            new_index = _rewrite_index(corpus, m)

        if m.kind in ("move", "copy"):
            assert m.src is not None and m.dst is not None
            if m.dst.exists():
                raise SystemExit(
                    f"[relocate] {m.corpus_id}: destination already exists: {m.dst}\n"
                    f"        Refusing to merge two corpora into one directory."
                )
            logger.info("%s: %s %s -> %s", m.corpus_id, m.kind, m.src, m.dst)
            if m.kind == "move":
                shutil.move(str(m.src), str(m.dst))
                moved += 1
            else:
                shutil.copytree(str(m.src), str(m.dst))
                copied += 1
            moved_bytes += m.bytes

        if new_index is not None:
            m.index_dst.parent.mkdir(parents=True, exist_ok=True)
            new_index.to_csv(m.index_dst, index=False)
            indexed += 1
        elif corpus.scan is not None:
            # Scan-based corpora have no upstream index; materialize one now
            # that the tree has a stable home, so the corpus stops being
            # defined by a directory walk at read time.
            rows = _scan_after_move(corpus, m)
            m.index_dst.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(m.index_dst, index=False)
            indexed += 1

        _write_corpus_record(corpus, m.rows)

    return {"moved": moved, "copied": copied, "indexed": indexed, "bytes": moved_bytes}


def _scan_after_move(corpus: Corpus, move: Move) -> pd.DataFrame:
    """Re-run a scan-based corpus's enumeration against its new location."""
    from aigc_detect.data.corpus import Corpus as _C
    from aigc_detect.data.corpus import _scan_rows

    relocated = _C(id=corpus.id, role=corpus.role, index=None, images=move.dst,
                   provenance=corpus.provenance, scan=corpus.scan, notes=corpus.notes)
    df = _scan_rows(relocated)
    df["image_path"] = [_rel_posix(Path(p)) for p in df["image_path"].astype(str)]
    return df


def report(moves: list[Move]) -> None:
    print(f"{'corpus':<26} {'action':<11} {'rows':>8} {'GB':>7}  destination")
    print("-" * 96)
    total_bytes = 0
    for m in moves:
        dest = str(m.dst.relative_to(DATA_DIR)) if m.dst else (
            str(m.index_dst.relative_to(DATA_DIR)) if m.index_dst else "-")
        note = f"  ({m.note})" if m.note else ""
        print(f"{m.corpus_id:<26} {m.kind:<11} {m.rows:>8,} {m.bytes / 1e9:>7.2f}  {dest}{note}")
        total_bytes += m.bytes
    print("-" * 96)
    actionable = [m for m in moves if m.kind != "skip"]
    print(f"{len(actionable)} corpora to relocate, {total_bytes / 1e9:.2f} GB")
