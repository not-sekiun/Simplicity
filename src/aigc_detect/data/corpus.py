"""The corpus registry: what image collections exist, and what may be done with them.

A *corpus* is a provenance -- one pull from one upstream source, with its own
directory of images and its own row list. A *manifest* is a selection across
corpora, and lives in :mod:`aigc_detect.data.manifest`. Keeping the two apart is
the point of Tier 5: today `data/` mixes them, with eleven top-level directories
where `wildrf` (a provenance) sits beside `heldout` (an evaluation tier).

WHY DECLARED, NOT DISCOVERED. `scripts/make_splits.py` builds the training split
by globbing `data/raw/*_index.csv`. That glob currently matches
`cifake_index.csv`, whose 120,000 images were deleted in Tier 1 -- so re-running
`split` today would produce a manifest three quarters of which points at nothing,
and no code path would notice until an embed run started failing on missing
files. A glob cannot express "this CSV is a record of a corpus we no longer
have". `registry/corpora.yaml` can, and does, with ``role: dropped``.

ROLE IS ENFORCED. `role: eval` corpora raise if a training manifest tries to
include them. The current safeguard for that is a structural accident -- the
splitter only globs one directory -- which is a fact about code, not a rule, and
it silently stops protecting anything the moment someone globs differently.
See :func:`assert_trainable`.

THE BLIND-PROBE AUDIT IS ENFORCED HERE TOO, NOT SEPARATELY. Tier 6 added
`aigc_detect.data.audit`, which runs a blind probe at the end of every pull
and writes its verdict into the corpus's own `corpus.yaml` (see
`data/audit/gate.py`). "May this corpus enter a training manifest" was already
this function's job for `role`; a corpus whose probe cleared the shortcut
threshold is the same kind of fact as a corpus whose role is `eval`, and a
second enforcement point next to this one would just be a second place for
the two to drift apart. `assert_trainable` is the one gate; the audit adds a
second reason it can fire, not a second function that can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from aigc_detect.config import DATA_DIR
from aigc_detect.log import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "corpora.yaml"

#: Roles a training manifest may draw from. Everything else raises.
TRAINABLE_ROLES = frozenset({"train"})

#: The column contract every corpus index and every manifest satisfies.
COLUMNS = ("image_path", "label", "source", "generator")


@dataclass(frozen=True)
class Corpus:
    """One declared corpus. Paths are resolved against ``$AIGC_DATA_ROOT``."""

    id: str
    role: str
    index: Path | None
    images: Path | None
    provenance: dict = field(default_factory=dict)
    scan: dict | None = None
    notes: str = ""

    @property
    def trainable(self) -> bool:
        return self.role in TRAINABLE_ROLES

    def rows(self) -> pd.DataFrame:
        """This corpus's rows, in its canonical order.

        Order is part of the contract, not an implementation detail: the
        committed manifests were written by concatenating these rows in this
        sequence, and the Tier 5 acceptance test is that a recipe reproduces
        them *row for row*.
        """
        if self.scan is not None:
            return _scan_rows(self)
        if self.index is None:
            raise SystemExit(
                f"[corpus] '{self.id}' has neither an index nor a scan block, so it has no rows. "
                f"It is registered for its images alone (role={self.role})."
            )
        if not self.index.exists():
            raise SystemExit(
                f"[corpus] '{self.id}' index is missing: {self.index}\n"
                f"        Pull it first, or mark the corpus `role: dropped` in corpora.yaml."
            )
        df = pd.read_csv(self.index)
        missing_cols = {"image_path", "label", "source"} - set(df.columns)
        if missing_cols:
            raise SystemExit(f"[corpus] '{self.id}' index lacks required column(s): {sorted(missing_cols)}")
        if "generator" not in df.columns:
            # Two corpora predate the column. A recipe that cares assigns one;
            # leaving it absent here would make the concat ragged.
            df["generator"] = pd.NA
        return df[list(COLUMNS)]


def _scan_rows(corpus: Corpus) -> pd.DataFrame:
    """Enumerate a corpus that ships a directory layout instead of an index.

    Directories are walked in the order the registry lists them and each
    directory's files in sorted order, because that is the order the committed
    manifests carry. Sorting per directory rather than globally also keeps the
    row order stable when a platform's images are added or removed.
    """
    assert corpus.images is not None, f"[corpus] '{corpus.id}' has a scan block but no images root"
    exts = {e.lower() for e in corpus.scan["exts"]}
    rows: list[dict] = []
    for spec in corpus.scan["dirs"]:
        directory = corpus.images / spec["dir"]
        if not directory.is_dir():
            logger.warning("corpus %s: scan directory absent, skipped: %s", corpus.id, directory)
            continue
        rows += [
            {"image_path": str(p.resolve()), "label": spec["label"],
             "source": spec["source"], "generator": spec["generator"]}
            for p in sorted(directory.iterdir())
            if p.suffix.lower() in exts
        ]
    if not rows:
        raise SystemExit(
            f"[corpus] '{corpus.id}' scanned to zero images under {corpus.images}. "
            f"Check the extracted layout."
        )
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _resolve(value, base: Path) -> Path | None:
    return None if value in (None, "null", "") else base / str(value)


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Corpus]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: dict[str, Corpus] = {}
    for cid, spec in raw["corpora"].items():
        out[cid] = Corpus(
            id=cid,
            role=spec["role"],
            index=_resolve(spec.get("index"), DATA_DIR),
            images=_resolve(spec.get("images"), DATA_DIR),
            provenance=spec.get("provenance", {}) or {},
            scan=spec.get("scan"),
            notes=(spec.get("notes") or "").strip(),
        )
    return out


def all_corpora() -> dict[str, Corpus]:
    return dict(_load_registry())


def get_corpus(corpus_id: str) -> Corpus:
    registry = _load_registry()
    if corpus_id not in registry:
        raise SystemExit(
            f"[corpus] unknown corpus '{corpus_id}'. Registered: {sorted(registry)}\n"
            f"        Add it to {REGISTRY_PATH.name} -- corpora are declared, not discovered."
        )
    return registry[corpus_id]


def assert_trainable(corpus_id: str, *, manifest_name: str) -> None:
    """Raise unless this corpus may enter a training manifest.

    The message names the corpus, the role and the manifest, because the whole
    value of the check is that someone reads it and stops rather than reaching
    for an override.
    """
    corpus = get_corpus(corpus_id)
    if corpus.trainable:
        from aigc_detect.data.audit.gate import is_suspect

        if is_suspect(corpus_id):
            raise SystemExit(
                f"[manifest] '{manifest_name}' is a TRAINING manifest and includes corpus "
                f"'{corpus_id}', whose blind-probe audit cleared the shortcut threshold -- "
                f"see its corpus.yaml `audit:` block for the numbers.\n"
                f"        A label shortcut probably survives in this corpus; training on it "
                f"risks repeating the SID_Set / SD3 incidents docs/findings.md records.\n"
                f"        If this verdict is wrong, add `audit: {{override: true, "
                f"override_reason: \"...\"}}` to data/corpora/{corpus_id}/corpus.yaml "
                f"deliberately -- do not work around it here."
            )
        return
    reason = {
        "eval": "it is an evaluation tier -- every published number is measured on it",
        "quarantine": "it was rejected as a corpus; it is kept as evidence, not as data",
        "dropped": "its images are no longer on disk; only the index survives",
    }.get(corpus.role, f"its role is '{corpus.role}'")
    raise SystemExit(
        f"[manifest] '{manifest_name}' is a TRAINING manifest and includes corpus "
        f"'{corpus_id}', which may not be trained on: {reason}.\n"
        f"        If this tier really should become trainable, change its role in "
        f"{REGISTRY_PATH.name} deliberately -- do not work around it here."
    )
