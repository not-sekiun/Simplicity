"""Where a pull's audit verdict lives, and the one place training checks it.

The verdict is written into the corpus's OWN `corpus.yaml`, not a side table
in this package, for the same reason `relocate.py`'s `_write_corpus_record`
put provenance there in the first place: a corpus directory handed to someone
else -- copied, relocated, zipped up -- should still say what it is and
whether it passed the shortcut check, without a second file to keep in sync
or lose. `relocate.py`'s docstring on that function names this module by
name: "Tier 6 extends this with the pull's audit verdict, which is the thing
that turns it from documentation into a gate."

THE GATE ITSELF LIVES IN `corpus.assert_trainable`, NOT HERE. Role and audit
suspicion are the same kind of fact -- "may this corpus enter a training
manifest" -- and `corpus.py`'s own docstring already argues for exactly one
enforcement point instead of two that can drift apart. This module only reads
and writes the verdict; `assert_trainable` is what refuses to resolve.

THE OVERRIDE IS A YAML EDIT, NOT A CLI FLAG. `assert_trainable`'s message for
a wrong `role` already says "change it in corpora.yaml deliberately -- do not
work around it here"; a suspect verdict gets the same treatment. Add
`audit: {override: true, override_reason: "..."}` to the corpus's own
`corpus.yaml` by hand, after actually looking at why the probe fired, and
`is_suspect` stops blocking it. A `--force-suspect` flag on `manifest resolve`
would let a bad shortcut back in with one keystroke and no record of who
decided that was fine or why; a YAML edit under version control is neither
fast nor invisible, which is the point.

A CORPUS WITH NO VERDICT IS NOT SUSPECT. Every corpus already on disk before
this tier existed was relocated without ever being probed, and re-pulling
each one just to generate a verdict is not this tier's job. Absence of an
audit means "never checked", not "checked and safe" -- but it also must not
retroactively block every manifest that already trains on them. `aigc pull
run` writes a verdict at the end of every future pull; until then, silence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from aigc_detect.config import CORPORA_DIR


def verdict_path(corpus_id: str) -> Path:
    return CORPORA_DIR / corpus_id / "corpus.yaml"


def read_verdict(corpus_id: str) -> dict | None:
    """This corpus's last-written `audit:` block, or None if it has never
    been audited (see the module docstring -- that is not the same as safe)."""
    path = verdict_path(corpus_id)
    if not path.is_file():
        return None
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return record.get("audit")


def write_verdict(corpus_id: str, verdict: dict) -> Path:
    """Merge `verdict` into `audit:`, preserving everything else the file
    already carries -- provenance, role, notes. Creates the file if a corpus
    somehow has none yet, so a pull never fails purely because relocate.py's
    writer has not run for this id."""
    path = verdict_path(corpus_id)
    record = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    record = record or {"id": corpus_id}
    record["audit"] = verdict
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def is_suspect(corpus_id: str) -> bool:
    """True only for a verdict that fired AND was never overridden -- see
    the module docstring on why the override lives in the YAML, not here."""
    verdict = read_verdict(corpus_id)
    if not verdict:
        return False
    return bool(verdict.get("suspect")) and not bool(verdict.get("override"))
