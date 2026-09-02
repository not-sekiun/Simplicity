"""The source registry: how to (re)fetch a corpus this project trains or evaluates on.

`registry/corpora.yaml` says what a corpus IS -- its role, its committed index,
whether it may be trained on (see :mod:`aigc_detect.data.corpus`).
`registry/sources.yaml` says how one gets PULLED: which HF repo and split, what
label the whole stream gets versus which column carries a per-row one, how many
images to keep, and the quality gates that caught two real corpus corruptions
before they reached training -- a recaptioning corpus scored as generator
output (the SD3 rejection, see ``sd3`` below), and a depth-map mirror scored as
photographs (see the ``quality_gate`` block on ``pexels`` in
``registry/sources.yaml``).

WHY A SEPARATE FILE AND NOT A `pull:` BLOCK INSIDE `corpora.yaml`. A corpus can
exist -- and most of the ones below do -- with no working fetcher at all:
`wildfake_dalle_advanced` is a ModelScope page nothing here can reach
programmatically; `wildrf` is a paper's own release, not an API; `cifake` and
`pexels` are `role: dropped` / `deleted` in `corpora.yaml`, meaning their images
are gone from disk but the corpus is still declared so nothing globs a stray
index back in. `corpora.yaml`'s job is describing what is ON DISK right now, in
the shape the training/eval code reads; conflating that with "and here is how
you would get it back" would mean touching the file every pull consumer reads
just to fix a repo id or a quota, and it would make an eval tier's row count
depend on whether its fetch recipe happens to parse.

ONE ID SPACE. A source's key here is always the corpus id it fills in
`corpora.yaml` -- ``get_source("nano_banana")`` and ``get_corpus("nano_banana")``
name the same thing, deliberately, so a fetcher's output has exactly one place
to land: ``data/corpora/<id>/``.

WHAT IS DELIBERATELY NOT HERE. `sd3` is `quarantine`-role in `corpora.yaml`
with `provenance: {repo: unrecorded}` -- its own registration record says its
origin was never written down, because it was rejected before anyone thought to
record where it came from (see `docs/` / `data/quarantine/README.md`). Writing
a source entry that names a repo would mean inventing one; not registering it
records the same fact the corpus entry already does, honestly, instead of
papering over it. ``UNREGISTERED`` below is the list of such deliberate gaps,
and ``tests/test_fetchers.py`` asserts against it rather than against "zero
gaps", so a real gap (a corpus added to `corpora.yaml` and simply forgotten
here) still fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "sources.yaml"

#: corpora.yaml ids whose `provenance.fetcher` is not "manual" but which are
#: deliberately unregistered here -- see the module docstring's last section.
UNREGISTERED = frozenset({"sd3"})


@dataclass(frozen=True)
class Source:
    """One pullable source. Everything fetcher-specific lives in ``config``,
    which is intentionally untyped: a kagglehub pull's ``handle`` and an
    HF streaming pull's ``per_generator`` quota have nothing in common, and a
    dataclass field per fetcher kind would just be a wider way to spell the
    same YAML dict.
    """

    id: str
    fetcher: str
    license: str
    config: dict = field(default_factory=dict)
    notes: str = ""


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Source]:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: dict[str, Source] = {}
    for sid, spec in raw["sources"].items():
        spec = dict(spec)
        try:
            fetcher = spec.pop("fetcher")
        except KeyError:
            raise SystemExit(f"[sources] '{sid}' in {REGISTRY_PATH.name} has no `fetcher:` key") from None
        out[sid] = Source(
            id=sid,
            fetcher=fetcher,
            license=spec.pop("license", "unverified"),
            notes=(spec.pop("notes", "") or "").strip(),
            config=spec,  # whatever remains is fetcher-specific
        )
    return out


def all_sources() -> dict[str, Source]:
    return dict(_load_registry())


def get_source(source_id: str) -> Source:
    registry = _load_registry()
    if source_id not in registry:
        raise SystemExit(
            f"[sources] unknown source '{source_id}'. Registered: {sorted(registry)}\n"
            f"        Run `aigc pull list`, or add it to {REGISTRY_PATH.name} -- sources are "
            f"declared, not discovered."
        )
    return registry[source_id]
