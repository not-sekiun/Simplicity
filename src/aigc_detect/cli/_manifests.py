"""Single source of truth for `--manifest`, derived from the recipes on disk.

WHY THIS IS NOT A HARDCODED LIST ANY MORE. Four argparse choices lists used to
carry their own copy of these names and drifted the moment a corpus was added;
tier 2 collapsed them into one dict here, which fixed the drift but kept the
deeper problem: the dict named fifteen `*_MANIFEST` constants from
`config.paths`, so ADDING A MANIFEST MEANT EDITING TWO PYTHON FILES. Tier 5
made a manifest a declarative recipe -- `data/manifests/<name>.yaml`, resolved
by `aigc manifest resolve` -- and this module was the one place still insisting
a recipe also be written down in code before the CLI would accept it. A
first end-to-end run of a brand-new corpus is what surfaced it: every other
step (source registry, pull, audit, recipe, experiment config) took YAML, and
`embed-views --manifest` rejected the new name at the argparse boundary.

`list_recipes()` is now the list. The hyphenated spellings (`train-ext`,
`wildrf-test`, `demo-val`) are kept as ALIASES rather than dropped: every
document, every findings entry and every command in the README uses that form,
and breaking them to save a `replace` call would be a gratuitous cost. Both
spellings resolve to the same CSV.

The `config.paths` constants are untouched and still exported -- other modules
import them directly, and a constant naming one well-known path is fine. What
is not fine is a constant being the ONLY way the CLI can learn a manifest
exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aigc_detect.data.manifest import list_recipes, resolved_path


def _alias(name: str) -> str:
    """`train_ext` -> `train-ext`; the spelling every document already uses."""
    return name.replace("_", "-")


def manifest_names() -> list[str]:
    """Every accepted `--manifest` value: each recipe, plus its hyphen alias.

    Sorted so `--help` reads deterministically. A recipe whose name has no
    underscore contributes one entry, not two.
    """
    names: list[str] = []
    for recipe in list_recipes():
        names.append(recipe)
        if (alias := _alias(recipe)) != recipe:
            names.append(alias)
    return sorted(names)


#: argparse `choices`. Empty when `data/manifests/` is absent (a clone that has
#: never resolved anything), and the registrations below treat empty as
#: "accept any string" rather than "accept nothing" -- an unconstrained value
#: that fails in `_resolve_manifest` with a real hint beats argparse rejecting
#: every possible answer with an empty choices list.
MANIFEST_CHOICES = manifest_names()


def _resolve_manifest(name: str) -> Path:
    """Map a --manifest value to its resolved CSV, exiting with a hint if absent.

    demo-val is embeddable for EVALUATION ONLY (brief 5.4 forbids training on
    it), which the manifest's own `never_train: true` now enforces rather than
    leaving to convention.

    The hint used to be a per-manifest table naming which of seven `make_*.py`
    scripts to run. There is one answer now, because there is one way a manifest
    comes into existence.
    """
    recipe = name.replace("-", "_")
    known = list_recipes()
    if recipe not in known:
        print(f"No recipe '{recipe}' under data/manifests/. Available: {sorted(known)}")
        print("     (`aigc manifest list` shows every recipe; `aigc corpus list` what backs them)")
        sys.exit(1)
    manifest = resolved_path(recipe)
    if not manifest.exists():
        print(f"No {name} manifest at {manifest}.")
        print(f"Run: uv run aigc manifest resolve {recipe}")
        print("     (`aigc manifest list` shows every recipe; `aigc corpus list` what backs them)")
        sys.exit(1)
    return manifest
