"""Manifests as recipes: a declarative selection over corpora, not a hand-built CSV.

Ten CSVs exist in this project and seven Python scripts exist to write them.
Every one of those scripts does the same four things -- concatenate some corpus
indexes, filter, occasionally split, write -- differing only in which corpora and
which filter. `make_ood.py` and `make_heldout.py` are the same 60 lines twice.

They also exist in that number for a reason that is now obsolete. Each script's
docstring says some version of *"a new manifest under a new name means a new
cache stem, so nothing already computed goes stale"*: rebuilding `train.csv`
would have invalidated every embedding keyed on its path fingerprint, so the
data design was bent around the cache design. Tier 4 removed that coupling --
an embedding is keyed on image content now -- so manifests are free to be
rebuilt, composed, and renamed, and this module is what they become.

WHAT A RECIPE IS

    include:  [tiny_genimage]              # corpora, or manifest:<name>
    filter:   {label: 0}                   # optional
    assign:   {source: sid_set_real}       # optional
    split:    {fraction: 0.15, seed: 42, take: train}

Resolution is: concatenate `include` in the order written -> filter -> assign ->
split -> drop duplicate `image_path`, keeping the first. Every step is order
preserving, because the acceptance test for this tier is that each recipe
reproduces its committed CSV *row for row*, and a set comparison would hide a
reordering that changes which rows a `--limit` prefix selects.

WHY `never_train` IS A FLAG AND NOT A COMMENT

Today nothing enforces "don't train on the eval tiers". The safeguard is that
`make_splits.py` happens to glob only `data/raw/`, which is a fact about one
script rather than a rule about the data -- it protects nothing the moment
someone globs differently, and the brief (5.4) is explicit that `demo_val` must
never be trained on. Here, a manifest declares `never_train: true`, a recipe
that `include:`s an eval-role corpus raises, and a trainer that loads a
never-train manifest raises. Three checks, one declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from aigc_detect.config import DATA_DIR
from aigc_detect.data.corpus import COLUMNS, assert_trainable, get_corpus
from aigc_detect.data.dataset import resolve_image_path
from aigc_detect.log import get_logger

logger = get_logger(__name__)

MANIFESTS_DIR = DATA_DIR / "manifests"
RESOLVED_DIR = MANIFESTS_DIR / "resolved"

_MANIFEST_PREFIX = "manifest:"


@dataclass(frozen=True)
class Recipe:
    name: str
    spec: dict

    @property
    def never_train(self) -> bool:
        return bool(self.spec.get("never_train", False))

    @property
    def includes(self) -> list[str]:
        inc = self.spec.get("include")
        if not inc:
            raise SystemExit(f"[manifest] recipe '{self.name}' has no `include:` -- nothing to resolve")
        return list(inc)


def recipe_path(name: str) -> Path:
    return MANIFESTS_DIR / f"{name}.yaml"


def resolved_path(name: str) -> Path:
    return RESOLVED_DIR / f"{name}.csv"


def load_recipe(name: str) -> Recipe:
    path = recipe_path(name)
    if not path.exists():
        available = sorted(p.stem for p in MANIFESTS_DIR.glob("*.yaml")) if MANIFESTS_DIR.is_dir() else []
        raise SystemExit(f"[manifest] no recipe '{name}' at {path}. Available: {available}")
    return Recipe(name=name, spec=yaml.safe_load(path.read_text(encoding="utf-8")))


def list_recipes() -> list[str]:
    return sorted(p.stem for p in MANIFESTS_DIR.glob("*.yaml")) if MANIFESTS_DIR.is_dir() else []


# -- the four operations ------------------------------------------------------


def _gather(recipe: Recipe, _seen: frozenset[str]) -> pd.DataFrame:
    """Concatenate every `include` in the order written.

    An include may name another manifest, which is how `train_ext` says "train,
    plus this slice" without restating the split that produced train.
    """
    frames: list[pd.DataFrame] = []
    for ref in recipe.includes:
        if ref.startswith(_MANIFEST_PREFIX):
            other = ref[len(_MANIFEST_PREFIX):]
            if other in _seen:
                raise SystemExit(
                    f"[manifest] recipe cycle: {' -> '.join([*_seen, other])}. "
                    f"A manifest cannot include itself, transitively or otherwise."
                )
            frames.append(resolve(other, _seen=_seen | {recipe.name}))
            continue
        if not recipe.never_train:
            assert_trainable(ref, manifest_name=recipe.name)
        frames.append(get_corpus(ref).rows())
    return pd.concat(frames, ignore_index=True)


def _filter(df: pd.DataFrame, spec: dict, name: str) -> pd.DataFrame:
    """Row selection. Every clause is an intersection; order does not matter."""
    f = spec.get("filter")
    if not f:
        return df
    before = len(df)
    if "label" in f:
        df = df[df["label"] == int(f["label"])]
    if "sources" in f:
        df = df[df["source"].isin(list(f["sources"]))]
    if "generators" in f:
        df = df[df["generator"].isin(list(f["generators"]))]
    if "exclude_generators" in f:
        df = df[~df["generator"].isin(list(f["exclude_generators"]))]
    if "exclude_sources" in f:
        df = df[~df["source"].isin(list(f["exclude_sources"]))]
    if f.get("require_on_disk"):
        # Only for corpora whose index can outlive its files. It is a real
        # check, not a formality: sid_set's builder already carried it.
        present = df["image_path"].map(lambda p: resolve_image_path(p).is_file())
        gone = int((~present).sum())
        if gone:
            logger.warning("%s: %d indexed image(s) are not on disk -- dropped", name, gone)
        df = df[present]
    logger.debug("%s: filter kept %d of %d rows", name, len(df), before)
    return df.reset_index(drop=True)


def _assign(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Overwrite `source`/`generator` for corpora whose index predates them.

    `sid_set`'s index has no generator column and one undifferentiated source;
    the manifest that keeps its reals calls them what they are.
    """
    a = spec.get("assign")
    if not a:
        return df
    df = df.copy()
    for column, value in a.items():
        if column not in COLUMNS:
            raise SystemExit(f"[manifest] `assign` may only set {list(COLUMNS)}, not '{column}'")
        df[column] = value
    return df


def _split(df: pd.DataFrame, spec: dict, name: str) -> pd.DataFrame:
    """The seeded stratified split, or the whole frame when there is none.

    Stratification is on (source, label) and deliberately NOT on generator:
    spreading every generator across both halves would defeat held-out-generator
    evaluation, which is most of what `val` is for.
    """
    s = spec.get("split")
    if not s:
        return df
    from sklearn.model_selection import train_test_split

    take = s["take"]
    if take not in ("train", "val"):
        raise SystemExit(f"[manifest] '{name}': split.take must be 'train' or 'val', not {take!r}")

    strata = df["source"].astype(str)
    for column in s.get("by", ["source", "label"]):
        if column != "source":
            strata = strata + "_" + df[column].astype(str)

    # A stratum with a single member cannot be split; it goes entirely to train
    # rather than aborting the build.
    counts = strata.value_counts()
    tiny = counts[counts < 2].index
    if len(tiny):
        logger.warning("%s: strata with <2 rows go entirely to train: %s", name, list(tiny))
    ok, singletons = df[~strata.isin(tiny)], df[strata.isin(tiny)]

    train_df, val_df = train_test_split(
        ok, test_size=float(s["fraction"]), random_state=int(s["seed"]),
        stratify=strata[~strata.isin(tiny)],
    )
    if take == "val":
        return val_df.reset_index(drop=True)
    return pd.concat([train_df, singletons], ignore_index=True)


# -- resolution ---------------------------------------------------------------


def resolve(name: str, *, _seen: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Materialize a recipe into rows. Pure: reads indexes, writes nothing."""
    recipe = load_recipe(name)
    df = _gather(recipe, _seen)
    df = _filter(df, recipe.spec, name)
    df = _assign(df, recipe.spec)
    df = _split(df, recipe.spec, name)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return df[list(COLUMNS)]


def assert_trainable_manifest(name: str) -> None:
    """Raise if a trainer is about to load a manifest marked never_train."""
    recipe = load_recipe(name)
    if recipe.never_train:
        raise SystemExit(
            f"[manifest] '{name}' is marked `never_train: true` and cannot be used for training. "
            f"It is an evaluation tier; training on it would invalidate every number measured "
            f"against it."
        )


def write_resolved(name: str) -> Path:
    """Resolve and commit the materialized CSV.

    The resolved file is what tooling reads and what review sees in a diff; the
    recipe is what says why those rows and not others. Both are committed on
    purpose -- a recipe alone would make every historical number depend on the
    exact scikit-learn build that split it.
    """
    df = resolve(name)
    out = resolved_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("%s: %d rows -> %s", name, len(df), out)
    return out
