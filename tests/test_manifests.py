"""Manifest recipes resolve to exactly the rows the project already publishes.

This is Tier 5's acceptance test, and it is the one that makes the rest of the
tier safe: seven `make_*.py` scripts are being replaced by thirteen YAML files,
and the only way that is not a rewrite of the dataset is if every recipe lands
on the same rows, in the same order, as the CSV it replaces.

ORDER, NOT JUST MEMBERSHIP. A set comparison would pass on a reordering, and
row order is load-bearing here: `--limit N` takes a prefix, `fingerprint_paths`
hashes the sequence, and a reordered manifest would quietly become a different
eval set carrying the same name.

These tests read committed CSVs and corpus indexes. They do not open a single
image, so they run anywhere; the tests that need pixels belong to the embedder.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aigc_detect.config import DATA_DIR
from aigc_detect.data.corpus import all_corpora, get_corpus
from aigc_detect.data.manifest import (
    list_recipes,
    load_recipe,
    resolve,
    resolved_path,
)

#: Where each manifest lived before Tier 5 gathered them. Kept as the migration
#: proof: as long as these files exist, a recipe must still reproduce them.
LEGACY = {
    "train": "processed/train.csv",
    "val": "processed/val.csv",
    "train_ext": "processed/train_ext.csv",
    "heldout": "heldout/heldout.csv",
    "ood": "ood/ood.csv",
    "demo_val": "demo_val/demo_val.csv",
    "wildrf_test": "wildrf/wildrf_test.csv",
    "wildrf_real": "processed/wildrf_real.csv",
    "sid_real": "processed/sid_real.csv",
    "unsplash_real": "processed/unsplash_real.csv",
    "nano_banana": "processed/nano_banana.csv",
    "midjourney_v6": "processed/midjourney_v6.csv",
    "dalle3_holdout": "processed/dalle3_holdout.csv",
}

RECIPES = list_recipes()


def _frames_match(got: pd.DataFrame, ref: pd.DataFrame) -> bool:
    """Compare on the columns the reference actually carries.

    `demo_val` predates the `generator` column, so a strict column-set equality
    would fail it for a reason that has nothing to do with row selection.
    Everything is cast to str because a column read back from CSV can differ in
    dtype (int64 vs Arrow-backed) without differing in value.
    """
    cols = [c for c in ("image_path", "label", "source", "generator") if c in ref.columns]
    return (
        got[cols].reset_index(drop=True).astype(str)
        .equals(ref[cols].reset_index(drop=True).astype(str))
    )


def test_there_are_recipes_to_test():
    assert RECIPES, "no recipes found under data/manifests -- the registry is not wired up"


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_reproduces_its_resolved_csv(name):
    """The durable regression test: resolution is deterministic run to run."""
    committed = resolved_path(name)
    if not committed.exists():
        pytest.skip(f"{name} has not been resolved yet")
    assert _frames_match(resolve(name), pd.read_csv(committed))


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_resolved_matches_the_manifest_it_replaces(name):
    """The migration proof: the recipe lands on the pre-Tier-5 rows exactly."""
    legacy = DATA_DIR / LEGACY[name]
    if not legacy.exists():
        pytest.skip(f"legacy manifest {legacy.name} is gone; the recipe is now the only record")
    assert _frames_match(resolve(name), pd.read_csv(legacy))


# -- the rules the recipes are allowed to assume ------------------------------


def test_eval_corpora_cannot_enter_a_training_manifest():
    """The safeguard that used to be 'make_splits only globs one directory'."""
    from aigc_detect.data.corpus import assert_trainable

    with pytest.raises(SystemExit, match="may not be trained on"):
        assert_trainable("aigc_detect_bench", manifest_name="a-training-manifest")


def test_the_brief_s_benchmark_is_flagged_never_train():
    """5.4: 'Do not use the following data during training.'"""
    assert load_recipe("demo_val").never_train


def test_every_eval_tier_is_flagged_never_train():
    for name in ("demo_val", "ood", "heldout", "wildrf_test", "dalle3_holdout"):
        assert load_recipe(name).never_train, f"{name} is an eval tier but is not flagged"


def test_a_trainer_refuses_a_never_train_manifest():
    from aigc_detect.data.manifest import assert_trainable_manifest

    with pytest.raises(SystemExit, match="never_train"):
        assert_trainable_manifest("demo_val")
    assert_trainable_manifest("train")  # must not raise


def test_train_and_val_are_disjoint_and_exhaust_their_corpus():
    """They are two halves of one split; an overlap would be silent contamination."""
    train, val = resolve("train"), resolve("val")
    assert set(train.image_path).isdisjoint(set(val.image_path))
    assert len(train) + len(val) == len(get_corpus("tiny_genimage").rows())


def test_the_held_back_generators_are_absent_from_training():
    """DALLE2 and SDXL must stay unseen, or the OOD tier stops measuring anything."""
    generators = set(resolve("train_ext").generator.astype(str))
    assert not ({"DALLE2", "SDXL"} & generators)


def test_wildrf_train_reals_and_test_tier_share_no_image():
    assert set(resolve("wildrf_real").image_path).isdisjoint(set(resolve("wildrf_test").image_path))


def test_sid_real_carries_no_fakes():
    """The whole reason SID_Set is usable at all -- see sid_real.yaml."""
    assert set(resolve("sid_real").label.unique()) == {0}


def test_every_recipe_includes_only_registered_corpora():
    known = set(all_corpora())
    for name in RECIPES:
        for ref in load_recipe(name).includes:
            if ref.startswith("manifest:"):
                assert ref[len("manifest:"):] in RECIPES, f"{name} includes unknown manifest {ref}"
            else:
                assert ref in known, f"{name} includes unregistered corpus {ref!r}"


def test_dropped_corpora_are_marked_so_nothing_globs_them_back():
    """cifake's 120,000 rows point at images deleted in Tier 1."""
    assert get_corpus("cifake").role == "dropped"
    assert get_corpus("sd3").role == "quarantine"
