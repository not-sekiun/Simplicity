"""The backbone is a config value, not a code path.

THE ACCEPTANCE TEST THIS FILE EXISTS FOR: "change `backbone: dinov3-l` in an
experiment YAML and everything downstream works with no other edit."
`experiments/dinov3_linear.yaml` is that claim written down -- it is
`allsev_e1.yaml` with the backbone swapped -- and until now nothing checked
it. A claim that lives only in a YAML comment is exactly the kind that stays
true until someone adds a backbone-specific branch to the runner and no test
notices.

WHAT IS CHECKED HERE, AND WHAT IS NOT. Actually *running* `dinov3_linear`
needs `dinov3-l` embeddings cached for five manifests, which is a GPU re-embed
measured in hours -- see that config's own header. So this checks the half
that is a property of the code rather than of this machine's disk: the two
configs resolve, they differ in nothing but the backbone they name, and the
backbone they name is one the registry can actually load. `smoke_clean` is
the config that exercises the runner end to end for real (it asks for the
clean view only, which projects out of the store for free); `test_scripts.py`
and `test_features.py` cover the pieces underneath.

These tests read committed YAML only. They open no checkpoint, no embedding
and no image, so they run on a bare clone.
"""

from __future__ import annotations

import pytest

from aigc_detect.registry.backbones import list_backbones
from aigc_detect.train.experiment import config_hash, list_experiments, load_experiment

EXPERIMENTS = list_experiments()

#: The two configs the swap claim is about: same recipe, different backbone.
BASELINE = "allsev_e1"
SWAPPED = "dinov3_linear"

#: Keys of the resolved config that are ALLOWED to differ between them. Every
#: other key differing means the swap needed an edit somewhere else, which is
#: the thing this file exists to catch.
BACKBONE_BEARING_KEYS = {"backbone", "features"}


def test_there_are_experiments_to_test():
    assert EXPERIMENTS, "no configs found under experiments/ -- the registry is not wired up"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_config_resolves(name):
    """A malformed config must fail here, not three calls deep into training."""
    cfg = load_experiment(name)
    assert cfg["resolved"]["views"], f"{name}: views resolved to nothing"
    assert cfg["config_hash"], f"{name}: no config hash"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_names_a_registered_backbone(name):
    """`backbone:` is looked up, never constructed -- so a typo is a hard failure."""
    declared = load_experiment(name)["resolved"]["backbone"]
    assert declared in list_backbones(), f"{name}: unknown backbone {declared!r}"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_gather_step_names_the_experiment_s_backbone(name):
    """A `gather` naming a backbone the run never embeds reads an empty cache.

    `FeaturePipeline` gathers by backbone key, so a config whose `features:`
    still names the OLD backbone after a swap would fail at run time with a
    missing-cache error rather than here -- after the caller has waited for a
    manifest to resolve. Multi-backbone configs are the reason this is a
    subset check and not equality: a second `gather` is how an ensemble is
    declared, and only the first has to be the one `backbone:` names.
    """
    resolved = load_experiment(name)["resolved"]
    gathered = [s["backbone"] for s in resolved["features"] if s.get("op") == "gather"]
    assert gathered, f"{name}: a feature pipeline must start with a gather"
    assert resolved["backbone"] in gathered, (
        f"{name}: declares backbone {resolved['backbone']!r} but gathers {gathered}"
    )


def test_the_backbone_swap_touches_nothing_but_the_backbone():
    """experiments/dinov3_linear.yaml's whole reason for existing."""
    base = load_experiment(BASELINE)["resolved"]
    swapped = load_experiment(SWAPPED)["resolved"]

    differing = {k for k in base if base[k] != swapped[k]} | (set(swapped) - set(base))
    assert differing <= BACKBONE_BEARING_KEYS, (
        f"swapping the backbone also changed {sorted(differing - BACKBONE_BEARING_KEYS)} -- "
        "if that is deliberate, the swap is no longer a one-line demonstration"
    )
    assert base["backbone"] != swapped["backbone"], "the two configs name the same backbone"
    assert base["manifest"] == swapped["manifest"]
    assert base["views"] == swapped["views"]
    assert base["head"] == swapped["head"]
    assert base["train"] == swapped["train"]


def test_the_config_hash_moves_with_the_backbone():
    """Two runs that trained on different embeddings must not claim one hash.

    `Bundle.config_hash` is what answers "did this checkpoint come from the
    config on disk right now"; a hash that ignored the backbone would answer
    yes for a checkpoint trained on a completely different feature space.
    """
    base = load_experiment(BASELINE)
    swapped = load_experiment(SWAPPED)
    assert base["config_hash"] != swapped["config_hash"]
    assert config_hash(base["resolved"]) == base["config_hash"], "hashing is not deterministic"


def test_the_hash_covers_the_resolved_config_not_the_yaml():
    """A preset that expands differently must change the hash -- see the module
    docstring of `train.experiment`: `views: all_severities` is a NAME, and
    hashing the raw YAML would let two runs train on different view sets under
    one hash that claims they match."""
    cfg = load_experiment(BASELINE)
    assert cfg["raw"]["views"] == "all_severities", "this test is about the preset form"
    assert len(cfg["resolved"]["views"]) > 1, "the preset did not expand"
    assert config_hash({**cfg["resolved"], "views": ["clean"]}) != cfg["config_hash"]
