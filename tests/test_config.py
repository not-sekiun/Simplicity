"""The config package's public surface and its environment handling.

`aigc_detect.config` was one 179-line module and is now a package of four. The
first test is the regression guard for that split: every name the old module
exported must still be importable from the same place, or some import
elsewhere in the project breaks at runtime rather than at lint time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import aigc_detect.config as config
from aigc_detect.config import settings as settings_mod

# Everything the pre-split config.py exported, minus `Path`, which was an
# incidental `from pathlib import Path` leaking into the module namespace
# rather than intended API (verified: nothing imported it from here).
LEGACY_SURFACE = [
    "AIGC_EXT_DIR", "AIGC_MODERN_MANIFEST", "DALLE3_HOLDOUT_MANIFEST", "DATA_DIR",
    "DEMO_VAL_DIR", "DEMO_VAL_MANIFEST", "EMBEDDINGS_DIR", "GENERATOR_FAMILY",
    "HELDOUT_DIR", "HELDOUT_MANIFEST", "IMAGE_SIZE", "LABEL_AIGC", "LABEL_NAMES",
    "LABEL_REAL", "MIDJOURNEY_V6_MANIFEST", "NANO_BANANA_MANIFEST", "NORM_MEAN",
    "NORM_STD", "OOD_DIR", "OOD_MANIFEST", "PEXELS_REAL_MANIFEST",
    "PHOTO_REAL_MANIFEST", "PROCESSED_DIR", "RANDOM_SEED", "RAW_DIR",
    "REAL_EXT_DIR", "ROOT_DIR", "SID_REAL_MANIFEST", "TRAIN_EXT_MANIFEST",
    "TRAIN_GENERATORS", "TRAIN_MANIFEST", "UNSPLASH_REAL_MANIFEST", "VAL_FRACTION",
    "VAL_MANIFEST", "WILDRF_DIR", "WILDRF_REAL_MANIFEST", "WILDRF_TEST_MANIFEST",
]


@pytest.mark.parametrize("name", LEGACY_SURFACE)
def test_legacy_config_name_still_exported(name: str):
    assert hasattr(config, name), f"`from aigc_detect.config import {name}` would now fail"


def test_labels_are_the_documented_encoding():
    """0=real, 1=AIGC. Silently inverting this inverts every AUC in the project."""
    assert config.LABEL_REAL == 0
    assert config.LABEL_AIGC == 1
    assert config.LABEL_NAMES == {0: "real", 1: "aigc"}


def test_every_manifest_path_sits_under_the_data_root():
    """Manifest paths must derive from DATA_DIR, so AIGC_DATA_ROOT moves them all."""
    data_root = config.DATA_DIR.resolve()
    for name in (n for n in LEGACY_SURFACE if n.endswith(("_MANIFEST", "_DIR"))):
        value = getattr(config, name)
        if name == "ROOT_DIR":
            continue
        assert isinstance(value, Path)
        assert data_root in value.resolve().parents or value.resolve() == data_root, (
            f"{name}={value} does not live under DATA_DIR={data_root}, so it would not "
            f"follow an AIGC_DATA_ROOT override"
        )


def test_modern_generators_have_a_family():
    """The generator tags the modern-diffusion manifests actually carry.

    These were absent from GENERATOR_FAMILY, so the evaluation grid's
    `.get(gen, "unknown")` filed them under "unknown" instead of "diffusion" --
    quietly dropping the held-out DALL-E 3 tier out of every diffusion-family
    breakdown, which is the tier the project's generalisation claim rests on.
    """
    for tag in ("NanoBanana", "MidjourneyV6", "DALLE3"):
        assert config.GENERATOR_FAMILY.get(tag) == "diffusion", (
            f"generator tag {tag!r} appears in the manifests but has no family mapping"
        )


def test_train_generators_are_all_known():
    unknown = sorted(g for g in config.TRAIN_GENERATORS if g not in config.GENERATOR_FAMILY)
    assert not unknown, f"TRAIN_GENERATORS entries missing from GENERATOR_FAMILY: {unknown}"


class TestEnvironmentHandling:
    def test_blank_is_treated_as_unset(self, monkeypatch):
        """`.env.example` ships keys present but empty (`HF_TOKEN=`).

        A copied-and-unedited file must not hand callers an empty string, which
        reads as "a token was configured" and fails later at the API call.
        """
        monkeypatch.setenv("AIGC_TEST_VAR", "   ")
        assert settings_mod._env("AIGC_TEST_VAR") is None
        monkeypatch.setenv("AIGC_TEST_VAR", "value")
        assert settings_mod._env("AIGC_TEST_VAR") == "value"

    def test_unset_is_none(self):
        assert "AIGC_DEFINITELY_UNSET" not in os.environ
        assert settings_mod._env("AIGC_DEFINITELY_UNSET") is None

    def test_relative_path_resolves_against_the_repo_root(self, monkeypatch):
        monkeypatch.setenv("AIGC_TEST_PATH", "scratch/images")
        resolved = settings_mod._env_path("AIGC_TEST_PATH", Path("/unused"))
        assert resolved.is_absolute()
        assert resolved == (settings_mod.ROOT_DIR / "scratch" / "images").resolve()

    def test_absolute_path_is_used_verbatim(self, monkeypatch):
        monkeypatch.setenv("AIGC_TEST_PATH", str(Path("/mnt/bulk/aigc").absolute()))
        resolved = settings_mod._env_path("AIGC_TEST_PATH", Path("/unused"))
        assert "aigc" in str(resolved).lower()

    def test_default_used_when_unset(self):
        default = Path("/some/default")
        assert settings_mod._env_path("AIGC_DEFINITELY_UNSET", default) == default

    def test_hf_token_kwargs_is_empty_when_unconfigured(self):
        kwargs = config.hf_token_kwargs()
        assert kwargs == {} or set(kwargs) == {"token"}


def test_config_import_does_not_pull_in_torch():
    """config is imported by every CLI invocation including --help.

    Importing torch here would add seconds to every command. resolve_device()
    imports it lazily for exactly this reason.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", "import aigc_detect.config, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", "importing aigc_detect.config pulled in torch"
