"""The model bundle: round-tripping, legacy-checkpoint upgrade, and the
threshold-derivation protocol's regression check against FINDINGS 2j.

See `aigc_detect.inference.bundle`'s module docstring for what a bundle
replaces (a checkpoint dict plus a module constant three files away) and why
a legacy `.pt` must upgrade in memory rather than be migrated on disk: 25
archived heads under `models/archive/` back numbers cited in
`docs/findings.md`, and nothing here may make those numbers unreproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aigc_detect.inference.bundle import (
    BUNDLE_VERSION,
    LEGACY_DEFAULT_THRESHOLD,
    BundleBackbone,
    load_bundle,
    save_bundle,
)
from aigc_detect.registry.heads import build_head
from aigc_detect.train.features import FeaturePipeline

IN_DIM = 4


def _fitted_pipeline(backbone_key: str = "test-bb") -> FeaturePipeline:
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": backbone_key}, {"op": "standardize"}])
    rng = np.random.RandomState(0)
    x = rng.normal(size=(50, IN_DIM)).astype(np.float32)
    pipeline.fit({backbone_key: x})
    return pipeline


def _backbone_ref(key: str = "test-bb") -> BundleBackbone:
    return BundleBackbone(
        key=key, checkpoint="test/checkpoint", revision="abc123", dim=IN_DIM, native_res=224,
        norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5), bb_id="deadbeef",
    )


# -- native bundle round trip --------------------------------------------------


def test_save_then_load_round_trips_every_field(tmp_path: Path):
    pipeline = _fitted_pipeline()
    head = build_head("linear", IN_DIM)
    path = save_bundle(
        tmp_path / "bundle.pt",
        backbone=_backbone_ref(), features=pipeline, head_kind="linear",
        head_state_dict=head.state_dict(), threshold=0.75, threshold_source="unit test",
        config_hash="abc123hash", metrics={"final_val_auc": 0.99},
    )
    bundle = load_bundle(path)

    assert bundle.bundle_version == BUNDLE_VERSION
    assert bundle.backbone.key == "test-bb"
    assert bundle.backbone.bb_id == "deadbeef"
    assert bundle.head_kind == "linear"
    assert bundle.threshold == 0.75
    assert bundle.threshold_source == "unit test"
    assert bundle.config_hash == "abc123hash"
    assert bundle.metrics["final_val_auc"] == 0.99


def test_round_tripped_pipeline_produces_identical_output(tmp_path: Path):
    """The property that actually matters: a fresh process's bundle transforms
    new data exactly as the pipeline that trained the head did, having never
    called `.fit()` -- see FeaturePipeline's module docstring."""
    pipeline = _fitted_pipeline()
    head = build_head("linear", IN_DIM)
    x = np.random.RandomState(1).normal(size=(10, IN_DIM)).astype(np.float32)
    expected = pipeline.transform({"test-bb": x})

    path = save_bundle(
        tmp_path / "bundle.pt", backbone=_backbone_ref(), features=pipeline, head_kind="linear",
        head_state_dict=head.state_dict(), threshold=0.5, threshold_source="unit test",
    )
    bundle = load_bundle(path)
    got = bundle.features.transform({"test-bb": x})
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_round_tripped_head_produces_identical_predictions(tmp_path: Path):
    pipeline = _fitted_pipeline()
    head = build_head("linear", IN_DIM)
    x = torch.randn(6, IN_DIM)
    with torch.no_grad():
        expected = head(x)

    path = save_bundle(
        tmp_path / "bundle.pt", backbone=_backbone_ref(), features=pipeline, head_kind="linear",
        head_state_dict=head.state_dict(), threshold=0.5, threshold_source="unit test",
    )
    restored_head = load_bundle(path).build_head()
    with torch.no_grad():
        got = restored_head(x)
    torch.testing.assert_close(got, expected)


def test_config_hash_and_metrics_default_to_none_and_empty(tmp_path: Path):
    pipeline = _fitted_pipeline()
    head = build_head("linear", IN_DIM)
    path = save_bundle(
        tmp_path / "bundle.pt", backbone=_backbone_ref(), features=pipeline, head_kind="linear",
        head_state_dict=head.state_dict(), threshold=0.5, threshold_source="unit test",
    )
    bundle = load_bundle(path)
    assert bundle.config_hash is None
    assert bundle.metrics == {}


def test_missing_bundle_file_raises_a_clear_error(tmp_path: Path):
    with pytest.raises(SystemExit, match="no checkpoint/bundle"):
        load_bundle(tmp_path / "does-not-exist.pt")


# -- legacy checkpoint upgrade --------------------------------------------------


def _write_legacy_checkpoint(path: Path, *, backbone: str = "test-bb", in_dim: int = IN_DIM) -> None:
    head = build_head("linear", in_dim)
    mean = np.zeros(in_dim, dtype=np.float32)
    std = np.ones(in_dim, dtype=np.float32)
    torch.save(
        {
            "state_dict": head.state_dict(), "head_kind": "linear", "backbone": backbone,
            "in_dim": in_dim, "scaler_mean": mean, "scaler_std": std,
            "final_val_auc": 0.987, "epochs": 1, "lr": 1e-3,
        },
        path,
    )


def test_a_legacy_checkpoint_upgrades_to_bundle_version_zero(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    _write_legacy_checkpoint(path)
    bundle = load_bundle(path)
    assert bundle.bundle_version == 0
    assert bundle.head_kind == "linear"
    assert bundle.backbone.key == "test-bb"
    assert bundle.backbone.dim == IN_DIM
    # Registry-only identity: this backbone key is not in BACKBONE_REGISTRY, so
    # everything the registry would have supplied is honestly None/absent.
    assert bundle.backbone.norm_mean is None
    assert bundle.backbone.norm_std is None
    assert bundle.backbone.bb_id is None


def test_a_legacy_checkpoint_gets_the_documented_legacy_threshold_honestly_sourced(tmp_path: Path):
    """The number this test pins (0.980) is not invented here -- it is the
    threshold every checkpoint shipped before this tier implicitly used as a
    module constant (`inference.predict.DECISION_THRESHOLD`). Upgrading the
    FORMAT must not silently change the SCORE a legacy head flips a verdict
    at, and `threshold_source` must say so plainly rather than reading like a
    freshly calibrated number."""
    path = tmp_path / "legacy.pt"
    _write_legacy_checkpoint(path)
    bundle = load_bundle(path)
    assert bundle.threshold == LEGACY_DEFAULT_THRESHOLD == 0.980
    assert bundle.threshold_source == "legacy default 0.980"
    assert bundle.config_hash is None


def test_legacy_scaler_mean_std_become_a_fitted_standardize_step(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    _write_legacy_checkpoint(path, backbone="test-bb")
    bundle = load_bundle(path)
    assert bundle.features.ops[-1].fitted
    x = np.ones((2, IN_DIM), dtype=np.float32)
    # mean=0, std=1 in the fixture -> transform is the identity.
    np.testing.assert_allclose(bundle.features.transform({"test-bb": x}), x)


def test_a_dict_missing_legacy_keys_is_rejected_not_silently_misread(tmp_path: Path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"some_other_key": 1}, path)
    with pytest.raises(SystemExit, match="neither a bundle"):
        load_bundle(path)


def test_legacy_metrics_only_carries_fields_the_original_run_actually_recorded(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    _write_legacy_checkpoint(path)
    bundle = load_bundle(path)
    assert bundle.metrics["final_val_auc"] == 0.987
    assert bundle.metrics["epochs"] == 1
    # never written by _write_legacy_checkpoint -> must not appear, invented, in metrics
    assert "final_val_auc_robust_pooled" not in bundle.metrics


# -- the threshold-derivation protocol vs. FINDINGS 2j -------------------------


def test_verify_recorded_table_reproduces_findings_2j_where_the_data_exists():
    """Regression test on the PROTOCOL (train.calibrate), not on any one head.
    Skips rows whose checkpoint is not on this machine, so the suite stays
    green on a fresh clone that has not fetched `models/archive/`."""
    from aigc_detect.train.calibrate import verify_recorded_table

    rows = verify_recorded_table()
    assert rows, "RECORDED_TABLE is empty -- nothing to verify"
    if all(r["skipped"] for r in rows):
        pytest.skip("none of FINDINGS 2j's archived heads are on disk on this machine")
    for r in rows:
        if r["skipped"]:
            continue
        assert r["ok"], f"{r['head']}: protocol drift -- recorded {r['recorded']}, got {r['got']}"
