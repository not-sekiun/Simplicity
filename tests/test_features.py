"""FeaturePipeline: the one place raw-embedding-to-head-input arithmetic lives.

See `aigc_detect.train.features`'s module docstring for the failure mode this
closes (two hand-written copies of `(x - mean) / std`, one of which was
`demo/server.py`'s own reimplementation against the same checkpoint fields).
These tests assert the three properties that docstring promises: a fitted
`standardize` step survives a `to_dict`/`from_dict` round trip, fitting only
ever touches the array passed to `fit` (never a second "val" array), and a
two-`gather` pipeline concatenates rather than overwrites.
"""

from __future__ import annotations

import numpy as np
import pytest

from aigc_detect.train.features import FeaturePipeline, GatherOp, L2NormOp, StandardizeOp


def test_a_pipeline_must_start_with_gather():
    with pytest.raises(ValueError, match="first step"):
        FeaturePipeline.from_spec([{"op": "standardize"}])


def test_an_empty_spec_is_rejected():
    with pytest.raises(ValueError, match="at least one step"):
        FeaturePipeline.from_spec([])


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError, match="unknown op"):
        FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "whiten"}])


def test_transform_before_fit_raises_on_an_unfit_standardize():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    x = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="has not been fit"):
        pipeline.transform({"bb": x})


def test_gather_alone_passes_the_array_through_unchanged():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}])
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = pipeline.transform({"bb": x})
    np.testing.assert_allclose(out, x)


def test_l2norm_makes_every_row_unit_length():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "l2norm"}])
    x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    out = pipeline.transform({"bb": x})
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_l2norm_guards_the_zero_vector_instead_of_dividing_by_zero():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "l2norm"}])
    x = np.zeros((1, 4), dtype=np.float32)
    out = pipeline.transform({"bb": x})
    assert np.isfinite(out).all()


def test_standardize_zero_means_and_unit_stds_the_fit_data():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    rng = np.random.RandomState(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(500, 6)).astype(np.float32)
    pipeline.fit({"bb": x})
    out = pipeline.transform({"bb": x})
    np.testing.assert_allclose(out.mean(axis=0), np.zeros(6), atol=1e-4)
    np.testing.assert_allclose(out.std(axis=0), np.ones(6), atol=1e-3)


def test_standardize_guards_a_constant_dimension():
    """A dead/constant dimension has std=0; dividing by it must not produce inf/nan."""
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    x = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]], dtype=np.float32)
    pipeline.fit({"bb": x})
    out = pipeline.transform({"bb": x})
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 0], np.zeros(3))  # constant dim -> exactly zero after centering


def test_fit_only_reads_the_array_it_is_given_never_a_second_one():
    """The module docstring's whole point: `fit` takes exactly one array. A
    scaler fit on `train` and applied to `val` must use TRAIN's statistics,
    not something derived from val, however val is later passed to transform."""
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    train = np.array([[0.0], [2.0]], dtype=np.float32)  # mean=1, std=1
    val = np.array([[100.0], [200.0]], dtype=np.float32)  # wildly different distribution
    pipeline.fit({"bb": train})
    mean_before, std_before = pipeline.ops[-1].mean.copy(), pipeline.ops[-1].std.copy()
    pipeline.transform({"bb": val})  # must not mutate the fitted state
    np.testing.assert_allclose(pipeline.ops[-1].mean, mean_before)
    np.testing.assert_allclose(pipeline.ops[-1].std, std_before)
    # and the transform of train itself must be exactly (train - 1) / 1
    np.testing.assert_allclose(pipeline.transform({"bb": train}), np.array([[-1.0], [1.0]]))


def test_two_gather_steps_concatenate_not_overwrite():
    pipeline = FeaturePipeline.from_spec(
        [{"op": "gather", "backbone": "a"}, {"op": "gather", "backbone": "b"}]
    )
    xa = np.array([[1.0, 2.0]], dtype=np.float32)
    xb = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    out = pipeline.transform({"a": xa, "b": xb})
    np.testing.assert_allclose(out, np.array([[1.0, 2.0, 10.0, 20.0, 30.0]]))
    assert pipeline.backbones == ("a", "b")


def test_gather_missing_backbone_raises_with_the_available_keys():
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}])
    with pytest.raises(KeyError):
        pipeline.transform({"other": np.zeros((1, 2), dtype=np.float32)})


# -- round-tripping (to_dict / from_dict) --------------------------------------


def test_an_unfit_pipeline_round_trips_to_an_unfit_pipeline():
    pipeline = FeaturePipeline.from_spec(
        [{"op": "gather", "backbone": "bb"}, {"op": "l2norm"}, {"op": "standardize"}]
    )
    restored = FeaturePipeline.from_dict(pipeline.to_dict())
    assert len(restored.ops) == 3
    assert isinstance(restored.ops[0], GatherOp) and restored.ops[0].backbone == "bb"
    assert isinstance(restored.ops[1], L2NormOp)
    assert isinstance(restored.ops[2], StandardizeOp) and not restored.ops[2].fitted


def test_a_fitted_standardize_step_survives_round_tripping():
    """This is the property `inference.bundle` depends on: `torch.save`/`load`
    round-trips the dict form, and a fresh process must be able to `transform`
    with it having never called `fit`."""
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    x = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    pipeline.fit({"bb": x})
    expected = pipeline.transform({"bb": x})

    restored = FeaturePipeline.from_dict(pipeline.to_dict())
    assert restored.ops[-1].fitted
    # never called .fit() on `restored` -- it must still transform correctly.
    np.testing.assert_allclose(restored.transform({"bb": x}), expected, rtol=1e-5)


def test_to_dict_is_json_safe_plain_python_types():
    """A bundle is written with `torch.save`, but the dict itself must not
    depend on numpy surviving the round trip -- `to_dict` promises plain
    floats/lists, not ndarrays, in its own docstring's contract."""
    pipeline = FeaturePipeline.from_spec([{"op": "gather", "backbone": "bb"}, {"op": "standardize"}])
    pipeline.fit({"bb": np.array([[1.0, 2.0]], dtype=np.float32)})
    d = pipeline.to_dict()
    import json

    json.dumps(d)  # raises TypeError if anything is not JSON-serializable
