"""Calibration runs the model's OWN preprocessing, whatever it is.

`train.calibrate` used to take `(mean, std)` and apply `(x - mean) / std`
itself. That made it a third implementation of "raw pooled embedding -> head
input" -- alongside `train.probe` and the demo server's `Model` -- and it
quietly narrowed what a trained model was allowed to be, because a pipeline
that did anything else before standardizing would have been calibrated against
preprocessing its head never saw. `train.experiment` had to defend against
that with an explicit refusal: any `features:` list not ending in a bare
`standardize` raised rather than mis-calibrate.

Both halves of that are gone, and these tests are what keeps them gone. The
protocol itself -- the split, the sweep, the half-B report -- is unchanged and
is regression-tested against FINDINGS 2j's recorded numbers in
`test_bundle.py`; what is tested here is that the arithmetic applied on the way
in is the pipeline's, not a copy of one shape of it.

Hermetic: synthetic caches in a tmp directory, no `data/` and no checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aigc_detect.train.calibrate import CDN_VIEWS, WILDRF_STEM, pooled_wildrf_scores
from aigc_detect.train.features import FeaturePipeline

DIM = 8
N = 40


def _write_caches(tmp_path, backbone_key: str, seed: int = 0):
    """One .npz per CDN view, same rows across views -- the row-for-row
    projection shape `pooled_wildrf_scores` relies on to index by image."""
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1] * (N // 2))
    for view in CDN_VIEWS:
        emb = rng.normal(size=(N, DIM)).astype(np.float32)
        np.savez(tmp_path / f"{backbone_key}__{WILDRF_STEM}__{view}.npz", embeddings=emb, labels=labels)
    return labels


def _head(in_dim: int):
    torch.manual_seed(0)
    h = torch.nn.Linear(in_dim, 1)
    h.eval()
    return h


def _fit(pipeline: FeaturePipeline, arrays: dict) -> FeaturePipeline:
    return pipeline.fit(arrays)


def test_a_pipeline_with_l2norm_calibrates(tmp_path):
    """The case the old `(mean, std)` signature made unrepresentable.

    `l2norm` before `standardize` is a legitimate declared pipeline; before
    this, `train.experiment` refused to run any config containing one because
    calibration could not have honoured it.
    """
    _write_caches(tmp_path, "pe-core-l")
    train = np.random.default_rng(1).normal(size=(N, DIM)).astype(np.float32)

    pipeline = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"},
        {"op": "l2norm"},
        {"op": "standardize"},
    ])
    _fit(pipeline, {"pe-core-l": train})

    probs, labels, image_idx = pooled_wildrf_scores(_head(DIM), pipeline, tmp_path)

    assert probs.shape == (N * len(CDN_VIEWS),)
    assert labels.shape == probs.shape
    assert image_idx.max() == N - 1, "image index must span the manifest, not the pooled rows"
    assert np.isfinite(probs).all()


def test_l2norm_actually_changes_the_scores(tmp_path):
    """Guards against the pipeline being accepted and then ignored -- an op
    that is honoured must be visible in the output."""
    _write_caches(tmp_path, "pe-core-l")
    train = np.random.default_rng(1).normal(size=(N, DIM)).astype(np.float32)
    head = _head(DIM)

    plain = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"}, {"op": "standardize"},
    ])
    normed = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"}, {"op": "l2norm"}, {"op": "standardize"},
    ])
    _fit(plain, {"pe-core-l": train})
    _fit(normed, {"pe-core-l": train})

    a, _, _ = pooled_wildrf_scores(head, plain, tmp_path)
    b, _, _ = pooled_wildrf_scores(head, normed, tmp_path)
    assert not np.allclose(a, b), "l2norm was declared but made no difference to the scores"


def test_two_backbone_pipeline_gathers_both_caches(tmp_path):
    """The plan's multi-backbone concat, through calibration: two gathers, two
    caches, one hstacked matrix. The old signature took ONE `backbone_key`, so
    this could not be expressed at all."""
    _write_caches(tmp_path, "pe-core-l", seed=0)
    _write_caches(tmp_path, "dinov3-l", seed=2)
    rng = np.random.default_rng(3)
    train = {
        "pe-core-l": rng.normal(size=(N, DIM)).astype(np.float32),
        "dinov3-l": rng.normal(size=(N, DIM)).astype(np.float32),
    }

    pipeline = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"},
        {"op": "gather", "backbone": "dinov3-l"},
        {"op": "standardize"},
    ])
    _fit(pipeline, train)

    probs, _, _ = pooled_wildrf_scores(_head(DIM * 2), pipeline, tmp_path)
    assert probs.shape == (N * len(CDN_VIEWS),)


def test_a_missing_cache_names_the_backbone_that_is_missing(tmp_path):
    """A two-backbone pipeline with one cache present must say WHICH one to
    embed -- the old message could only ever name the single key it was given.
    """
    _write_caches(tmp_path, "pe-core-l")
    train = {
        "pe-core-l": np.random.default_rng(1).normal(size=(N, DIM)).astype(np.float32),
        "dinov3-l": np.random.default_rng(2).normal(size=(N, DIM)).astype(np.float32),
    }
    pipeline = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"},
        {"op": "gather", "backbone": "dinov3-l"},
        {"op": "standardize"},
    ])
    _fit(pipeline, train)

    with pytest.raises(SystemExit, match="dinov3-l"):
        pooled_wildrf_scores(_head(DIM * 2), pipeline, tmp_path)


def test_an_unfit_pipeline_refuses_to_calibrate(tmp_path):
    """A threshold derived through an unfit `standardize` would be a threshold
    for a model that does not exist yet. `FeaturePipeline.transform` is what
    refuses; this asserts calibration does not route around it."""
    _write_caches(tmp_path, "pe-core-l")
    pipeline = FeaturePipeline.from_spec([
        {"op": "gather", "backbone": "pe-core-l"}, {"op": "standardize"},
    ])
    with pytest.raises(RuntimeError, match="has not been fit"):
        pooled_wildrf_scores(_head(DIM), pipeline, tmp_path)
