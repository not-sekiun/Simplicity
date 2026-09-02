"""The feature pipeline: what turns a raw pooled embedding into what the head sees.

Standardization used to be baked directly into ``train_head_on_views`` (see
:mod:`aigc_detect.train.probe`): the function computed a train-set mean/std as a
local variable, folded it into the training rows before the loop ever started,
and stashed the two arrays into the checkpoint dict as ``scaler_mean``/
``scaler_std``. That made "how a checkpoint's raw embedding becomes its input"
an implementation detail of one function, private to it -- which is exactly why
``demo/server.py`` could not reuse it and re-implemented the same
subtract-divide by hand against the same checkpoint fields (see its
``_HeadRunner``). Two independent implementations of one preprocessing step is
the failure mode this module closes: there is now exactly one place that knows
how to turn embeddings into head input, it declares itself in a few lines of
YAML, and it is a value a bundle can carry rather than a fact a checkpoint's
consumer has to already know.

THE THREE OPS

    gather        select one backbone's embeddings and hstack onto whatever has
                  been gathered so far. A two-backbone concatenation is just two
                  `gather` steps in a row -- there is no separate "concat" op,
                  because gather already accumulates.
    l2norm        row-wise L2 normalization (x / ||x||_2). Stateless.
    standardize   (x - mean) / std, per dimension. The only op with state: mean
                  and std are FIT, not declared, and travel with the pipeline
                  once fit.

A pipeline is a sequence of these, always starting with at least one `gather`
(there is nothing to normalize or standardize before something has been
gathered). `fit`/`transform` mirror scikit-learn's contract deliberately, so
the shape is familiar; the pipeline itself has no scikit-learn dependency.

WHY THE SCALER IS FIT ON THE CLEAN TRAIN VIEW ONLY, NEVER ON VAL. Quoting
`train.probe`'s module docstring, which this preserves rather than restates
from memory: "Features are standardized (zero mean, unit std) using TRAIN-set
statistics only -- the val set never contributes to the scaler, to avoid
leakage." Concretely: a scaler is a summary of the training distribution. Let
val statistics leak into it and every val-set score is now partly a comparison
against itself, which inflates the exact metrics (AUC, balanced accuracy) that
`train_head_on_views` prints per epoch to decide whether the model is any
good -- the number would look better and mean less, and nothing about it would
signal that anything had gone wrong. `FeaturePipeline.fit` therefore takes
exactly one array (or one dict of arrays, for a multi-backbone gather):
whatever the caller passes as "train". There is no `fit(train, val)` overload
to make the leak a one-line accident.

ROUND-TRIPPING. `to_dict`/`from_dict` serialize the declared ops AND any fitted
state (a `standardize` op's mean/std) into one plain-dict form, so a bundle
(:mod:`aigc_detect.inference.bundle`) can carry a fitted pipeline through
`torch.save`/`torch.load` without any pipeline-specific pickling logic, and a
fresh process can `transform` new data with it having never called `fit` at
all -- which is exactly what inference does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class GatherOp:
    """Select one backbone's array and hstack it onto the running matrix."""

    backbone: str

    def to_dict(self) -> dict:
        return {"op": "gather", "backbone": self.backbone}


@dataclass
class L2NormOp:
    """Row-wise L2 normalization. Stateless -- nothing to fit."""

    def to_dict(self) -> dict:
        return {"op": "l2norm"}


@dataclass
class StandardizeOp:
    """Per-dimension (x - mean) / std. `mean`/`std` are None until fit."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def to_dict(self) -> dict:
        d: dict = {"op": "standardize"}
        if self.fitted:
            d["mean"] = [float(x) for x in self.mean]
            d["std"] = [float(x) for x in self.std]
        return d


FeatureOp = GatherOp | L2NormOp | StandardizeOp

_KNOWN_OPS = ("gather", "l2norm", "standardize")


def _op_from_dict(spec: dict) -> FeatureOp:
    op = spec.get("op")
    if op == "gather":
        if "backbone" not in spec:
            raise ValueError("[features] a `gather` step needs a `backbone` key")
        return GatherOp(backbone=str(spec["backbone"]))
    if op == "l2norm":
        return L2NormOp()
    if op == "standardize":
        has_mean, has_std = "mean" in spec, "std" in spec
        if has_mean != has_std:
            raise ValueError("[features] a `standardize` step must serialize `mean` and `std` together")
        mean = np.asarray(spec["mean"], dtype=np.float32) if has_mean else None
        std = np.asarray(spec["std"], dtype=np.float32) if has_std else None
        return StandardizeOp(mean=mean, std=std)
    raise ValueError(f"[features] unknown op {op!r}. Expected one of {_KNOWN_OPS}.")


@dataclass
class FeaturePipeline:
    """A declared, serializable sequence of feature ops.

    Construct via :meth:`from_spec` (the raw ``features:`` YAML list) or
    :meth:`from_dict` (a round-tripped, possibly-fitted pipeline). Call
    :meth:`fit` once, on train data only, before the first :meth:`transform` of
    anything that includes a `standardize` step -- `transform` raises if it
    reaches an unfit `standardize` op rather than silently skipping it.
    """

    ops: tuple[FeatureOp, ...]

    @classmethod
    def from_spec(cls, steps: Sequence[dict]) -> FeaturePipeline:
        if not steps:
            raise ValueError("[features] a pipeline needs at least one step")
        ops = tuple(_op_from_dict(dict(s)) for s in steps)
        if not isinstance(ops[0], GatherOp):
            raise ValueError(
                "[features] the first step must be `gather` -- there is nothing to l2norm or "
                "standardize before something has been gathered"
            )
        return cls(ops=ops)

    @property
    def backbones(self) -> tuple[str, ...]:
        """Every backbone this pipeline gathers from, in declaration order."""
        return tuple(op.backbone for op in self.ops if isinstance(op, GatherOp))

    def fit(self, X_train: Mapping[str, np.ndarray]) -> FeaturePipeline:
        """Fit every stateful op (today: `standardize`) against X_train, in place.

        `X_train` maps backbone key -> that backbone's pooled embeddings for the
        CLEAN train view, row-aligned across backbones for a multi-gather
        pipeline. Returns `self` so `pipeline = FeaturePipeline.from_spec(spec).fit(X_train)`
        reads as one expression. See the module docstring for why this takes
        train alone, with no val parameter to leak through.
        """
        self._run(X_train, fitting=True)
        return self

    def transform(self, X: Mapping[str, np.ndarray]) -> np.ndarray:
        """Apply the (already-fit, where stateful) pipeline to `X`."""
        return self._run(X, fitting=False)

    def _run(self, X: Mapping[str, np.ndarray], *, fitting: bool) -> np.ndarray:
        current: np.ndarray | None = None
        for op in self.ops:
            if isinstance(op, GatherOp):
                if op.backbone not in X:
                    raise KeyError(
                        f"[features] gather step needs backbone '{op.backbone}', got {sorted(X)}"
                    )
                arr = np.asarray(X[op.backbone], dtype=np.float32)
                current = arr if current is None else np.hstack([current, arr])
            elif isinstance(op, L2NormOp):
                assert current is not None
                norms = np.linalg.norm(current, axis=1, keepdims=True)
                current = current / np.maximum(norms, _EPS)
            elif isinstance(op, StandardizeOp):
                assert current is not None
                if fitting:
                    mean = current.mean(axis=0)
                    std = current.std(axis=0)
                    std = np.where(std == 0, 1.0, std)  # guard dead/constant dims
                    op.mean, op.std = mean.astype(np.float32), std.astype(np.float32)
                elif not op.fitted:
                    raise RuntimeError(
                        "[features] `standardize` step has not been fit -- call "
                        "pipeline.fit(X_train) before transform(), or load a pipeline that "
                        "already carries fitted mean/std (see FeaturePipeline.from_dict)"
                    )
                current = (current - op.mean) / op.std
            else:  # pragma: no cover -- exhaustive over FeatureOp by construction
                raise TypeError(f"[features] unreachable op {op!r}")
        assert current is not None
        return current

    def to_dict(self) -> dict:
        return {"steps": [op.to_dict() for op in self.ops]}

    @classmethod
    def from_dict(cls, d: dict) -> FeaturePipeline:
        return cls.from_spec(d["steps"])
