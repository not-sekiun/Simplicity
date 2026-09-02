"""The model bundle: one versioned artifact that knows what it needs to run.

WHAT WAS SCATTERED. A trained head used to be a dict with seven-ish keys
(`state_dict`, `head_kind`, `backbone`, `in_dim`, `scaler_mean`, `scaler_std`,
...) and NOTHING about the decision threshold, which lived instead as a module
constant, `inference.predict.DECISION_THRESHOLD`, calibrated for exactly one
checkpoint and never re-derived automatically on a swap (see
`scripts/derive_threshold.py`'s docstring for how badly that went the one time
it *was* re-derived by hand and disagreed with itself). Swap the checkpoint a
caller of `predict.py` points at without also updating that constant and every
prediction is silently thresholded against the wrong model's calibration --
the JSON output looks identical, the summary line lies with confidence.

WHAT A BUNDLE IS. One artifact carrying everything a checkpoint used to carry,
plus:

    bundle_version    schema version, so a future format change has somewhere
                       to branch on rather than guessing from which keys exist
    backbone           WHICH model, including its resolved revision -- see
                       `BundleBackbone` and `aigc_detect.cache.identity`
    features            the fitted `FeaturePipeline` (gather/l2norm/standardize),
                       not two bare arrays a caller has to know how to apply
    head_kind / head_state_dict
    threshold + threshold_source
                       the operating point AND, in one string, how it was
                       derived -- so "why 0.980" is a fact the artifact carries,
                       not a comment three modules away
    config_hash         which resolved experiment config produced this, or None
                       for anything not trained through `train.experiment`
    metrics              a snapshot (val AUC, robustness numbers, whatever the
                       trainer measured) for provenance, not for re-computation

`save_bundle`/`load_bundle` are the whole API. A caller that has a bundle knows
its own threshold and its own preprocessing, so `predict.py` and the demo
server become "load bundle, run" -- see `inference.predict.run_inference`,
which now does exactly that.

LEGACY CHECKPOINTS UPGRADE IN MEMORY. `docs/findings.md` cites numbers for 25
archived heads under `models/archive/`, and nothing may be done that makes
those numbers unreproducible. `load_bundle` recognises the old dict shape (no
`bundle_version` key) and synthesizes a bundle from it: a `gather` + already-
fit `standardize` pipeline from `scaler_mean`/`scaler_std`, and -- because a
legacy checkpoint never recorded one -- `threshold=LEGACY_DEFAULT_THRESHOLD`
with `threshold_source="legacy default 0.980"`, rather than inventing a number
that would read as calibrated when it was actually a guess carried forward.
`bundle_version=0` distinguishes an upgraded legacy checkpoint from a native
bundle for any caller that cares.

WHY NOT RESOLVE THE LEGACY BACKBONE'S NORMALIZATION STATS TOO. Those come from
the loaded checkpoint itself (`timm.data.resolve_model_data_config` or the HF
image processor) or from a hit in the embedding store's memo (see
`cache.identity.identity_from_store`) -- both cost either a network+GPU load or
a store this module has no business opening. `load_bundle` must stay a CPU,
no-GPU, no-network operation (25 archived heads have to stay loadable in a
test), so a legacy upgrade's `BundleBackbone` carries only what the backbone
REGISTRY already knows for free (checkpoint string, revision, native_res) and
leaves `norm_mean`/`norm_std`/`bb_id` as `None`. A caller that needs those for
a legacy bundle resolves them the same way any other cache-store consumer does
-- `cache.identity.resolve_identity` -- which is an explicit, visible cost
rather than one hidden inside a loader that looks cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aigc_detect.registry.backbones import BACKBONE_REGISTRY
from aigc_detect.train.features import FeaturePipeline

#: Bumped only when the on-disk dict shape changes in a way `load_bundle`
#: cannot paper over (a renamed key, a dropped field a loader relies on).
#: `bundle_version=0` is reserved for an upgraded legacy checkpoint -- it never
#: appears in a freshly `save_bundle`d file.
BUNDLE_VERSION = 1

# The threshold every checkpoint shipped before this tier implicitly used,
# because it lived as a module constant (`inference.predict.DECISION_THRESHOLD`)
# rather than travelling with the checkpoint. Kept here, not re-derived, so a
# legacy bundle's threshold is bit-identical to what `predict.py` has always
# applied to that same file -- upgrading the FORMAT must not silently change
# the SCORE at which a legacy head flips from "real" to "aigc".
#
# 0.98, chosen on a HELD-OUT split of WildRF (2,503 real Reddit/X/Facebook
# photographs and real-world AI, which nothing trains on), pooled over clean +
# the CDN-like views a browser extension actually sees (jpeg_q70, jpeg_q90,
# resize_0.5x, chain_light), split BY IMAGE, swept 0.50..0.999 in 0.005 steps,
# picked by F1 on half A, reported on half B. That protocol is
# `train.calibrate`'s `calibrate_threshold` (promoted from
# `scripts/derive_threshold.py`, which reconstructed it from prose twice and
# got two different answers the first time -- see that module's docstring).
# At this threshold the shipping head (`pe-core-l__linear__allsev_e1.pt`)
# measures HELD-OUT FPR 0.0215 / TPR 0.9797; at 0.5 the same tier gives FPR
# 0.1875 / TPR 0.9949, so 0.98 is a large cut in false positives for a small
# amount of recall -- telling someone their own photograph is AI-generated
# costs far more than missing one AI image among many.
LEGACY_DEFAULT_THRESHOLD = 0.980


@dataclass(frozen=True)
class BundleBackbone:
    """Which model produced the embeddings this bundle's head was trained on.

    Mirrors `cache.identity.BackboneIdentity` deliberately -- that dataclass IS
    the project's answer to "which model", `bb_id` included, and a freshly
    trained bundle should carry that identity verbatim (`from_identity`) rather
    than a bundle-specific reinvention of the same five-plus-one fields. The
    two are not merged into one class because `BackboneIdentity` requires
    `norm_mean`/`norm_std` to be actual floats -- correct for its job, which is
    keying vectors in a store that must never silently mix two normalizations
    -- while a legacy-upgraded bundle (`from_registry_key`) may not have them
    without loading a checkpoint this module promises not to load.
    """

    key: str
    checkpoint: str | None
    revision: str | None
    dim: int
    native_res: int | None
    norm_mean: tuple[float, ...] | None
    norm_std: tuple[float, ...] | None
    bb_id: str | None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "dim": self.dim,
            "native_res": self.native_res,
            "norm_mean": list(self.norm_mean) if self.norm_mean is not None else None,
            "norm_std": list(self.norm_std) if self.norm_std is not None else None,
            "bb_id": self.bb_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BundleBackbone:
        norm_mean = d.get("norm_mean")
        norm_std = d.get("norm_std")
        return cls(
            key=d["key"],
            checkpoint=d.get("checkpoint"),
            revision=d.get("revision"),
            dim=int(d["dim"]),
            native_res=d.get("native_res"),
            norm_mean=tuple(norm_mean) if norm_mean is not None else None,
            norm_std=tuple(norm_std) if norm_std is not None else None,
            bb_id=d.get("bb_id"),
        )

    @classmethod
    def from_identity(cls, identity: Any) -> BundleBackbone:
        """From an already-resolved `cache.identity.BackboneIdentity`."""
        return cls(
            key=identity.key, checkpoint=identity.checkpoint, revision=identity.revision,
            dim=identity.dim, native_res=identity.native_res,
            norm_mean=tuple(identity.norm_mean), norm_std=tuple(identity.norm_std),
            bb_id=identity.bb_id,
        )

    @classmethod
    def from_registry_key(cls, key: str, dim: int) -> BundleBackbone:
        """Best-effort ref for a legacy checkpoint: registry fields only.

        No `norm_mean`/`norm_std`/`bb_id` -- resolving those needs either the
        loaded weights or a store memo hit (`cache.identity`), and this is the
        path `load_bundle` takes for a plain checkpoint dict with no such
        context available. `dim` comes from the checkpoint's own `in_dim`
        rather than the registry's `pooled_dim`, so a bundle for a mismatched
        or since-changed registry entry still records what the head actually
        trained against.
        """
        entry = BACKBONE_REGISTRY.get(key, {})
        return cls(
            key=key, checkpoint=entry.get("checkpoint"), revision=entry.get("revision"),
            dim=dim, native_res=entry.get("native_res"), norm_mean=None, norm_std=None, bb_id=None,
        )


@dataclass(frozen=True)
class Bundle:
    """A loaded model bundle: backbone identity, feature pipeline, head,
    threshold and provenance, in memory. See the module docstring."""

    bundle_version: int
    backbone: BundleBackbone
    features: FeaturePipeline
    head_kind: str
    head_state_dict: dict
    threshold: float
    threshold_source: str
    config_hash: str | None
    metrics: dict = field(default_factory=dict)

    def build_head(self):
        """A ready-to-`eval()` head module with this bundle's weights loaded."""
        from aigc_detect.registry.heads import build_head

        head = build_head(self.head_kind, self.backbone.dim)
        head.load_state_dict(self.head_state_dict)
        head.eval()
        return head

    def to_dict(self) -> dict:
        return {
            "bundle_version": self.bundle_version,
            "backbone": self.backbone.to_dict(),
            "features": self.features.to_dict(),
            "head_kind": self.head_kind,
            "head_state_dict": self.head_state_dict,
            "threshold": float(self.threshold),
            "threshold_source": self.threshold_source,
            "config_hash": self.config_hash,
            "metrics": dict(self.metrics),
        }


def save_bundle(
    path: str | Path,
    *,
    backbone: BundleBackbone,
    features: FeaturePipeline,
    head_kind: str,
    head_state_dict: dict,
    threshold: float,
    threshold_source: str,
    config_hash: str | None = None,
    metrics: dict | None = None,
) -> Path:
    """Write a native (`bundle_version=BUNDLE_VERSION`) bundle. Returns `path`."""
    bundle = Bundle(
        bundle_version=BUNDLE_VERSION,
        backbone=backbone,
        features=features,
        head_kind=head_kind,
        head_state_dict=head_state_dict,
        threshold=float(threshold),
        threshold_source=threshold_source,
        config_hash=config_hash,
        metrics=dict(metrics or {}),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.to_dict(), path)
    return path


_LEGACY_REQUIRED_KEYS = frozenset({"state_dict", "head_kind", "backbone", "in_dim", "scaler_mean", "scaler_std"})

# Legacy metadata fields worth keeping as a metrics snapshot when present --
# whatever `train_head_on_views`/`train_head` happened to save alongside the
# state dict. Anything absent is simply omitted rather than defaulted, so a
# bundle's `metrics` never claims a number the original run never recorded.
_LEGACY_METRIC_KEYS = (
    "final_val_auc",
    "final_val_auc_robust_pooled",
    "final_val_balanced_acc",
    "epochs",
    "lr",
    "batch_size",
    "weight_decay",
    "seed",
    "train_stem",
    "train_views",
    "held_out_views",
    "balance_classes",
    "excluded_generators",
)


def load_bundle(path: str | Path) -> Bundle:
    """Load a bundle. Also accepts a legacy `models/*.pt` checkpoint, upgraded
    in memory -- see the module docstring for why and what is and is not
    recoverable that way. CPU-only, no network: `torch.load(..., map_location="cpu")`
    on a small state dict, never a backbone checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[bundle] no checkpoint/bundle at {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "bundle_version" in raw:
        return _bundle_from_dict(raw)
    return _upgrade_legacy(raw, path)


def _bundle_from_dict(raw: dict) -> Bundle:
    return Bundle(
        bundle_version=int(raw["bundle_version"]),
        backbone=BundleBackbone.from_dict(raw["backbone"]),
        features=FeaturePipeline.from_dict(raw["features"]),
        head_kind=raw["head_kind"],
        head_state_dict=raw["head_state_dict"],
        threshold=float(raw["threshold"]),
        threshold_source=raw["threshold_source"],
        config_hash=raw.get("config_hash"),
        metrics=dict(raw.get("metrics", {})),
    )


def _upgrade_legacy(raw: dict, path: Path) -> Bundle:
    missing = _LEGACY_REQUIRED_KEYS - raw.keys()
    if missing:
        raise SystemExit(
            f"[bundle] {path.name} is neither a bundle (no `bundle_version`) nor a recognisable "
            f"legacy checkpoint (missing {sorted(missing)}). Is this the right file?"
        )
    key = str(raw["backbone"])
    dim = int(raw["in_dim"])
    mean = np.asarray(raw["scaler_mean"], dtype=np.float32)
    std = np.asarray(raw["scaler_std"], dtype=np.float32)
    features = FeaturePipeline.from_spec(
        [
            {"op": "gather", "backbone": key},
            {"op": "standardize", "mean": mean.tolist(), "std": std.tolist()},
        ]
    )
    metrics = {k: raw[k] for k in _LEGACY_METRIC_KEYS if k in raw}
    return Bundle(
        bundle_version=0,
        backbone=BundleBackbone.from_registry_key(key, dim=dim),
        features=features,
        head_kind=raw["head_kind"],
        head_state_dict=raw["state_dict"],
        threshold=LEGACY_DEFAULT_THRESHOLD,
        threshold_source="legacy default 0.980",
        config_hash=None,
        metrics=metrics,
    )
