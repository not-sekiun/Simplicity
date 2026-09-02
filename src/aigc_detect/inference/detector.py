"""The `Detector` protocol: what a caller needs from "a model", nothing more.

WHAT THIS REPLACES. Before tier 8b, `demo/server.py` carried a `Model` class
that opened a checkpoint dict directly, read `scaler_mean`/`scaler_std` off
it, and applied `(x - mean) / std` by hand -- a SECOND, independent
implementation of the exact standardization step `train.features.FeaturePipeline`
already owns (`train.features`'s own module docstring calls this out by
name, though it remembers the class as `_HeadRunner`; whatever it was called,
it was the same hand-rolled subtract-divide, drifted from the pipeline it
duplicated the moment either one changed without the other). `inference.predict`
had a THIRD copy inline in `run_inference`, pre-tier-7. Tier 7 collapsed
predict.py's copy onto `Bundle.features` (`FeaturePipeline`). This module
collapses the server's the same way.

WHY A PROTOCOL, NOT JUST A CLASS. The frozen-probe-plus-linear-head recipe
("Simplicity Prevails", arXiv:2602.01738) is this project's current answer,
not its only possible one -- an ensemble of two backbones, or a fine-tuned
head, are both architectures a `FeaturePipeline` with a multi-`gather` spec
already anticipates (see its module docstring). `demo/server.py` and any
future CLI must not need to change to run either: they hold a `Detector`,
call `.score(images)` and `.describe()`, and never import
`registry.backbones` or `registry.heads` themselves. `FrozenProbeDetector`
below is the one shipping implementation of that protocol today.

WHAT `describe()` IS FOR. `/health` in `demo/server.py` renders it directly
for the extension's popup (backbone name, threshold) and for a human
checking what's loaded. It is deliberately NOT what `score()` uses
internally -- `score()` reads `self.bundle`/`self.module` directly, so
`describe()` can add or rename display fields without touching scoring.

PREPROCESSING PARITY WITH `inference.predict`. `FrozenProbeDetector.load`
reproduces `run_inference`'s four steps verbatim (bundle's own backbone key,
that backbone's own native_res/norm stats via `load_backbone`,
aspect-preserving resize + center crop via `build_backbone_transform`, the
bundle's own `FeaturePipeline` before the head, raw sigmoid) rather than
inventing a shorter version -- `tests/test_parity.py` is what holds the two
to that promise now, instead of a docstring asking the reader to trust it.
The one piece of arithmetic this module does NOT reimplement is
`FeaturePipeline`'s gather/l2norm/standardize -- that stays the single
implementation `bundle.features.transform` owns, imported and called, never
copied.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from PIL import Image
from torchvision.transforms import v2

from aigc_detect.data.transforms import build_backbone_transform
from aigc_detect.inference.bundle import Bundle
from aigc_detect.registry.backbones import load_backbone


@runtime_checkable
class Detector(Protocol):
    """What the HTTP layer (or any other caller) needs from "a model".

    Three operations, deliberately not more: `load` is the only place a
    concrete backend may touch `registry.backbones`/`registry.heads` or a
    GPU; `score` takes decoded images and returns `P(AIGC)` per image, in the
    same order; `describe` is display-only metadata for a health/status
    endpoint. Nothing here mentions checkpoints, backbones, or tensors by
    name -- `demo/server.py` (or `apps/server/app.py`) is written against
    exactly this surface and nothing wider, so a different backend (an
    ensemble, a fine-tuned model) drops in by satisfying this Protocol alone.
    """

    @classmethod
    def load(cls, bundle: Bundle) -> Detector:
        """Build a ready-to-score detector from an already-loaded `Bundle`.

        Takes a `Bundle`, not a path -- `inference.bundle.load_bundle` is the
        one place that knows how to turn a file on disk (native bundle or
        legacy checkpoint) into a `Bundle`; a `Detector` backend should never
        need to re-derive that.
        """
        ...

    def score(self, images: list[Image.Image]) -> list[float]:
        """`P(AIGC)` for each image, same order, empty list for empty input."""
        ...

    def describe(self) -> dict:
        """Display metadata for a health/status endpoint -- see module docstring."""
        ...


class FrozenProbeDetector:
    """The shipping backend: a frozen VFM backbone + linear/MLP probe head,
    exactly the architecture `Bundle`/`FeaturePipeline` were built to carry.

    Constructed only via `load` -- see that classmethod for the four
    preprocessing steps this reproduces from `inference.predict.run_inference`.
    """

    def __init__(
        self,
        *,
        bundle: Bundle,
        module: torch.nn.Module,
        head: torch.nn.Module,
        transform: v2.Compose,
        device: torch.device,
        native_res: int,
    ) -> None:
        self.bundle = bundle
        self.module = module
        self.head = head
        self.transform = transform
        self.device = device
        self.native_res = native_res

    @classmethod
    def load(cls, bundle: Bundle) -> FrozenProbeDetector:
        backbone_key = bundle.backbone.key
        module, pooled_dim, native_res = load_backbone(backbone_key)
        if pooled_dim != bundle.backbone.dim:
            raise SystemExit(
                f"[detector] backbone '{backbone_key}' pooled_dim={pooled_dim} does not match "
                f"the bundle's recorded dim={bundle.backbone.dim} -- checkpoint/backbone mismatch."
            )
        device = next(module.parameters()).device
        head = bundle.build_head().to(device)

        # Identical to embed.py/predict.py: aspect-preserving resize + center
        # crop at the backbone's OWN native_res, then the backbone's OWN norm
        # stats (from `load_backbone`, never config.IMAGE_SIZE/NORM_MEAN --
        # see inference.predict's module docstring point 2).
        transform = v2.Compose(
            [
                *build_backbone_transform(native_res),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=module.norm_mean, std=module.norm_std),
            ]
        )
        return cls(bundle=bundle, module=module, head=head, transform=transform, device=device, native_res=native_res)

    @torch.no_grad()
    def score(self, images: list[Image.Image]) -> list[float]:
        if not images:
            return []
        batch = torch.stack([self.transform(img) for img in images]).to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_amp):
            feats = self.module(batch)
        feats_np = feats.float().cpu().numpy()
        # Bundle's own FeaturePipeline, not batch statistics -- see
        # inference.predict's module docstring point 4. The whole point of
        # this class is that this line, and nothing else, is where
        # "embedding becomes head input" happens.
        x = torch.from_numpy(self.bundle.features.transform({self.bundle.backbone.key: feats_np})).to(self.device)
        logits = self.head(x).squeeze(-1)
        # Raw sigmoid, deliberately uncalibrated -- see inference.predict's
        # module docstring for why (AUC depends only on score order, and
        # calibration was measured to cost, not gain, on this project's
        # metric). Not repeated here at length; that module is the
        # authoritative explanation and this class must keep agreeing with it.
        probs = torch.sigmoid(logits).cpu().numpy()
        return [float(p) for p in probs]

    def describe(self) -> dict:
        """Rendered by `/health` for the extension popup and any human
        checking what's loaded. `threshold`/`backbone`/`backbone_revision`/
        `head_kind` are the fields the extension (`detector-client.js`,
        `resolveThreshold`) and this project's own docs promise are present;
        everything else is provenance, safe to grow without breaking either.
        """
        b = self.bundle
        return {
            "backbone": b.backbone.key,
            "backbone_revision": b.backbone.revision,
            "head_kind": b.head_kind,
            "threshold": b.threshold,
            "threshold_source": b.threshold_source,
            "bundle_version": b.bundle_version,
        }
