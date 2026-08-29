"""Registry of frozen vision-foundation-model backbones for the "Simplicity
Prevails" recipe (arXiv:2602.01738): train a classifier head on the pooled
output of a frozen backbone, nothing else. All four checkpoints below were
verified ungated and public on Hugging Face before being hardcoded here --
don't add a new entry without doing the same check.

    key           checkpoint                                          loader        pooled dim  native res
    ------------  --------------------------------------------------  ------------  ----------  ----------
    metaclip2-h   facebook/metaclip-2-worldwide-huge-quickgelu         transformers  1280        224
    dinov3-l      timm/vit_large_patch16_dinov3.lvd1689m                timm          1024        256
    pe-core-l     timm/vit_pe_core_large_patch14_336.fb                 timm          1024        336
    dinov2-g      facebook/dinov2-giant                                 transformers  1536        518

``metaclip2-h`` is the special case: the checkpoint is the full CLIP-style
model (image + text towers, 1.86B params total), but the head only needs the
*vision* tower (~632M params). We load the vision-only model class so the
text tower's weights are never even instantiated. If the installed
``transformers`` is too old to know about MetaCLIP2, we fall back to the
timm mirror ``vit_huge_patch14_clip_224.metaclip2_worldwide`` (identical
weights, same 1280-d pooled output) and print a loud warning saying so.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Hard competition rule: the vision tower actually used for inference must be
# under 2B parameters. Checked and printed every time a backbone is loaded.
MAX_BACKBONE_PARAMS = 2_000_000_000

BACKBONE_REGISTRY: dict[str, dict] = {
    "metaclip2-h": {
        "checkpoint": "facebook/metaclip-2-worldwide-huge-quickgelu",
        "loader": "transformers",
        "pooled_dim": 1280,
        "native_res": 224,
        "fallback_timm": "vit_huge_patch14_clip_224.metaclip2_worldwide",
    },
    "dinov3-l": {
        "checkpoint": "timm/vit_large_patch16_dinov3.lvd1689m",
        "loader": "timm",
        "pooled_dim": 1024,
        "native_res": 256,
    },
    "pe-core-l": {
        "checkpoint": "timm/vit_pe_core_large_patch14_336.fb",
        "loader": "timm",
        "pooled_dim": 1024,
        "native_res": 336,
    },
    # The variant the paper actually benchmarked as its robustness champion
    # (MetaCLIP2 blur sigma=2.0: 0.932, improving, vs PE-CLIP's 0.778 collapse).
    # We raced `metaclip2-h` instead because Giant's vision tower was ESTIMATED
    # near the 2B cap -- this entry exists to replace that estimate with a
    # measurement. Two things make it worth the check despite the cost:
    #   1. It is the ONLY available candidate running ABOVE pe-core-l's 336px,
    #      so it is the one test of the resolution hypothesis (NARRATIVE Run 7)
    #      in the favourable direction. If resolution drives degraded-input
    #      robustness, Giant should win; if it loses anyway, the hypothesis is
    #      wrong and architecture/pretraining matter more.
    #   2. We are using 316M of a 2B budget.
    # NOTE: cc-by-nc-4.0 (non-commercial), unlike the other entries.
    "metaclip2-giant": {
        "checkpoint": "facebook/metaclip-2-worldwide-giant-378",
        "loader": "transformers",
        "pooled_dim": 1664,
        "native_res": 378,
    },
    "dinov2-g": {
        "checkpoint": "facebook/dinov2-giant",
        "loader": "transformers",
        "pooled_dim": 1536,
        "native_res": 518,
    },
}


def list_backbones() -> list[str]:
    return list(BACKBONE_REGISTRY.keys())


class _PooledVisionWrapper(nn.Module):
    """Wraps a transformers vision-only model so ``forward(pixel_values)``
    returns just the pooled embedding, matching timm's num_classes=0
    contract (a plain (B, dim) tensor, no HF output dataclass)."""

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]  # CLS token fallback
        return pooled


def _strip_timm_org_prefix(checkpoint: str) -> str:
    """The registry's ``timm/...`` checkpoint strings are the actual HF hub
    repo ids (for documentation); timm.create_model wants the bare
    architecture.tag name and resolves the ``timm/`` org itself."""
    return checkpoint[len("timm/"):] if checkpoint.startswith("timm/") else checkpoint


def _load_timm_backbone(checkpoint: str):
    import timm
    from timm.data import resolve_model_data_config

    model_name = _strip_timm_org_prefix(checkpoint)
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    data_cfg = resolve_model_data_config(model)
    mean, std = tuple(data_cfg["mean"]), tuple(data_cfg["std"])
    return model, mean, std, "timm.data.resolve_model_data_config", checkpoint


def _load_transformers_vision_tower(checkpoint: str, model_cls):
    from transformers import AutoImageProcessor

    model = model_cls.from_pretrained(checkpoint)
    try:
        proc = AutoImageProcessor.from_pretrained(checkpoint)
        mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        norm_source = "transformers AutoImageProcessor"
    except Exception as exc:
        from aigc_detect.config import NORM_MEAN, NORM_STD

        print(
            f"[backbones] WARNING: no image processor found for '{checkpoint}' "
            f"({type(exc).__name__}: {exc}); falling back to config.NORM_MEAN/STD"
        )
        mean, std = NORM_MEAN, NORM_STD
        norm_source = "config.NORM_MEAN/STD fallback"
    return model, mean, std, norm_source, checkpoint


def load_backbone(key: str):
    """Load a frozen backbone by registry key.

    Returns (module, pooled_dim, native_res). ``module`` is in eval() mode,
    every parameter has requires_grad=False, and it lives on CUDA if
    available. Extra bookkeeping attributes are attached to the returned
    module for callers (embed.py) that need them without changing this
    function's return signature: ``norm_mean``, ``norm_std``, ``norm_source``,
    ``checkpoint_used``, ``used_fallback``.
    """
    if key not in BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone '{key}'. Available: {list_backbones()}")
    entry = BACKBONE_REGISTRY[key]
    used_fallback = False

    if entry["loader"] == "timm":
        module, mean, std, norm_source, used_checkpoint = _load_timm_backbone(entry["checkpoint"])
    elif key.startswith("metaclip2"):
        # Family dispatch, not per-key: every MetaCLIP2 checkpoint loads through
        # the same vision-tower class. Keying on the exact name meant adding a
        # second MetaCLIP2 variant raised "No loader implemented", which reads
        # like an unsupported architecture rather than a missing elif.
        try:
            from transformers import MetaClip2VisionModel

            vision_model, mean, std, norm_source, used_checkpoint = _load_transformers_vision_tower(
                entry["checkpoint"], MetaClip2VisionModel
            )
            module = _PooledVisionWrapper(vision_model)
        except Exception as exc:
            fallback = entry.get("fallback_timm")
            if not fallback:
                raise
            print(
                f"[backbones] WARNING: transformers could not load the MetaCLIP2 vision "
                f"tower from '{entry['checkpoint']}' ({type(exc).__name__}: {exc}); "
                f"falling back to timm mirror '{fallback}'"
            )
            module, mean, std, norm_source, used_checkpoint = _load_timm_backbone(fallback)
            used_fallback = True
    elif key == "dinov2-g":
        from transformers import Dinov2Model

        vision_model, mean, std, norm_source, used_checkpoint = _load_transformers_vision_tower(
            entry["checkpoint"], Dinov2Model
        )
        module = _PooledVisionWrapper(vision_model)
    else:
        raise ValueError(f"No loader implemented for backbone '{key}'")

    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)

    n_params = sum(p.numel() for p in module.parameters())
    fallback_note = " (FALLBACK checkpoint used)" if used_fallback else ""
    print(f"[backbones] {key}: checkpoint={used_checkpoint} vision-tower params={n_params:,}{fallback_note}")
    assert n_params < MAX_BACKBONE_PARAMS, (
        f"[backbones] {key} vision tower has {n_params:,} params, which exceeds the "
        f"{MAX_BACKBONE_PARAMS:,} competition limit"
    )

    module.norm_mean = mean
    module.norm_std = std
    module.norm_source = norm_source
    module.checkpoint_used = used_checkpoint
    module.used_fallback = used_fallback
    return module, entry["pooled_dim"], entry["native_res"]
