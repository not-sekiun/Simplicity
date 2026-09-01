"""Generator taxonomy: which architecture family each source belongs to.

This split is load-bearing for scoping. The competition is expected to use
DIFFUSION generators, so a collapse confined to the GAN families is out of
scope and must not be allowed to drag the headline down or drive a backbone
choice. Reporting one pooled OOD number would hide exactly that distinction.

Orthogonal to seen/unseen: BigGAN is a GAN that IS in the training pool, while
SD14/SDXL/DALLE2 are diffusion models that are NOT. Always report both axes --
family answers "is this in scope", seen/unseen answers "is this
generalization".

Kept as Python rather than YAML deliberately. The entries are consumed as a
typed mapping by the grid and the error analysis, and two of the comments below
are findings that cost real measurement time to establish; they belong beside
the values they explain.
"""

from __future__ import annotations

GENERATOR_FAMILY: dict[str, str] = {
    # Diffusion / autoregressive-diffusion -- the in-scope families.
    "ADM": "diffusion",
    "DALLE2": "diffusion",
    "GLIDE": "diffusion",
    "Midjourney": "diffusion",
    "SD14": "diffusion",
    "SD15": "diffusion",
    "SDXL": "diffusion",
    "VQDM": "diffusion",
    "Wukong": "diffusion",
    # Modern diffusion, added 2026-08-30 to close the era gap.
    "MidjourneyV6": "diffusion",
    "NanoBanana": "diffusion",
    "DALLE3": "diffusion",
    # GANs -- treated as out of scope for the competition, reported separately.
    "BigGAN": "gan",
    "CycleGAN": "gan",
    "GauGAN": "gan",
    "ProGAN": "gan",
    "StarGAN": "gan",
    "StyleGAN": "gan",
    "StyleGAN2": "gan",
    # NOT a real source, despite the name and despite the upstream dataset card
    # calling it "Real human face sourced from the WhichFaceIsReal dataset".
    # whichfaceisreal.com shows an FFHQ photo BESIDE a StyleGAN fake; this HF
    # port ships only the fake half. Upstream's own label column agrees --
    # every sampled row is label=1 with names ['real','fake'] -- and the pixels
    # agree too (incoherent backgrounds, melted hair, blob artefacts). The card
    # prose is the only signal saying "real", and it loses 2-to-1.
    #
    # We had this as "real", which let the OOD builder's folder-name label
    # inference rewrite 250 StyleGAN faces to label=0. That single mapping
    # produced the entire "100% portrait FPR" finding: the model scored them
    # >0.997 because they ARE fake. Correcting it moved ood clean AUC
    # 0.9670 -> 0.9971.
    "WhichFaceIsReal": "gan",
    # Real-image sources. Kept as distinct entries rather than one "Real" label
    # because pooling them is exactly how a 100% failure on one real
    # subpopulation stayed invisible behind a healthy aggregate.
    "Real": "real",                        # ImageNet, via Tiny-GenImage/GenImage
    "Real_OpenImages": "real",             # SID_Set's real half
    "Real_Unsplash": "real",               # curated photography
    "Real_Pexels": "real",                 # curated photography
    "Real_WildRF_train": "real",           # WildRF social-media reals (training)
    "Real_WildRF_reddit": "real",          # WildRF eval tier, per platform
    "Real_WildRF_twitter": "real",
    "Real_WildRF_facebook": "real",
    # WildRF's AI half: real-world social-media fakes, unknown provenance.
    "WildRF_reddit": "social",
    "WildRF_twitter": "social",
    "WildRF_facebook": "social",
}

# Generators present in data/raw/ (the training pool). The seen/unseen axis:
# anything scored that is NOT in here is a generalization measurement.
TRAIN_GENERATORS = frozenset({"Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"})

__all__ = ["GENERATOR_FAMILY", "TRAIN_GENERATORS"]
