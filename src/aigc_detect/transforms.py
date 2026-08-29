"""Data augmentation / robustness-transform pipeline.

Implements exactly the transform table from the challenge brief (5.2):

    Transform         Parameters                          Real-world analog
    ----------------  -----------------------------------  ------------------------
    JPEG Compression  quality = 90, 70, 50, 30              social re-encode, chat apps
    Gaussian Blur     kernel sigma = 0.5, 1.0, 2.0           out-of-focus
    Resize            scale 0.5x / 0.25x then upscale        thumbnail generation
    Gaussian Noise    sigma = 0.02, 0.05, 0.10                low-light sensor noise
    Color Jitter      brightness/contrast/sat. +/-20%         filter apps, auto-enhance
    Center Crop       crop 80%                                profile-pic cropping

Two pipelines are exposed:

  * ``build_train_transform``     — light standard aug + a stochastic mix of the
    real-world degradations above, so the classifier learns to be robust to them.
  * ``build_eval_transform``      — deterministic resize/normalize only ("clean").
  * ``build_robustness_eval_transforms`` — one deterministic pipeline *per*
    (transform, severity) combination in the table, plus "clean", for the
    robustness evaluation summary deliverable (5.5.4): apply each in isolation
    at a fixed severity so clean-vs-transformed accuracy is directly comparable.

Resize tail (``build_backbone_transform``): all three pipelines above end in
an *aspect-preserving* resize-shortest-side + square crop, not a plain
``Resize((S, S))``. A non-aspect-preserving square resize anisotropically
stretches every non-square image, and in this project's data that stretch is
almost a perfect label proxy on its own (SID_Set's AIGC images are ~100%
exactly 1024x1024 square; its real photos are ~96% non-square) --
`scripts/audit_data.py`'s blind probe measured ~0.95-0.97 balanced accuracy
from a 16x16-grayscale vector alone, i.e. a model could "solve" real-vs-AIGC
by looking at aspect ratio and never at content. Resizing the shortest side
and center/random-cropping to a square removes that shortcut while keeping
every image at the required IMAGE_SIZE x IMAGE_SIZE tensor shape.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field

import torch
from PIL import Image
from torchvision.transforms import v2

from aigc_detect.config import IMAGE_SIZE, NORM_MEAN, NORM_STD

# ---------------------------------------------------------------------------
# Parameter table (5.2), kept as plain constants so eval code can enumerate it.
# ---------------------------------------------------------------------------

JPEG_QUALITIES = (90, 70, 50, 30)
BLUR_SIGMAS = (0.5, 1.0, 2.0)
RESIZE_SCALES = (0.5, 0.25)
NOISE_SIGMAS = (0.02, 0.05, 0.10)
COLOR_JITTER_STRENGTH = 0.20  # +/-20% brightness/contrast/saturation
CENTER_CROP_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Individual transforms. Each is a plain callable operating on a PIL Image
# (JPEG, blur, resize, crop, color jitter) or a float tensor in [0, 1] (noise),
# so they compose with torchvision.transforms.v2 in either domain.
# ---------------------------------------------------------------------------


class JPEGCompression:
    """Round-trips the image through a JPEG encoder at one of the given qualities."""

    def __init__(self, qualities: tuple[int, ...] = JPEG_QUALITIES):
        self.qualities = qualities

    def __call__(self, img: Image.Image, quality: int | None = None) -> Image.Image:
        q = quality if quality is not None else random.choice(self.qualities)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlurLevels:
    """Gaussian blur at one of a discrete set of sigmas (kernel size derived from sigma)."""

    def __init__(self, sigmas: tuple[float, ...] = BLUR_SIGMAS):
        self.sigmas = sigmas

    @staticmethod
    def _kernel_size(sigma: float) -> int:
        radius = max(1, math.ceil(3 * sigma))
        return 2 * radius + 1

    def __call__(self, img: Image.Image, sigma: float | None = None) -> Image.Image:
        s = sigma if sigma is not None else random.choice(self.sigmas)
        k = self._kernel_size(s)
        return v2.functional.gaussian_blur(img, kernel_size=[k, k], sigma=[s, s])


class ResizeRoundTrip:
    """Downscale then upscale back — simulates thumbnail generation / re-upload."""

    def __init__(self, scales: tuple[float, ...] = RESIZE_SCALES):
        self.scales = scales

    def __call__(self, img: Image.Image, scale: float | None = None) -> Image.Image:
        s = scale if scale is not None else random.choice(self.scales)
        w, h = img.size
        small_w, small_h = max(1, round(w * s)), max(1, round(h * s))
        small = v2.functional.resize(img, [small_h, small_w], antialias=True)
        return v2.functional.resize(small, [h, w], antialias=True)


class GaussianNoiseLevels:
    """Additive Gaussian noise on a [0, 1]-scaled tensor, clipped back to [0, 1]."""

    def __init__(self, sigmas: tuple[float, ...] = NOISE_SIGMAS):
        self.sigmas = sigmas

    def __call__(self, img: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
        s = sigma if sigma is not None else random.choice(self.sigmas)
        noise = torch.randn_like(img) * s
        return (img + noise).clamp(0.0, 1.0)


class CenterCropFraction:
    """Center-crops to a fraction of the original H/W (no resize back — the
    pipeline's final Resize step renormalizes size)."""

    def __init__(self, fraction: float = CENTER_CROP_FRACTION):
        self.fraction = fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return v2.functional.center_crop(img, [round(h * self.fraction), round(w * self.fraction)])


def make_color_jitter(strength: float = COLOR_JITTER_STRENGTH) -> v2.ColorJitter:
    return v2.ColorJitter(brightness=strength, contrast=strength, saturation=strength)


# ---------------------------------------------------------------------------
# Stochastic compound augmentation used at TRAIN time.
# ---------------------------------------------------------------------------


@dataclass
class RobustnessAugment:
    """With probability ``p_any``, applies a random subset (1..max_ops) of the
    real-world degradations above, in random order, each with a randomly
    sampled severity from its table. Otherwise returns the image unchanged.

    This is deliberately compositional (e.g. resize + JPEG can co-occur, as
    they would for a re-uploaded, re-compressed thumbnail) so the model sees
    realistic combinations rather than only single isolated corruptions.
    """

    p_any: float = 0.8
    max_ops: int = 2
    jpeg: JPEGCompression = field(default_factory=JPEGCompression)
    blur: GaussianBlurLevels = field(default_factory=GaussianBlurLevels)
    resize: ResizeRoundTrip = field(default_factory=ResizeRoundTrip)
    color_jitter: v2.ColorJitter = field(default_factory=make_color_jitter)
    center_crop: CenterCropFraction = field(default_factory=CenterCropFraction)
    noise: GaussianNoiseLevels = field(default_factory=GaussianNoiseLevels)

    def _pil_ops(self):
        return [self.jpeg, self.blur, self.resize, self.color_jitter, self.center_crop]

    def __call__(self, img: Image.Image):
        if random.random() < self.p_any:
            ops = self._pil_ops()
            random.shuffle(ops)
            n = random.randint(1, self.max_ops)
            for op in ops[:n]:
                img = op(img)
        return img

    def apply_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Noise operates on a float tensor, so it's applied after ToImage/ToDtype
        inside the composed pipeline (see build_train_transform)."""
        if random.random() < self.p_any * 0.5:
            tensor = self.noise(tensor)
        return tensor


class FixedSeverity:
    """Binds one severity to one of the transform callables above.

    Exists because ``v2.Lambda(lambda img: op(img, sigma=s))`` is NOT picklable,
    and Windows' DataLoader uses the spawn start method, which pickles the
    dataset (and therefore its transforms) to send to each worker. With lambdas
    the robustness grid dies with "Can't pickle local object" the moment
    num_workers > 0 -- i.e. exactly when it matters, since the per-view embedder
    is decode-bound. A module-level class with plain attributes pickles fine.
    """

    def __init__(self, op, **kwargs):
        self.op = op
        self.kwargs = kwargs

    def __call__(self, x):
        return self.op(x, **self.kwargs)


class _MaybeNoise:
    """Wraps GaussianNoiseLevels with an application probability, for use inside
    a v2.Compose tensor stage."""

    def __init__(self, noise: GaussianNoiseLevels, p: float):
        self.noise = noise
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.noise(tensor)
        return tensor


# ---------------------------------------------------------------------------
# Public pipeline builders
# ---------------------------------------------------------------------------


def build_backbone_transform(image_size: int = IMAGE_SIZE) -> list:
    """Aspect-preserving resize of the shortest side followed by a square
    center crop. Replaces the non-aspect-preserving Resize((S, S)) tail --
    see the module docstring for why that tail was a label shortcut.

    Returns a plain list of v2 transforms (not a Compose) so callers can
    splice it into a larger pipeline alongside PIL-domain augmentation steps.
    """
    return [
        v2.Resize(image_size, antialias=True),  # int -> resizes shortest side, preserves aspect
        v2.CenterCrop(image_size),
    ]


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    robustness_p: float = 0.8,
    norm_mean: tuple[float, ...] = NORM_MEAN,
    norm_std: tuple[float, ...] = NORM_STD,
) -> v2.Compose:
    """Training pipeline: standard light aug + stochastic real-world degradations.

    ``norm_mean``/``norm_std`` default to config's ImageNet stats but MUST be
    overridden with the backbone's own stats when feeding a frozen VFM -- see
    build_robustness_eval_transforms' docstring.
    """
    aug = RobustnessAugment(p_any=robustness_p)
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            aug,  # PIL-domain: jpeg/blur/resize/color-jitter/center-crop (maybe none)
            v2.Resize(image_size, antialias=True),  # shortest side, aspect-preserving
            v2.RandomCrop(image_size, pad_if_needed=True),  # crop diversity at train time
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            _MaybeNoise(aug.noise, p=robustness_p * 0.5),  # [0,1] domain -- BEFORE Normalize
            v2.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def build_eval_transform(
    image_size: int = IMAGE_SIZE,
    norm_mean: tuple[float, ...] = NORM_MEAN,
    norm_std: tuple[float, ...] = NORM_STD,
) -> v2.Compose:
    """Deterministic "clean" pipeline: aspect-preserving resize + center crop +
    normalize. Used for local validation accuracy and as the baseline row of
    the robustness summary."""
    return v2.Compose(
        [
            *build_backbone_transform(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


def build_robustness_eval_transforms(
    image_size: int = IMAGE_SIZE,
    norm_mean: tuple[float, ...] = NORM_MEAN,
    norm_std: tuple[float, ...] = NORM_STD,
) -> dict[str, v2.Compose]:
    """One deterministic pipeline per (transform, severity) in the table, plus
    "clean". Keys are stable, human-readable names, e.g. "jpeg_q50",
    "blur_sigma1.0", "resize_0.25x", "noise_sigma0.05", "color_jitter",
    "center_crop_80". Use these to build separate eval DataLoaders and report
    clean-vs-transformed accuracy per 5.5.4 (Robustness Evaluation Summary).

    Two things here are easy to get wrong and are silent when wrong:

    1. **Noise must be added in the [0, 1] domain, before Normalize.** The
       brief's sigmas (0.02/0.05/0.10) are fractions of the pixel range, and
       ``GaussianNoiseLevels`` clamps to [0, 1] as a valid-pixel guard. Run
       after Normalize, that clamp instead floors every value below the channel
       mean and saturates everything above mean+std -- destroying the image
       rather than perturbing it, at *any* sigma. Keep noise between ToDtype
       and Normalize.
    2. **norm_mean/norm_std must match the backbone.** These default to
       config's ImageNet stats, but the frozen VFMs use their own (PE-Core is
       0.5/0.5, MetaCLIP2 uses OpenAI-CLIP stats) -- ``embed.py`` normalizes
       with ``module.norm_mean``/``module.norm_std``. Pass the backbone's stats
       here too, or every view is evaluated under normalization the model was
       never trained with and the "clean" view silently disagrees with the
       existing cached clean embeddings.
    """
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]  # -> float in [0, 1]
    normalize = [v2.Normalize(mean=norm_mean, std=norm_std)]
    base_post = [*to_tensor, *normalize]
    resize_step = v2.Compose(build_backbone_transform(image_size))  # shortest-side resize + center crop

    pipelines: dict[str, v2.Compose] = {
        "clean": v2.Compose([resize_step, *base_post]),
    }

    # FixedSeverity rather than v2.Lambda(lambda ...) throughout: these pipelines
    # get pickled to DataLoader workers under Windows' spawn start method.
    jpeg = JPEGCompression()
    for q in JPEG_QUALITIES:
        pipelines[f"jpeg_q{q}"] = v2.Compose(
            [FixedSeverity(jpeg, quality=q), resize_step, *base_post]
        )

    blur = GaussianBlurLevels()
    for s in BLUR_SIGMAS:
        pipelines[f"blur_sigma{s}"] = v2.Compose(
            [FixedSeverity(blur, sigma=s), resize_step, *base_post]
        )

    resize_rt = ResizeRoundTrip()
    for s in RESIZE_SCALES:
        pipelines[f"resize_{s}x"] = v2.Compose(
            [FixedSeverity(resize_rt, scale=s), resize_step, *base_post]
        )

    # Noise is the one tensor-domain transform: applied to the [0, 1] tensor and
    # normalized afterwards (see note 1 above -- the ordering is load-bearing).
    noise = GaussianNoiseLevels()
    for s in NOISE_SIGMAS:
        pipelines[f"noise_sigma{s}"] = v2.Compose(
            [resize_step, *to_tensor, FixedSeverity(noise, sigma=s), *normalize]
        )

    pipelines["color_jitter"] = v2.Compose([make_color_jitter(), resize_step, *base_post])
    pipelines["center_crop_80"] = v2.Compose([CenterCropFraction(), resize_step, *base_post])

    return pipelines
