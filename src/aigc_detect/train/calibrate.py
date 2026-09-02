"""In-loop threshold calibration: the protocol from `scripts/derive_threshold.py`,
promoted into the library and called automatically at the end of training.

WHY IT MOVED. The decision threshold used to be a separate manual step: train a
head, remember to run `scripts/derive_threshold.py --head <it>`, remember to
copy the printed number into `inference.predict.DECISION_THRESHOLD` by hand.
Every step of that chain is a place to forget, and the project's own history
shows it happening -- FINDINGS 2j and 2k record two reconstructions of "the
protocol" from prose that disagreed with each other in the third decimal
(trainext scored .0283/.9686 the first time and .0280/.9663 the second) because
the split-half assignment was never written down as code, only described. This
module IS that code, with the split pinned, and `train.experiment`'s probe
trainer calls it as the last step of every run -- see that module -- so a
freshly trained head's bundle always carries a threshold nobody had to
remember to derive.

THE SPLIT IS LOAD-BEARING: `numpy.random.RandomState(0).permutation(n)`, first
half = A. That exact call is the only thing that reproduces FINDINGS 2j's
originally recorded table:

    head        recorded (2j)              this module
    photoreal   0.920 / .0408 / .9721      0.920 / .0408 / .9721
    trainext    0.940 / .0283 / .9686      0.940 / .0283 / .9686

`verify_recorded_table` re-runs exactly that check, which makes it a regression
test on the PROTOCOL (does the split still reproduce recorded numbers) rather
than a test of any particular head -- see `tests/test_bundle.py`.

THE PROTOCOL, in full (unchanged from `scripts/derive_threshold.py`, which this
duplicates rather than imports from, because `scripts/` is being retired tier
by tier and a library module must not depend on it):

  1. WildRF test (2,503 real Reddit/X/Facebook photographs and real-world AI) --
     a corpus no head trains on, and the hardest REAL images available here.
  2. Pooled over clean + the CDN-like views a browser extension actually sees:
     jpeg_q70, jpeg_q90, resize_0.5x, chain_light.
  3. Split BY IMAGE, not by row, so no image contributes to both halves through
     different views.
  4. Threshold swept 0.50..0.999 in 0.005 steps; picked by F1 on half A.
  5. FPR and TPR reported on half B, which the choice never saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from aigc_detect.config import EMBEDDINGS_DIR, ROOT_DIR
from aigc_detect.registry.heads import build_head
from aigc_detect.train.features import FeaturePipeline

# Kept in sync by hand with `scripts/export_eval_stats.CDN_VIEWS` and the
# comment on `inference.bundle.LEGACY_DEFAULT_THRESHOLD`.
CDN_VIEWS = ("clean", "jpeg_q70", "jpeg_q90", "resize_0.5x", "chain_light")
WILDRF_STEM = "wildrf_test"
SPLIT_SEED = 0
GRID = np.round(np.arange(0.50, 0.9991, 0.005), 4)

# (head file, threshold, held-out FPR, held-out TPR), exactly as FINDINGS 2j
# recorded them. `verify_recorded_table` asserts this module still reproduces
# every row.
RECORDED_TABLE = (
    ("pe-core-l__linear__photoreal.pt", 0.920, 0.0408, 0.9721),
    ("pe-core-l__linear__trainext.pt", 0.940, 0.0283, 0.9686),
)


def half_a_mask(n_images: int) -> np.ndarray:
    """The pinned split. Seeded on the IMAGE index, so every view of an image
    lands on the same side -- splitting rows instead would let half A and half
    B share images through different degradations, and the reported FPR would
    then be measured partly on images the threshold was chosen on."""
    rng = np.random.RandomState(SPLIT_SEED)
    mask = np.zeros(n_images, dtype=bool)
    mask[rng.permutation(n_images)[: n_images // 2]] = True
    return mask


def _f1(probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    pred = probs >= t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def pooled_wildrf_scores(
    head, pipeline: FeaturePipeline, embeddings_dir: Path = EMBEDDINGS_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(AIGC), labels and IMAGE INDEX for every (image, CDN view) pair.

    `image_idx` is the position within one view's cache, which is the same
    position across every view because every view cache is a row-for-row
    projection of the same manifest -- see `embed.views`. It is what
    `half_a_mask` splits on.

    THE PIPELINE IS APPLIED, NOT RE-IMPLEMENTED. This function used to take
    `(mean, std)` and compute `(x - mean) / std` inline, which made it a THIRD
    implementation of "raw pooled embedding -> head input" -- next to
    `train.probe` and `demo/server.py`'s `Model`, the exact duplication
    `train.features` exists to end. It also quietly constrained what a trained
    model was allowed to be: a config declaring `l2norm` before `standardize`,
    or gathering two backbones, would have been calibrated against
    preprocessing its head never saw, so `train.experiment` had to refuse any
    pipeline that did not end in a bare `standardize`. Taking the fitted
    pipeline instead removes the duplication and the restriction together --
    calibration now runs exactly the arithmetic inference will.
    """
    backbones = pipeline.backbones
    probs, labels, image_idx = [], [], []
    for view in CDN_VIEWS:
        arrays, view_labels = {}, None
        for key in backbones:
            cache = embeddings_dir / f"{key}__{WILDRF_STEM}__{view}.npz"
            if not cache.exists():
                raise SystemExit(
                    f"[calibrate] missing cache {cache.name}. Run:\n"
                    f"  uv run aigc embed-views --backbone {key} --manifest wildrf-test"
                )
            data = np.load(cache)
            arrays[key] = data["embeddings"]
            view_labels = data["labels"]
        x = pipeline.transform(arrays)
        with torch.no_grad():
            p = torch.sigmoid(head(torch.from_numpy(x.astype(np.float32))).squeeze(-1)).numpy()
        probs.append(p)
        labels.append(view_labels)
        image_idx.append(np.arange(len(p)))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(image_idx)


def calibrate(probs: np.ndarray, labels: np.ndarray, image_idx: np.ndarray) -> dict:
    """The core protocol: split by image, F1-optimal threshold on half A,
    report FPR/TPR on half B. Pure function of scores -- no head, no cache, no
    filesystem -- so it is directly unit-testable against synthetic scores."""
    in_a = half_a_mask(int(image_idx.max()) + 1)[image_idx]

    pa, ya = probs[in_a], labels[in_a]
    # First argmax, i.e. the LOWEST equally-optimal threshold on the grid. Ties
    # do occur; picking the lowest is the conservative side for recall.
    threshold = float(GRID[int(np.argmax([_f1(pa, ya, t) for t in GRID]))])

    pb, yb = probs[~in_a], labels[~in_a]
    return {
        "threshold": threshold,
        "fpr": float((pb[yb == 0] >= threshold).mean()),
        "tpr": float((pb[yb == 1] >= threshold).mean()),
        "n_real_b": int((yb == 0).sum()),
        "n_aigc_b": int((yb == 1).sum()),
    }


def calibrate_threshold(head, pipeline: FeaturePipeline, *, embeddings_dir: Path = EMBEDDINGS_DIR) -> dict:
    """Run the full protocol against a trained (head, fitted pipeline) pair.

    Returns `calibrate`'s dict plus a `source` string describing the protocol,
    suitable for a bundle's `threshold_source` field -- see
    `train.experiment`'s probe trainer, which is the one caller meant to run
    this automatically, at the end of every training run.

    The pipeline must already be FIT: a threshold derived through an unfit
    `standardize` would be a threshold for a model that does not exist yet.
    `FeaturePipeline.transform` raises on that rather than silently skipping
    the op, so this needs no check of its own.
    """
    probs, labels, image_idx = pooled_wildrf_scores(head, pipeline, embeddings_dir)
    result = calibrate(probs, labels, image_idx)
    result["source"] = (
        f"derived: WildRF({WILDRF_STEM}) pooled over {'+'.join(CDN_VIEWS)}, "
        f"RandomState({SPLIT_SEED}).permutation split by image, F1-optimal threshold on half A, "
        f"reported on half B"
    )
    return result


def legacy_pipeline(backbone_key: str, mean: np.ndarray, std: np.ndarray) -> FeaturePipeline:
    """The pipeline a pre-bundle checkpoint implies: gather its one backbone,
    then apply the scaler it stored. Identical in construction to
    `inference.bundle._upgrade_legacy`'s, and for the same reason -- a legacy
    checkpoint's `scaler_mean`/`scaler_std` ARE a fitted `standardize` step,
    written down before there was a name for it.
    """
    return FeaturePipeline.from_spec([
        {"op": "gather", "backbone": backbone_key},
        {"op": "standardize", "mean": [float(x) for x in mean], "std": [float(x) for x in std]},
    ])


def calibrate_checkpoint(head_path: str | Path, *, embeddings_dir: Path = EMBEDDINGS_DIR) -> dict:
    """Load a legacy-shaped checkpoint (`state_dict`/`head_kind`/`backbone`/
    `in_dim`/`scaler_mean`/`scaler_std`) and calibrate it. Mirrors
    `scripts/derive_threshold.load_head` -- CPU only, no backbone load."""
    ckpt = torch.load(Path(head_path), map_location="cpu", weights_only=False)
    head = build_head(ckpt["head_kind"], ckpt["in_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
    pipeline = legacy_pipeline(ckpt["backbone"], mean, std)
    return calibrate_threshold(head, pipeline, embeddings_dir=embeddings_dir)


def _resolve_head(name: str, models_dir: Path) -> Path | None:
    """Look in `models_dir` then `models_dir/archive` -- see
    `scripts/export_eval_stats.resolve_ckpt`, the same rule, duplicated for the
    same reason `calibrate`'s protocol is duplicated rather than imported: a
    library module must not depend on `scripts/`. Superseded ablation arms move
    to `archive/` rather than being deleted, precisely so a check like this one
    keeps finding them."""
    for candidate in (models_dir / name, models_dir / "archive" / name):
        if candidate.exists():
            return candidate
    return None


def verify_recorded_table(models_dir: Path | None = None) -> list[dict]:
    """Regression test on the PROTOCOL: does it still reproduce RECORDED_TABLE.

    One dict per row of `RECORDED_TABLE`: `{"head", "skipped": True}` if the
    checkpoint is not on disk, else `{"head", "skipped": False, "ok", "recorded",
    "got"}`. Returning data rather than asserting/printing (as
    `scripts/derive_threshold.py --verify` does) lets a test assert `ok` for
    every non-skipped row and see the mismatch inline on failure.
    """
    models_dir = Path(models_dir) if models_dir is not None else (ROOT_DIR / "models")
    out = []
    for name, thr, fpr, tpr in RECORDED_TABLE:
        path = _resolve_head(name, models_dir)
        if path is None:
            out.append({"head": name, "skipped": True})
            continue
        r = calibrate_checkpoint(path)
        ok = (
            abs(r["threshold"] - thr) < 1e-9
            and abs(r["fpr"] - fpr) < 5e-5
            and abs(r["tpr"] - tpr) < 5e-5
        )
        out.append(
            {
                "head": name,
                "skipped": False,
                "ok": ok,
                "recorded": (thr, fpr, tpr),
                "got": (r["threshold"], r["fpr"], r["tpr"]),
            }
        )
    return out
