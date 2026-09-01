"""Inference: score a directory of images with a trained head, emit JSON.

Deliverable 5.5.2. Contract (from the challenge brief):

    uv run python predict.py --input_dir <dir> --output preds.json

    Output: a JSON array of objects [{"image_path": <str>, "pred": <float 0..1>}, ...]
    where pred is P(AIGC) -- 1.0 = AI-generated, 0.0 = real.

PREPROCESSING PARITY IS THE WHOLE POINT OF THIS MODULE. It must reproduce
embed.py's pipeline exactly:

  1. Read the backbone key from the checkpoint's "backbone" field (never
     hardcode it -- a checkpoint trained on a different backbone than assumed
     would silently produce plausible-looking garbage scores).
  2. Load that backbone via `aigc_detect.registry.backbones.load_backbone`, which
     reports the backbone's OWN native_res and norm_mean/norm_std. These are
     per-backbone (PE-Core-L is 336px / 0.5,0.5,0.5; others differ) --
     config.IMAGE_SIZE and config.NORM_MEAN are ImageNet defaults for a
     from-scratch model and are NOT what the frozen VFMs were pretrained
     with. Using them here would be a silent, unit-test-invisible mismatch:
     every image would still produce *a* number in [0, 1], just the wrong
     one.
  3. Build the transform the same way embed.py does: aspect-preserving
     resize + center crop to native_res (build_backbone_transform), then
     ToImage/ToDtype/Normalize with the backbone's own stats. No training-
     time augmentation -- inference is always the deterministic "clean" path.
  4. Standardize the pooled embedding with the checkpoint's OWN scaler_mean/
     scaler_std before the head. This scaler is a trained artifact (TRAIN-set
     embedding statistics from train_head.py / train_head_on_views), not
     something to recompute from the inference batch -- recomputing it here
     would re-center every batch around its own mean, which erases exactly
     the shift a real-vs-AIGC embedding carries relative to the training
     distribution (the signal the head was fit to detect).

Everything above matches eval_grid.py's "load a saved head and apply it"
logic (same scaler-from-checkpoint, same standardize-then-sigmoid), just
fed from raw images on disk instead of a cached .npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from aigc_detect.data.transforms import build_backbone_transform
from aigc_detect.registry.backbones import load_backbone
from aigc_detect.registry.heads import build_head

# Case-insensitive; brief doesn't specify tiff/gif so kept to the common web set.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_images(input_dir: str | Path) -> list[Path]:
    """Recurse input_dir for image files by extension (case-insensitive).

    Sorted for deterministic ordering (requirement 8) -- Path.rglob's
    traversal order is filesystem-dependent and must not leak into output
    ordering, which downstream consumers may diff run-to-run.
    """
    input_dir = Path(input_dir)
    paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


class _InferenceImageDataset(Dataset):
    """Loads images by absolute path (not a manifest), skipping unreadable
    files instead of crashing the whole run.

    Unlike ManifestImageDataset, this has no label column to fall back on and
    no manifest CSV to pre-validate against -- a directory of arbitrary,
    possibly-corrupt user images is exactly the case where "one bad file
    kills the batch" is unacceptable for a competition deliverable. Bad files
    are recorded and reported, not raised.
    """

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform
        self.skipped: list[tuple[Path, str]] = []
        # Pre-filter unreadable files up front so __getitem__ never needs to
        # change the effective dataset length mid-run (DataLoader assumes a
        # fixed __len__ <-> valid-index mapping).
        good = []
        for p in self.paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                good.append(p)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                self.skipped.append((p, f"{type(exc).__name__}: {exc}"))
        self.paths = good

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


def _to_posix_relative(path: Path, input_dir: Path) -> str:
    """Forward-slash path relative to --input_dir, per requirement 6 (platform
    portability of the emitted JSON, independent of the host OS)."""
    try:
        rel = path.relative_to(input_dir)
    except ValueError:
        rel = path
    return rel.as_posix()


# Operating point for turning P(AIGC) into a verdict.
#
# The JSON contract is fixed by the brief -- {"image_path", "pred"} with pred the
# raw probability -- so this never changes what is written. It sets the decision
# boundary for the console summary, for demo/server.py, and for anything else
# that needs a yes/no rather than a score.
#
# 0.98, not 0.5. Chosen on a HELD-OUT split so the number is not tuned on the
# tier it is reported against. The protocol is no longer prose: it lives in
# scripts/derive_threshold.py, which pins the split that was twice reconstructed
# by hand and twice came out different. Re-derive on every head swap --
#
#     uv run python scripts/derive_threshold.py --head models/<name>.pt
#
# WildRF (2,503 real Reddit/X/Facebook photographs and real-world AI, which
# nothing trains on) pooled over clean + the CDN-like views a browser extension
# actually sees (jpeg_q70, jpeg_q90, resize_0.5x, chain_light), split by IMAGE so
# no image appears on both sides, swept in 0.005 steps, picked by F1 on half A,
# reported on half B:
#
#     head              chosen   HELD-OUT FPR   HELD-OUT TPR
#     photoreal          0.920         0.0408         0.9721
#     trainext           0.940         0.0283         0.9686
#     nosd3_e1           0.990         0.0180         0.9692   <-- previous
#     alltransforms_e1   0.985         0.0248         0.9768
#     allsev_e1          0.980         0.0215         0.9797   <-- SHIPPING
#
# NOTE the previous head shipped at 0.985, not the 0.990 above. That 0.985 came
# from a split that was never recorded and cannot be reproduced. Every row here
# comes from the one pinned split, so the rows are comparable to each other even
# where they differ from what was historically shipped.
#
# allsev_e1 vs nosd3_e1 is NOT strict dominance. It is +1.05 points of recall
# (95% CI [+0.51, +1.68]) at an FPR difference of +0.34 points whose CI spans
# zero. Matched on either axis it wins the other: at nosd3_e1's own FPR its
# recall is .9771 vs .9692 (97 misses -> 72); at allsev_e1's own recall its FPR
# is .0215 vs .0251. The decisive margin is on degradations NEITHER head trained
# on -- the three scored chains -- where OOD recall at a matched 2.5% FPR goes
# .4183 -> .6098. See FINDINGS 2l.
#
# It is the 1-EPOCH arm. train_head_on_views saves the final epoch with no
# best-epoch selection and epoch 2 overfits robustness (allsev_e2 gives the whole
# gain back: ood transformed mean .9724 -> .9693). Train with --epochs 1.
#
# At threshold 0.5 the same tier gives FPR 0.1875 / TPR 0.9949, so this is a
# large cut in false positives for a small amount of recall. That trade is right
# for this product: telling someone their own photograph is AI-generated costs
# far more than missing one AI image among many.
DECISION_THRESHOLD = 0.980


def run_inference(
    input_dir: str | Path,
    head_path: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    threshold: float | None = None,
) -> Path:
    threshold = DECISION_THRESHOLD if threshold is None else float(threshold)
    input_dir = Path(input_dir)
    head_path = Path(head_path)
    output_path = Path(output_path)

    if not input_dir.is_dir():
        raise SystemExit(f"[predict] --input_dir does not exist or is not a directory: {input_dir}")
    if not head_path.exists():
        raise SystemExit(f"[predict] head checkpoint not found: {head_path}")

    # Requirement 3: the backbone key comes from the checkpoint, never a flag
    # or a hardcoded default -- a head is only valid against the exact
    # backbone it was trained on (eval_grid.py enforces the same rule).
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    backbone_key = ckpt["backbone"]
    head_kind = ckpt["head_kind"]
    in_dim = ckpt["in_dim"]
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)

    print(f"[predict] head={head_path.name} backbone={backbone_key} head_kind={head_kind} in_dim={in_dim}")

    module, pooled_dim, native_res = load_backbone(backbone_key)
    if pooled_dim != in_dim:
        raise SystemExit(
            f"[predict] backbone '{backbone_key}' pooled_dim={pooled_dim} does not match the "
            f"checkpoint's in_dim={in_dim} -- checkpoint/backbone mismatch."
        )
    print(
        f"[predict] backbone native_res={native_res} pooled_dim={pooled_dim} "
        f"norm_source={module.norm_source} mean={module.norm_mean} std={module.norm_std}"
    )

    head = build_head(head_kind, in_dim)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    device = next(module.parameters()).device
    head.to(device)

    # Same transform as embed.py's precompute_embeddings: aspect-preserving
    # resize + center crop at the backbone's OWN native_res, then the
    # backbone's OWN norm stats. Deliberately not build_eval_transform()'s
    # defaults (config.IMAGE_SIZE / config.NORM_MEAN) -- see module docstring.
    transform = v2.Compose(
        [
            *build_backbone_transform(native_res),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=module.norm_mean, std=module.norm_std),
        ]
    )

    all_paths = find_images(input_dir)
    print(f"[predict] found {len(all_paths)} candidate image file(s) under {input_dir}")
    if not all_paths:
        print("[predict] no images found -- writing an empty JSON array")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return output_path

    ds = _InferenceImageDataset(all_paths, transform)
    for p, reason in ds.skipped:
        print(f"[predict] WARNING: skipping unreadable file {p}: {reason}")
    if ds.skipped:
        print(f"[predict] skipped {len(ds.skipped)} of {len(all_paths)} file(s) as unreadable/corrupt")
    if len(ds) == 0:
        print("[predict] no readable images remained -- writing an empty JSON array")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return output_path

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    use_amp = device.type == "cuda"

    all_probs = np.empty((len(ds),), dtype=np.float32)
    with torch.no_grad():
        for images, idxs in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                feats = module(images)
            feats = feats.float()
            # Checkpoint's scaler, not batch statistics -- see module docstring
            # point 4. Standardizing on the inference batch would re-center
            # around whatever this particular folder happens to contain.
            mean_t = torch.from_numpy(scaler_mean).to(feats.device)
            std_t = torch.from_numpy(scaler_std).to(feats.device)
            x = (feats - mean_t) / std_t
            logits = head(x).squeeze(-1)
            # RAW sigmoid, deliberately. Do not "improve" this with calibration.
            #
            # These scores are badly calibrated as probabilities -- measured on
            # held-out WildRF, a raw 0.95 corresponds to an actual P(AIGC) of
            # 0.26, and a raw 0.61 to 0.026. Platt scaling fixes that (Brier
            # 0.0722 -> 0.0157, log-loss 0.2432 -> 0.0615) and is tempting.
            #
            # It would not buy a single point of score. The competition metric is
            #     0.5 * AUC_clean + 0.5 * AUC_robust
            # and AUC depends only on the ORDER of the scores, so any strictly
            # monotone map leaves it bit-identical -- measured, not assumed:
            # raw and Platt both give 0.99717 on the held-out half. Isotonic is
            # monotone but not STRICTLY monotone (it is a step function), so it
            # ties scores together and actively costs AUC: 0.99671.
            #
            # Ensembling several heads is not monotone in any one head and so
            # COULD move AUC. It was tried: all 41 two-, three- and four-head
            # combinations of the checkpoints in models/ scored below this head
            # alone (best 0.99337 vs 0.99357 mean over ood/wildrf/demo_val),
            # logit- and probability-averaging alike.
            #
            # So the score-maximising inference path is the simplest one, and
            # calibration only becomes worth adding if the deliverable ever
            # needs a probability rather than a ranking.
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs[idxs.numpy()] = probs

    results = [
        {"image_path": _to_posix_relative(p, input_dir), "pred": float(all_probs[i])}
        for i, p in enumerate(ds.paths)
    ]
    # Requirement 8: deterministic ordering. ds.paths already came from a
    # sorted find_images(); re-sort here too so the guarantee holds even if
    # dataset construction order ever changes independently of that.
    results.sort(key=lambda r: r["image_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    n_skipped = len(ds.skipped)
    print(f"[predict] wrote {len(results)} prediction(s) -> {output_path}")
    # Summary only -- the JSON above carries raw probabilities, unchanged.
    n_flagged = int((all_probs >= threshold).sum())
    print(f"[predict] at threshold {threshold:g}: {n_flagged} of {len(results)} "
          f"flagged AIGC ({n_flagged / max(len(results), 1):.1%})")
    if n_skipped:
        print(f"[predict] {n_skipped} file(s) skipped as unreadable/corrupt (see warnings above)")
    return output_path
