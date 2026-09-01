"""Re-derive a head's decision threshold under the project's recorded protocol.

WHY THIS EXISTS. The threshold is a property of the head, not a constant, so it
is re-derived on every swap (FINDINGS 2j). The protocol was described in prose
in `src/aigc_detect/predict.py` but never written down as code, and the prose
omitted one detail: the seed and rule behind the split-half assignment. It was
reconstructed twice from the recorded numbers, and the second reconstruction did
not match the first -- FINDINGS 2j's table and FINDINGS 2k's reproduction of it
disagree in the third decimal (trainext .0283/.9686 vs .0280/.9663) for no
reason other than a different unrecorded split. This module is the fix: the
protocol as executable code, with the split pinned.

THE SPLIT. `numpy.random.RandomState(0).permutation(n)`, first half = A. That is
the rule that reproduces FINDINGS 2j's originally recorded table exactly, on
every value:

    head        recorded (2j)              this module
    photoreal   0.920 / .0408 / .9721      0.920 / .0408 / .9721
    trainext    0.940 / .0283 / .9686      0.940 / .0283 / .9686

`--verify` re-runs exactly that check and fails loudly if it ever stops holding,
which makes this a regression test on the protocol and not just a calculator.

THE PROTOCOL, in full:

  1. WildRF test (2,503 real Reddit/X/Facebook photographs and real-world AI) --
     a corpus no head trains on, and the hardest REAL images available here.
  2. Pooled over clean + the CDN-like views a browser extension actually sees:
     jpeg_q70, jpeg_q90, resize_0.5x, chain_light.
  3. Split BY IMAGE, not by row, so no image contributes to both halves through
     different views.
  4. Threshold swept 0.50..0.999 in 0.005 steps; picked by F1 on half A.
  5. FPR and TPR reported on half B, which the choice never saw.

Usage:
    uv run python scripts/derive_threshold.py --head models/<name>.pt
    uv run python scripts/derive_threshold.py --all
    uv run python scripts/derive_threshold.py --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from aigc_detect.config import EMBEDDINGS_DIR
from aigc_detect.registry.heads import build_head

ROOT = Path(__file__).resolve().parents[1]

# The pool the threshold is chosen over. Kept in sync by hand with
# export_eval_stats.CDN_VIEWS and the comment on predict.DECISION_THRESHOLD.
CDN_VIEWS = ("clean", "jpeg_q70", "jpeg_q90", "resize_0.5x", "chain_light")
WILDRF_STEM = "wildrf_test"
SPLIT_SEED = 0
GRID = np.round(np.arange(0.50, 0.9991, 0.005), 4)

# (head file, threshold, held-out FPR, held-out TPR) exactly as FINDINGS 2j
# recorded them. --verify asserts this module still reproduces all of it.
RECORDED = [
    ("pe-core-l__linear__photoreal.pt", 0.920, 0.0408, 0.9721),
    ("pe-core-l__linear__trainext.pt", 0.940, 0.0283, 0.9686),
]


def load_head(name: str | Path):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / "models" / path if path.parent == Path(".") else ROOT / path
    if not path.exists():
        raise SystemExit(f"[threshold] no such checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = build_head(ckpt["head_kind"], ckpt["in_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt["scaler_mean"], ckpt["scaler_std"], ckpt["backbone"]


def pooled_scores(name: str | Path):
    """P(AIGC), labels and IMAGE INDEX for every (image, CDN view) pair."""
    head, mean, std, backbone = load_head(name)
    probs, labels, image_idx = [], [], []
    for view in CDN_VIEWS:
        cache = EMBEDDINGS_DIR / f"{backbone}__{WILDRF_STEM}__{view}.npz"
        if not cache.exists():
            raise SystemExit(
                f"[threshold] missing cache {cache.name}. Run:\n"
                f"  uv run main.py embed-views --backbone {backbone} --manifest wildrf-test"
            )
        data = np.load(cache)
        x = (data["embeddings"] - mean) / std
        with torch.no_grad():
            p = torch.sigmoid(head(torch.from_numpy(x.astype(np.float32))).squeeze(-1)).numpy()
        probs.append(p)
        labels.append(data["labels"])
        image_idx.append(np.arange(len(p)))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(image_idx)


def half_a_mask(n_images: int) -> np.ndarray:
    """The pinned split. Seeded on the IMAGE index, so every view of an image
    lands on the same side -- splitting rows instead would let half A and half B
    share images through different degradations, and the reported FPR would then
    be measured partly on images the threshold was chosen on."""
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


def derive(name: str | Path) -> dict:
    probs, labels, image_idx = pooled_scores(name)
    in_a = half_a_mask(int(image_idx.max()) + 1)[image_idx]

    pa, ya = probs[in_a], labels[in_a]
    # First argmax, i.e. the LOWEST equally-optimal threshold on the grid. Ties
    # do occur; picking the lowest is the conservative side for recall.
    threshold = float(GRID[int(np.argmax([_f1(pa, ya, t) for t in GRID]))])

    pb, yb = probs[~in_a], labels[~in_a]
    return {
        "head": Path(name).name,
        "threshold": threshold,
        "fpr": float((pb[yb == 0] >= threshold).mean()),
        "tpr": float((pb[yb == 1] >= threshold).mean()),
        "n_real_b": int((yb == 0).sum()),
        "n_aigc_b": int((yb == 1).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", nargs="+", default=None, help="Checkpoint path(s), or a bare name under models/.")
    ap.add_argument("--all", action="store_true", help="Every pe-core-l linear head in models/.")
    ap.add_argument("--verify", action="store_true",
                    help="Assert this module still reproduces FINDINGS 2j's recorded table.")
    a = ap.parse_args()

    if a.verify:
        ok = True
        print("[threshold] verifying against FINDINGS 2j's recorded table")
        for name, thr, fpr, tpr in RECORDED:
            if not (ROOT / "models" / name).exists():
                print(f"  SKIP {name} (not on disk)")
                continue
            r = derive(name)
            match = (abs(r["threshold"] - thr) < 1e-9
                     and abs(r["fpr"] - fpr) < 5e-5 and abs(r["tpr"] - tpr) < 5e-5)
            ok &= match
            print(f"  {'OK  ' if match else 'FAIL'} {name}: recorded {thr:.3f}/{fpr:.4f}/{tpr:.4f} "
                  f"-> got {r['threshold']:.3f}/{r['fpr']:.4f}/{r['tpr']:.4f}")
        raise SystemExit(0 if ok else "[threshold] PROTOCOL DRIFT -- the split no longer reproduces 2j.")

    heads = list(a.head or [])
    if a.all:
        heads = sorted(p.name for p in (ROOT / "models").glob("pe-core-l__linear__*.pt"))
    if not heads:
        raise SystemExit("[threshold] pass --head, --all or --verify.")

    rows = [derive(h) for h in heads]
    print(f"WildRF {WILDRF_STEM}, pooled over {', '.join(CDN_VIEWS)}")
    print(f"split RandomState({SPLIT_SEED}).permutation by image; F1-optimal on half A, "
          f"reported on half B (n={rows[0]['n_real_b']} real / {rows[0]['n_aigc_b']} aigc rows)")
    print()
    print(f"{'head':46s}{'threshold':>10s}{'HELD-OUT FPR':>14s}{'HELD-OUT TPR':>14s}")
    for r in rows:
        print(f"{r['head']:46s}{r['threshold']:10.3f}{r['fpr']:14.4f}{r['tpr']:14.4f}")


if __name__ == "__main__":
    main()
