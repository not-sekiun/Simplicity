"""Shortcut audit for the raw training data (5.x pre-training sanity check).

Real-vs-AIGC datasets can leak the label through something that has nothing
to do with image content -- most notoriously, aspect ratio: if every AIGC
image happens to be a square render and every real photo is not, a model
(or even a non-aspect-preserving resize in the pipeline) can "solve" the task
by looking at width/height alone. This script is a repeatable check for that
class of bug, run against data/raw/*_index.csv (NEVER data/demo_val/, which
is a policy-isolated benchmark this script must not touch).

It does three things per source (plus pooled across all sources):

  1. Descriptive stats per (source, label): image count, PIL format mix,
     top-5 resolutions, aspect-ratio min/median/max, and % exactly square.
  2. A "blind probe": downscale each sampled image to 16x16 grayscale,
     flatten to a 256-dim vector (no real visual content, just coarse shape/
     tone), and fit sklearn LogisticRegression to see how well *that alone*
     predicts the label. Balanced accuracy / ROC-AUC well above chance
     (~0.70+) means some shortcut survives; near 0.50 means the probe found
     nothing to exploit.
  3. Optionally (--transform), run the probe on tensors produced by the
     actual eval pipeline (build_eval_transform()) instead of raw images --
     this proves whether a fix in transforms.py actually closes the leak
     end to end, not just on paper.

Usage:
    uv run main.py audit-data
    uv run main.py audit-data --sample 600
    uv run main.py audit-data --transform

Or directly:
    uv run python scripts/audit_data.py --sample 600 --transform
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from aigc_detect.config import LABEL_NAMES, RANDOM_SEED, RAW_DIR

PROBE_SIDE = 16  # downscale side length for the blind probe (16x16 = 256-dim)
SHORTCUT_THRESHOLD = 0.70  # balanced accuracy above this -> a shortcut survives


def load_source_frames() -> dict[str, pd.DataFrame]:
    """One DataFrame per data/raw/<source>_index.csv (never demo_val)."""
    index_files = sorted(RAW_DIR.glob("*_index.csv"))
    if not index_files:
        raise FileNotFoundError(f"No *_index.csv files found under {RAW_DIR}.")
    frames: dict[str, pd.DataFrame] = {}
    for f in index_files:
        df = pd.read_csv(f)
        source = df["source"].iloc[0] if len(df) else f.stem.replace("_index", "")
        frames[source] = df
    return frames


def _sample_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (RAW_DIR.parents[1] / p)


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------


def describe_group(paths: list[Path]) -> dict:
    formats: Counter = Counter()
    resolutions: Counter = Counter()
    ratios: list[float] = []
    n_square = 0
    n_ok = 0
    for p in paths:
        try:
            with Image.open(p) as img:
                formats[img.format or "?"] += 1
                w, h = img.size
                resolutions[(w, h)] += 1
                ratios.append(w / h)
                if w == h:
                    n_square += 1
                n_ok += 1
        except Exception:
            continue

    ratios_arr = np.array(ratios) if ratios else np.array([float("nan")])
    return {
        "n_sampled": n_ok,
        "formats": formats.most_common(),
        "top_resolutions": resolutions.most_common(5),
        "aspect_min": float(np.min(ratios_arr)),
        "aspect_median": float(np.median(ratios_arr)),
        "aspect_max": float(np.max(ratios_arr)),
        "pct_square": 100.0 * n_square / n_ok if n_ok else float("nan"),
    }


# ---------------------------------------------------------------------------
# Blind probe
# ---------------------------------------------------------------------------


def _image_probe_vector(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((PROBE_SIDE, PROBE_SIDE), Image.BILINEAR)
            return (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)
    except Exception:
        return None


def _tensor_probe_vector(path: Path, transform) -> np.ndarray | None:
    """Runs the actual eval pipeline on the image, then downsamples the
    resulting tensor to a 16x16 grayscale vector -- same probe, but on what
    the model would actually see."""
    import torch
    import torch.nn.functional as tF

    try:
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB"))
    except Exception:
        return None
    if tensor.ndim != 3:
        return None
    # Luminance-weighted grayscale from the (normalized) RGB tensor.
    weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    gray = (tensor * weights).sum(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    small = tF.interpolate(gray, size=(PROBE_SIDE, PROBE_SIDE), mode="bilinear", align_corners=False)
    return small.squeeze().reshape(-1).numpy()


def run_blind_probe(records: list[tuple[Path, int]], seed: int, use_transform: bool) -> dict:
    """records: list of (path, label). Builds probe vectors, fits/evaluates
    a train/test split LogisticRegression, returns balanced accuracy + AUC."""
    vectors: list[np.ndarray] = []
    labels: list[int] = []

    transform = None
    if use_transform:
        from aigc_detect.transforms import build_eval_transform

        transform = build_eval_transform()

    for path, label in records:
        vec = _tensor_probe_vector(path, transform) if use_transform else _image_probe_vector(path)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(label)

    if len(set(labels)) < 2 or len(labels) < 10:
        return {"n": len(labels), "balanced_acc": float("nan"), "roc_auc": float("nan"), "skipped": True}

    X = np.stack(vectors)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    try:
        y_score = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")

    return {"n": len(labels), "balanced_acc": bal_acc, "roc_auc": auc, "skipped": False}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _verdict(bal_acc: float) -> str:
    if bal_acc != bal_acc:  # NaN
        return "N/A"
    return "WARN (shortcut likely survives)" if bal_acc >= SHORTCUT_THRESHOLD else "PASS"


def run_audit(sample: int, use_transform: bool, seed: int = RANDOM_SEED) -> None:
    frames = load_source_frames()

    print("=" * 78)
    print("DATA SHORTCUT AUDIT" + (" (transformed tensors)" if use_transform else " (raw images)"))
    print("=" * 78)

    pooled_records: list[tuple[Path, int]] = []

    for source, df in frames.items():
        print(f"\n--- source: {source} ---")
        source_records: list[tuple[Path, int]] = []

        for label, group in df.groupby("label"):
            label_name = LABEL_NAMES.get(int(label), str(label))
            sampled = _sample_group(group, sample, seed)
            paths = [_resolve(p) for p in sampled["image_path"]]

            print(f"  [{label_name} label={label}] total={len(group)} sampled={len(paths)}")
            stats = describe_group(paths)
            print(f"    formats:          {stats['formats']}")
            print(f"    top resolutions:  {stats['top_resolutions']}")
            print(
                f"    aspect ratio:     min={stats['aspect_min']:.3f} "
                f"median={stats['aspect_median']:.3f} max={stats['aspect_max']:.3f}"
            )
            print(f"    pct exactly sq.:  {stats['pct_square']:.1f}%")

            source_records.extend((p, int(label)) for p in paths)

        pooled_records.extend(source_records)

        probe = run_blind_probe(source_records, seed, use_transform)
        if probe.get("skipped"):
            print(f"  [blind probe] n={probe['n']} -- skipped (need >=2 classes, >=10 samples)")
        else:
            print(
                f"  [blind probe] n={probe['n']} balanced_acc={probe['balanced_acc']:.4f} "
                f"roc_auc={probe['roc_auc']:.4f} -> {_verdict(probe['balanced_acc'])}"
            )

    print(f"\n--- pooled (all sources, n={len(pooled_records)}) ---")
    pooled_probe = run_blind_probe(pooled_records, seed, use_transform)
    if pooled_probe.get("skipped"):
        print(f"  [blind probe] n={pooled_probe['n']} -- skipped")
    else:
        print(
            f"  [blind probe] n={pooled_probe['n']} balanced_acc={pooled_probe['balanced_acc']:.4f} "
            f"roc_auc={pooled_probe['roc_auc']:.4f} -> {_verdict(pooled_probe['balanced_acc'])}"
        )

    print("\n" + "=" * 78)
    print(f"Verdict threshold: balanced_acc >= {SHORTCUT_THRESHOLD:.2f} on a 16x16-grayscale blind")
    print("probe means a label shortcut (e.g. aspect ratio / resolution) survives.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=600, help="Max images sampled per (source, label) group.")
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Run the blind probe on build_eval_transform() tensors instead of raw images.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_audit(args.sample, args.transform, args.seed)


if __name__ == "__main__":
    main()
