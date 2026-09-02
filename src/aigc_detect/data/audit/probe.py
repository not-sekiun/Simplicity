"""The blind probe: a classifier that must FAIL, on purpose.

A real-vs-AIGC corpus can leak its label through something that has nothing to
do with image content -- most notoriously composition and aspect ratio. SID_Set's
FLUX half and OpenImages-photo half are separable at 0.935 balanced accuracy
from an 8x8 greyscale thumbnail alone, geometry controlled (docs/findings.md
1): two unrelated image piles, correlated with the label only because nobody
matched their composition. `gmongaras/Stable_Diffusion_3_Recaption` -- real
photographs paired with SD3-authored captions, not SD3 output -- would have
scored 0.0230 (indistinguishable from genuine photographs) on exactly this
probe had it been run before training; it was caught only after the fact (see
docs/findings.md's SD3 section). This module is what makes that check
run BEFORE a corpus reaches a training manifest instead of after.

WHY 16x16 GREYSCALE, LOGISTIC REGRESSION, AND ~0.70. The probe is deliberately
too weak to see anything but the shortcut: 256 pixels of blurred luminance
carries no texture, no generator artifact, nothing a real detector would use.
A linear model that clears 0.70 balanced accuracy on that alone found
something structural -- composition, aspect ratio, a resampling cue -- not
content. 0.70 is not a tuned threshold; it is comfortably above chance (0.50)
and comfortably below what an intentional, content-based classifier reaches
on the same task, so it flags exactly the "solved by a shortcut" regime this
exists to catch without also flagging ordinary noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Downscale side length for the blind probe -- 16x16 = 256-dim input.
PROBE_SIDE = 16

#: Balanced accuracy at or above this means a shortcut probably survives.
SHORTCUT_THRESHOLD = 0.70


@dataclass(frozen=True)
class BlindProbeResult:
    n: int
    balanced_acc: float
    roc_auc: float
    skipped: bool = False

    @property
    def suspect(self) -> bool:
        return (not self.skipped) and self.balanced_acc == self.balanced_acc and self.balanced_acc >= SHORTCUT_THRESHOLD

    @property
    def verdict(self) -> str:
        if self.skipped or self.balanced_acc != self.balanced_acc:  # NaN
            return "N/A"
        return "SUSPECT (shortcut likely survives)" if self.suspect else "PASS"

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "balanced_acc": None if self.balanced_acc != self.balanced_acc else round(self.balanced_acc, 4),
            "roc_auc": None if self.roc_auc != self.roc_auc else round(self.roc_auc, 4),
            "skipped": self.skipped,
            "suspect": self.suspect,
            "threshold": SHORTCUT_THRESHOLD,
            "probe": f"blind_{PROBE_SIDE}x{PROBE_SIDE}_greyscale_logreg",
        }


def image_probe_vector(path: Path) -> np.ndarray | None:
    from PIL import Image

    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((PROBE_SIDE, PROBE_SIDE), Image.BILINEAR)
            return (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)
    except Exception:
        return None


def tensor_probe_vector(path: Path, transform) -> np.ndarray | None:
    """Runs the actual eval pipeline on the image, then downsamples the
    resulting tensor to a 16x16 grayscale vector -- same probe, but on what
    the model would actually see. Used by `--transform` to prove a fix in
    `transforms.py` closes a leak end to end, not just on paper."""
    import torch
    import torch.nn.functional as tF
    from PIL import Image

    try:
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB"))
    except Exception:
        return None
    if tensor.ndim != 3:
        return None
    weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    gray = (tensor * weights).sum(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
    small = tF.interpolate(gray, size=(PROBE_SIDE, PROBE_SIDE), mode="bilinear", align_corners=False)
    return small.squeeze().reshape(-1).numpy()


def run_blind_probe(records: list[tuple[Path, int]], seed: int, use_transform: bool = False) -> BlindProbeResult:
    """records: (path, label) pairs. Fits/evaluates a held-out split
    LogisticRegression on the probe vectors; returns balanced accuracy + AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    transform = None
    if use_transform:
        from aigc_detect.data.transforms import build_eval_transform

        transform = build_eval_transform()

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    for path, label in records:
        vec = tensor_probe_vector(path, transform) if use_transform else image_probe_vector(path)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(label)

    if len(set(labels)) < 2 or len(labels) < 10:
        return BlindProbeResult(n=len(labels), balanced_acc=float("nan"), roc_auc=float("nan"), skipped=True)

    X = np.stack(vectors)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    try:
        y_score = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = float("nan")

    return BlindProbeResult(n=len(labels), balanced_acc=bal_acc, roc_auc=auc, skipped=False)
