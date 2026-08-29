"""Train a classifier head on cached frozen-backbone embeddings.

Paper defaults ("Simplicity Prevails", arXiv:2602.01738): AdamW, lr 1e-3,
batch 128, 2 epochs, BCEWithLogitsLoss on a single linear layer. All
overridable by CLI flag (see main.py's `train-head` subcommand).

Features are standardized (zero mean, unit std) using TRAIN-set statistics
only -- the val set never contributes to the scaler, to avoid leakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from aigc_detect.config import ROOT_DIR, TRAIN_MANIFEST, VAL_MANIFEST
from aigc_detect.heads import build_head


def _load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    sources = data["sources"].astype(str)
    return embeddings, labels, sources, data


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _assert_fresh(npz_path: Path, manifest_path: Path, tag: str) -> None:
    """Refuse to train on embeddings whose manifest has changed since they were
    computed. See FINDINGS.md trap 7 -- a stale cache here is silent and every
    downstream metric would be wrong but plausible."""
    import numpy as _np

    from aigc_detect.embed import manifest_fingerprint

    with _np.load(npz_path, allow_pickle=True) as d:
        cached = str(d["manifest_fingerprint"]) if "manifest_fingerprint" in d else None
    if cached is None:
        raise SystemExit(
            f"[train-head] {npz_path.name} has no manifest fingerprint (written before that "
            f"check existed). Re-run: uv run main.py embed --backbone <key> --manifest {tag} --force"
        )
    current = manifest_fingerprint(manifest_path)
    if cached != current:
        raise SystemExit(
            f"[train-head] STALE EMBEDDINGS: {npz_path.name} was computed from a different "
            f"{tag} manifest than {manifest_path} holds now (the split was rebuilt). "
            f"Re-run: uv run main.py embed --backbone <key> --manifest {tag} --force"
        )
    print(f"[train-head] {tag} embeddings match {manifest_path.name} (fingerprint OK)")


def train_head(
    train_npz: str | Path,
    val_npz: str | Path,
    backbone_key: str,
    head_kind: str = "linear",
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 128,
    weight_decay: float = 0.0,
    out_path: str | Path | None = None,
):
    train_npz, val_npz = Path(train_npz), Path(val_npz)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # `main.py split` rewrites train.csv/val.csv in place, so a cached .npz can
    # belong to a completely different set of images while keeping its filename.
    # embed.py fingerprints the manifest it embedded; refuse to train if that no
    # longer matches, because the failure is otherwise silent and the resulting
    # metrics look entirely plausible. See FINDINGS.md trap 7.
    _assert_fresh(train_npz, TRAIN_MANIFEST, "train")
    _assert_fresh(val_npz, VAL_MANIFEST, "val")

    train_emb, train_labels, train_sources, train_meta = _load_npz(train_npz)
    val_emb, val_labels, val_sources, _ = _load_npz(val_npz)

    in_dim = train_emb.shape[1]
    assert val_emb.shape[1] == in_dim, (
        f"train embedding dim {in_dim} != val embedding dim {val_emb.shape[1]} "
        f"-- did you generate them with the same --backbone?"
    )

    # Standardize using TRAIN statistics only (no val leakage).
    mean = train_emb.mean(axis=0)
    std = train_emb.std(axis=0)
    std[std == 0] = 1.0  # guard against dead/constant dims

    train_emb_s = _standardize(train_emb, mean, std)
    val_emb_s = _standardize(val_emb, mean, std)

    print(
        f"[train-head] backbone={backbone_key} head={head_kind} in_dim={in_dim} "
        f"train_n={len(train_emb)} val_n={len(val_emb)} device={device}"
    )
    print(f"[train-head] train label counts: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"[train-head] val label counts:   {dict(zip(*np.unique(val_labels, return_counts=True)))}")

    train_ds = TensorDataset(torch.from_numpy(train_emb_s), torch.from_numpy(train_labels.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.from_numpy(val_emb_s).to(device)
    val_y = val_labels

    head = build_head(head_kind, in_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = total_loss / n_seen

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x).squeeze(-1)
            val_loss = criterion(val_logits, torch.from_numpy(val_y.astype(np.float32)).to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(np.int64)
        val_auc = roc_auc_score(val_y, val_probs)
        val_bacc = balanced_accuracy_score(val_y, val_preds)

        print(
            f"[train-head] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  val_balanced_acc={val_bacc:.4f}"
        )

    # Per-source breakdown, using the final epoch's val predictions/probs.
    print("[train-head] val ROC-AUC by source:")
    for source in sorted(set(val_sources)):
        mask = val_sources == source
        y_src = val_y[mask]
        if len(set(y_src.tolist())) < 2:
            print(f"[train-head]   {source}: n={mask.sum()} -- skipped (single class in this source's val split)")
            continue
        auc_src = roc_auc_score(y_src, val_probs[mask])
        bacc_src = balanced_accuracy_score(y_src, val_preds[mask])
        print(f"[train-head]   {source}: n={mask.sum()} auc={auc_src:.4f} balanced_acc={bacc_src:.4f}")

    if val_auc > 0.98:
        print(
            f"[train-head] WARNING: val_auc={val_auc:.4f} is suspiciously high for this task -- "
            f"this dataset is known to contain shortcuts (see scripts/audit_data.py); treat this "
            f"as evidence of a possible leak or shortcut, not a clean win, until checked further."
        )

    if out_path is None:
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{backbone_key}__{head_kind}.pt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_kind": head_kind,
            "backbone": backbone_key,
            "checkpoint": str(train_meta["checkpoint"]) if "checkpoint" in train_meta else None,
            "in_dim": in_dim,
            "scaler_mean": mean,
            "scaler_std": std,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "final_val_auc": val_auc,
            "final_val_balanced_acc": val_bacc,
        },
        out_path,
    )
    print(f"[train-head] saved head + scaler + metadata -> {out_path}")
    return out_path
