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

from aigc_detect.config import EMBEDDINGS_DIR, RANDOM_SEED, ROOT_DIR, TRAIN_MANIFEST, VAL_MANIFEST
from aigc_detect.heads import build_head
from aigc_detect.transforms import eval_view_names, train_chain_view_names


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


# Degradations the augmented head is allowed to TRAIN on: one severity per
# family. Everything else in the 18-view grid -- the other severities and, most
# importantly, all three chained views -- is held out and only ever evaluated.
#
# This split is the entire point of the ablation. Training on every view and
# then reporting a high grid score would prove nothing: a head that saw
# blur_sigma2.0 in training is expected to handle blur_sigma2.0. Only the
# held-out views can distinguish "learned which embedding directions to ignore"
# from "memorized these specific corruptions", and the chains are the strictest
# of them, being compositions the head never saw in any form.
TRAIN_VIEWS_DEFAULT = (
    "clean",
    "jpeg_q70",
    "blur_sigma1.0",
    "resize_0.5x",
    "noise_sigma0.05",
    "color_jitter",
    "center_crop_80",
)

# TRAIN_VIEWS_DEFAULT plus the four training-only chains. Those chains are built
# exclusively from severities already in TRAIN_VIEWS_DEFAULT, so switching to
# this set adds *composition* and nothing else -- the severity holdout
# (q90/q50/q30, blur 0.5/2.0, resize 0.25x, noise 0.02/0.1) is unchanged, and
# the three scored chains remain unseen. See transforms.TRAIN_CHAIN_SPECS.
TRAIN_VIEWS_WITH_CHAINS = TRAIN_VIEWS_DEFAULT + train_chain_view_names()


def _grid_auc(head, device, view_arrays, mean, std) -> tuple[float, float]:
    """(AUC_clean, AUC_robust pooled) over cached val views -- the actual
    objective, watched per epoch instead of clean accuracy alone."""
    head.eval()
    probs_by_view = {}
    with torch.no_grad():
        for name, (emb, _lab) in view_arrays.items():
            x = torch.from_numpy((emb - mean) / std).to(device)
            probs_by_view[name] = torch.sigmoid(head(x).squeeze(-1)).cpu().numpy()

    clean_labels = view_arrays["clean"][1]
    auc_clean = roc_auc_score(clean_labels, probs_by_view["clean"])
    degraded = [n for n in view_arrays if n != "clean"]
    pooled_p = np.concatenate([probs_by_view[n] for n in degraded])
    pooled_y = np.concatenate([view_arrays[n][1] for n in degraded])
    return float(auc_clean), float(roc_auc_score(pooled_y, pooled_p))


def train_head_on_views(
    backbone_key: str,
    train_stem: str,
    val_stem: str,
    train_views: tuple[str, ...] = TRAIN_VIEWS_DEFAULT,
    head_kind: str = "linear",
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 128,
    weight_decay: float = 0.0,
    out_path: str | Path | None = None,
    train_manifest: str | Path | None = None,
    train_sample_rows: int | None = None,
    val_manifest: str | Path | None = None,
    val_sample_rows: int | None = None,
    seed: int = RANDOM_SEED,
):
    """Train a head on cached CLEAN + DEGRADED embeddings of the same images.

    FINDINGS trap 4: with a frozen backbone only ~1,025 parameters train, so
    augmentation's usual anti-overfitting justification does not apply here.
    Its value in this setup is different -- showing the head paired
    clean/degraded embeddings of the *same* image teaches it which directions
    in embedding space carry degradation rather than generation evidence. Run 4
    of NARRATIVE.md is what this is aimed at: the failure there is a decision
    boundary that moves under blur/resize, not a loss of ranking signal.

    The scaler is computed from the CLEAN train view only, in this arm and in
    the clean-only control alike. It is a preprocessing constant, and holding
    it fixed keeps the ablation's single variable "which rows the head saw"
    rather than also "which statistics standardized them".

    Every view cache is loaded through ``load_view_cache``, which verifies its
    view_fingerprint (catches an edited transform) and, when
    ``train_manifest``/``val_manifest`` are given, its manifest_fingerprint
    against the row selection those manifests yield right now (catches a
    rebuilt split) -- see FINDINGS trap 7 and eval_grid.evaluate_grid, which
    does the same check. When the manifest is not given, the weaker invariant
    that ALL train views (and, separately, ALL val views) share one
    manifest_fingerprint is still enforced, so caches from different row
    selections cannot be silently mixed. Train and val are never
    cross-compared -- they are different manifests and legitimately differ.
    """
    from aigc_detect.embed import fingerprint_paths
    from aigc_detect.embed_views import load_view_cache, select_rows
    from aigc_detect.transforms import build_robustness_views

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, all_specs = build_robustness_views()  # specs are resolution-independent

    def _expected_fp(manifest, sample_rows):
        if manifest is None:
            return None
        df = select_rows(manifest, sample_rows=sample_rows)
        return fingerprint_paths(df["image_path"])

    train_expected_fp = _expected_fp(train_manifest, train_sample_rows)
    val_expected_fp = _expected_fp(val_manifest, val_sample_rows)

    train_arrays = {}
    train_fps: set = set()
    for view in train_views:
        emb, labels, meta = load_view_cache(
            backbone_key, train_stem, view, all_specs[view], expected_manifest_fp=train_expected_fp
        )
        train_arrays[view] = (emb, labels)
        train_fps.add(meta["manifest_fingerprint"])
    if len(train_fps) > 1:
        raise SystemExit(
            f"[train-views] STALE: the requested train views do not share one manifest "
            f"fingerprint ({sorted(train_fps)}) -- caches from different row selections were "
            f"mixed under stem '{train_stem}'. Re-run embed-views for the train manifest with "
            f"matching flags."
        )

    # Validate on the SCORED views only, so the per-epoch AUC_robust printed
    # here is the same quantity eval-grid reports. A trainchain_* cache landing
    # in this glob would silently score the head on its own training data.
    scored = set(eval_view_names())
    val_arrays = {}
    val_fps: set = set()
    for p in sorted(EMBEDDINGS_DIR.glob(f"{backbone_key}__{val_stem}__*.npz")):
        view = p.name[len(f"{backbone_key}__{val_stem}__") : -len(".npz")]
        if view not in scored:
            continue
        emb, labels, meta = load_view_cache(
            backbone_key, val_stem, view, all_specs[view], expected_manifest_fp=val_expected_fp
        )
        val_arrays[view] = (emb, labels)
        val_fps.add(meta["manifest_fingerprint"])
    if len(val_fps) > 1:
        raise SystemExit(
            f"[train-views] STALE: the cached val views do not share one manifest fingerprint "
            f"({sorted(val_fps)}) -- caches from different row selections were mixed under stem "
            f"'{val_stem}'. Re-run embed-views for the val manifest with matching flags."
        )
    if "clean" not in val_arrays:
        raise SystemExit(f"[train-views] no cached '{val_stem}' clean view to validate against.")

    print(
        f"[train-views] fingerprint OK: {len(train_arrays)} train view(s) share "
        f"manifest_fingerprint {next(iter(train_fps))}"
    )
    print(
        f"[train-views] fingerprint OK: {len(val_arrays)} val view(s) share "
        f"manifest_fingerprint {next(iter(val_fps))}"
    )

    clean_emb, clean_labels = train_arrays["clean"]
    mean = clean_emb.mean(axis=0)
    std = clean_emb.std(axis=0)
    std[std == 0] = 1.0

    x = np.concatenate([(train_arrays[v][0] - mean) / std for v in train_views])
    y = np.concatenate([train_arrays[v][1] for v in train_views]).astype(np.float32)

    held_out = sorted(set(val_arrays) - set(train_views))
    print(f"[train-views] backbone={backbone_key} head={head_kind} device={device}")
    print(f"[train-views] train stem={train_stem} images={len(clean_emb)} views={len(train_views)} "
          f"-> {len(x):,} training rows")
    print(f"[train-views] TRAINED views:  {', '.join(train_views)}")
    print(f"[train-views] HELD-OUT views: {', '.join(held_out)}")
    print(f"[train-views] validating on {val_stem} ({len(val_arrays)} cached views)")

    # SEEDED. Two sources of run-to-run variation live here: the head's weight
    # init and the shuffle order of the DataLoader. Left unseeded, repeat runs
    # of an identical configuration differ by roughly +/-0.0005 in
    # AUC_robust(pooled) -- measured, not estimated.
    #
    # That is small against the effects this harness has been used to measure
    # (augmented-vs-clean training moved the score by +0.04, eighty times
    # larger), but it is NOT small against a backbone race: the realistic gap
    # between candidate backbones on this data is +0.003-0.006, so unseeded
    # training noise would be a tenth of the effect size and could reorder two
    # close backbones on seed luck alone. Every other stochastic element in this
    # project is already pinned (view generation, subsample selection); this was
    # the one that was not.
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    head = build_head(head_kind, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        head.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            seen += xb.size(0)
        auc_clean, auc_robust = _grid_auc(head, device, val_arrays, mean, std)
        print(
            f"[train-views] epoch {epoch}/{epochs}  train_loss={total / seen:.4f}  "
            f"val AUC_clean={auc_clean:.4f}  AUC_robust(pooled)={auc_robust:.4f}  "
            f"score={0.5 * auc_clean + 0.5 * auc_robust:.4f}"
        )

    if out_path is None:
        tag = "aug" if len(train_views) > 1 else "cleanonly"
        out_path = ROOT_DIR / "models" / f"{backbone_key}__{head_kind}__{tag}.pt"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "head_kind": head_kind,
            "backbone": backbone_key,
            "in_dim": int(x.shape[1]),
            "scaler_mean": mean,
            "scaler_std": std,
            "train_stem": train_stem,
            "train_views": list(train_views),
            "held_out_views": held_out,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
            "final_val_auc": auc_clean,
            "final_val_auc_robust_pooled": auc_robust,
        },
        out_path,
    )
    print(f"[train-views] saved -> {out_path}")
    return out_path


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
