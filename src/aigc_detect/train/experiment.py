"""`aigc experiment`: a declared training run, config in, run directory out.

WHAT THIS REPLACES. Reproducing the shipping head used to mean retyping a
seven-flag `train-head-views` command by hand, then a SEPARATE manual step
(`scripts/derive_threshold.py --head <it>`) to get its threshold, then a THIRD
step to copy that threshold into `inference.predict.DECISION_THRESHOLD`. Three
places to forget, and the project's own history (FINDINGS 2j/2k) shows the
threshold step being forgotten and reconstructed twice, disagreeing with itself
both times. This module is one function: read a YAML file, run the recipe it
declares, and write back a directory that carries the config, the trained
bundle, the eval grid and the derived threshold together -- so "what produced
this checkpoint" is a question the run directory answers, not a question
someone has to remember the answer to.

WHY A CONFIG_HASH OVER THE *RESOLVED* CONFIG. `experiments/allsev_e1.yaml`
declares `views: all_severities`, a name, not a list -- the actual 19 view
names it expands to live in `train.probe.TRAIN_VIEWS_ALL_SEVERITIES` and
would change if that constant ever did. Hashing the raw YAML would miss that:
two runs could produce different training data under a hash that claims they
match. `config_hash` is computed over the config AFTER every default is filled
in and every preset expanded -- the thing that actually determined what got
trained on -- and it travels inside the bundle (`Bundle.config_hash`) as the
answer to "did this checkpoint come from the config on disk right now".

WHY THE PROBE PATH BUILDS ITS OWN LOOP INSTEAD OF CALLING
`train.probe.train_head_on_views`. Two reasons, not one:

  1. The declared `features:` pipeline is strictly more expressive than that
     function's hardcoded "one gather, one standardize" -- a config can name
     `l2norm`, or in principle multiple `gather` steps once a second backbone
     is embedded (see `FeaturePipeline`'s module docstring). Calling into a
     function that assumes exactly one shape would silently ignore the rest of
     a declared pipeline, which is worse than not accepting one.
  2. This module owns the run directory, the config hash and the bundle;
     `train_head_on_views` owns the legacy `models/*.pt` checkpoint shape that
     `train-head-views` and 25 archived heads still depend on. Merging them
     would make one function serve two on-disk contracts.

They do NOT duplicate the standardization math: both delegate to
`FeaturePipeline` (`train.probe` was refactored in this tier to do the same),
which is the one place that arithmetic lives now -- see that module's
docstring for the history of why two copies of it was the failure mode.

THE finetune SEAM. `trainer: {kind: finetune}` is accepted by the config
loader and dispatched to `_run_finetune`, which raises `NotImplementedError`
with a specific message rather than attempting a partial training loop. A
finetune run would read IMAGES through `data.dataset.ManifestImageDataset` and
`data.transforms.build_backbone_transform` (the same modules `embed.views`
already uses) and unfreeze the backbone -- the seam is real, but nobody has
run it, and a half-working finetune loop that silently produces bad weights is
a worse failure than a clear "not implemented".
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from aigc_detect.config import EMBEDDINGS_DIR, RANDOM_SEED, ROOT_DIR, RUNS_DIR
from aigc_detect.data.manifest import assert_trainable_manifest, resolved_path
from aigc_detect.data.transforms import build_robustness_views, eval_view_names
from aigc_detect.embed.embeddings import fingerprint_paths
from aigc_detect.embed.views import cache_stem, load_view_cache, select_rows
from aigc_detect.inference.bundle import BundleBackbone, save_bundle
from aigc_detect.log import get_logger
from aigc_detect.registry.heads import build_head
from aigc_detect.train.calibrate import GRID, calibrate_threshold, pooled_wildrf_scores
from aigc_detect.train.features import FeaturePipeline

logger = get_logger(__name__)

EXPERIMENTS_DIR = ROOT_DIR / "experiments"

_TRAIN_DEFAULTS = {"epochs": 2, "lr": 1e-3, "batch_size": 128, "weight_decay": 0.0, "eval_every": 50}
_VAL_DEFAULTS = {"manifest": "val", "sample_rows": None}


# -- config: load, validate, resolve, hash ------------------------------------


def _view_preset(name: str) -> tuple[str, ...]:
    """Resolve a named view preset the same way `--with-chains`/`--all-severities`
    do on `train-head-views` -- see `train.probe` for what each one means."""
    from aigc_detect.train.probe import (
        TRAIN_VIEWS_ALL_SEVERITIES,
        TRAIN_VIEWS_DEFAULT,
        TRAIN_VIEWS_WITH_CHAINS,
    )

    presets = {
        "default": TRAIN_VIEWS_DEFAULT,
        "with_chains": TRAIN_VIEWS_WITH_CHAINS,
        "all_severities": TRAIN_VIEWS_ALL_SEVERITIES,
    }
    if name not in presets:
        raise SystemExit(
            f"[experiment] unknown views preset '{name}'. Expected a YAML list of view names, "
            f"or one of {sorted(presets)}."
        )
    return tuple(presets[name])


def experiment_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    candidate = EXPERIMENTS_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise SystemExit(
        f"[experiment] no config '{name_or_path}' under {EXPERIMENTS_DIR} and it is not a path "
        f"to an existing .yaml file. `aigc experiment list` shows what is available."
    )


def list_experiments() -> list[str]:
    return sorted(p.stem for p in EXPERIMENTS_DIR.glob("*.yaml")) if EXPERIMENTS_DIR.is_dir() else []


_REQUIRED_KEYS = ("trainer", "manifest", "backbone", "views", "features", "head", "train")


def load_experiment(name_or_path: str | Path) -> dict:
    """Load one experiment YAML, validate it, and fill in every default.

    Returns ``{"name", "path", "raw", "resolved", "config_hash"}``. `resolved`
    is what every other function in this module reads -- it is `raw` with
    presets expanded (`views`) and defaults filled (`train`, `val`, `seed`,
    `balance`, ...), which is also exactly what `config_hash` hashes. Raises
    `SystemExit` on a structurally invalid config (missing key, bad trainer
    kind) rather than letting a typo surface as a `KeyError` three calls deep
    into training.
    """
    path = experiment_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"[experiment] {path.name}: expected a YAML mapping at the top level")

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise SystemExit(f"[experiment] {path.name}: missing required key(s) {missing}")

    trainer_spec = raw["trainer"]
    trainer_kind = trainer_spec.get("kind") if isinstance(trainer_spec, dict) else trainer_spec
    if trainer_kind not in ("probe", "finetune"):
        raise SystemExit(
            f"[experiment] {path.name}: trainer.kind must be 'probe' or 'finetune', got {trainer_kind!r}"
        )

    views_spec = raw["views"]
    views = _view_preset(views_spec) if isinstance(views_spec, str) else tuple(views_spec)
    if not views:
        raise SystemExit(f"[experiment] {path.name}: `views` resolved to an empty list")

    features_spec = raw["features"]
    if not isinstance(features_spec, list) or not features_spec:
        raise SystemExit(f"[experiment] {path.name}: `features` must be a non-empty list of op dicts")

    head_spec = raw["head"]
    head_spec = dict(head_spec) if isinstance(head_spec, dict) else {"kind": head_spec}
    if "kind" not in head_spec:
        raise SystemExit(f"[experiment] {path.name}: `head` needs a `kind`")

    resolved = {
        "trainer": {"kind": trainer_kind},
        "manifest": str(raw["manifest"]),
        "backbone": str(raw["backbone"]),
        "views": list(views),
        "features": [dict(s) for s in features_spec],
        "head": head_spec,
        "train": {**_TRAIN_DEFAULTS, **dict(raw.get("train") or {})},
        "val": {**_VAL_DEFAULTS, **dict(raw.get("val") or {})},
        "extra_train": [dict(e) for e in raw.get("extra_train") or []],
        "balance": bool(raw.get("balance", False)),
        "exclude_generators": list(raw.get("exclude_generators") or []),
        "seed": int(raw.get("seed", RANDOM_SEED)),
    }
    return {
        "name": path.stem,
        "path": str(path),
        "raw": raw,
        "resolved": resolved,
        "config_hash": config_hash(resolved),
    }


def config_hash(resolved: dict) -> str:
    """SHA-256 over the RESOLVED config in canonical JSON -- see module docstring
    for why this hashes the resolved form and not the YAML on disk verbatim."""
    canon = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# -- shared cache-loading helpers ----------------------------------------------


@dataclass
class _Pool:
    """One manifest's cached views, loaded and fingerprint-checked."""

    stem: str
    arrays: dict[str, tuple[np.ndarray, np.ndarray]]


def _expected_fp(manifest_path: Path, sample_rows: int | None) -> str:
    df = select_rows(manifest_path, sample_rows=sample_rows)
    return fingerprint_paths(df["image_path"])


def _load_pool(backbone_key: str, manifest_path: Path, views: tuple[str, ...], all_specs: dict,
               sample_rows: int | None = None) -> _Pool:
    stem = cache_stem(manifest_path, sample_rows=sample_rows)
    expected_fp = _expected_fp(manifest_path, sample_rows)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fps: set = set()
    for view in views:
        emb, labels, meta = load_view_cache(backbone_key, stem, view, all_specs[view], expected_manifest_fp=expected_fp)
        arrays[view] = (emb, labels)
        fps.add(meta["manifest_fingerprint"])
    if len(fps) > 1:
        raise SystemExit(
            f"[experiment] STALE: views for stem '{stem}' do not share one manifest fingerprint "
            f"({sorted(fps)}). Re-run embed-views for this manifest."
        )
    return _Pool(stem=stem, arrays=arrays)


def _manifest_path(name: str) -> Path:
    path = resolved_path(name)
    if not path.exists():
        raise SystemExit(
            f"[experiment] no resolved manifest '{name}' at {path}. Run: uv run aigc manifest resolve {name}"
        )
    return path


# -- probe trainer --------------------------------------------------------------


def _build_pipeline(resolved: dict, clean_emb: np.ndarray) -> FeaturePipeline:
    backbone_key = resolved["backbone"]
    pipeline = FeaturePipeline.from_spec(resolved["features"])
    if backbone_key not in pipeline.backbones:
        raise SystemExit(
            f"[experiment] `features` never gathers backbone '{backbone_key}' (declared in "
            f"`backbone:`); it gathers {pipeline.backbones}. A `gather` step's `backbone` must "
            f"match."
        )
    pipeline.fit({backbone_key: clean_emb})
    return pipeline


def _resolve_backbone_ref(backbone_key: str, dim: int) -> BundleBackbone:
    """Best-effort backbone identity, without loading a checkpoint or the GPU.

    Tries the content-addressed store's memo first (populated for every
    backbone `embed-views` has already run against -- see `cache.identity`);
    falls back to registry-only fields, the same fallback `bundle.load_bundle`
    uses for a legacy checkpoint, if the store has never seen this backbone.
    """
    try:
        from aigc_detect.cache.identity import resolve_identity
        from aigc_detect.cache.store import EmbeddingStore
        from aigc_detect.config import get_settings

        store = EmbeddingStore(get_settings().store_root)
        try:
            identity = resolve_identity(store, backbone_key, allow_load=False)
            return BundleBackbone.from_identity(identity)
        finally:
            store.close()
    except SystemExit:
        logger.info(
            "backbone '%s' not yet in the content-addressed store's memo; bundle will carry "
            "registry-only identity (no norm_mean/std/bb_id)",
            backbone_key,
        )
        return BundleBackbone.from_registry_key(backbone_key, dim=dim)


def _run_probe(cfg: dict, *, out: Path | None, log_dir: Path | None) -> dict:
    resolved = cfg["resolved"]
    backbone_key = resolved["backbone"]
    seed = resolved["seed"]

    assert_trainable_manifest(resolved["manifest"])
    train_path = _manifest_path(resolved["manifest"])
    views = tuple(resolved["views"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, all_specs = build_robustness_views()

    train_pool = _load_pool(backbone_key, train_path, views, all_specs)
    extra_pools = []
    for entry in resolved["extra_train"]:
        m_name = entry["manifest"]
        extra_pools.append(_load_pool(backbone_key, _manifest_path(m_name), views, all_specs))
        n = len(extra_pools[-1].arrays["clean"][0])
        print(f"[experiment] extra manifest={m_name} images={n:,} fingerprint OK")

    scored = set(eval_view_names())
    val_manifest = resolved["val"]["manifest"]
    val_sample_rows = resolved["val"]["sample_rows"]
    val_path = _manifest_path(val_manifest)
    val_stem = cache_stem(val_path, sample_rows=val_sample_rows)
    val_expected_fp = _expected_fp(val_path, val_sample_rows)
    val_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    val_fps: set = set()
    for p in sorted(EMBEDDINGS_DIR.glob(f"{backbone_key}__{val_stem}__*.npz")):
        view = p.name[len(f"{backbone_key}__{val_stem}__"):-len(".npz")]
        if view not in scored:
            continue
        emb, labels, meta = load_view_cache(backbone_key, val_stem, view, all_specs[view], expected_manifest_fp=val_expected_fp)
        val_arrays[view] = (emb, labels)
        val_fps.add(meta["manifest_fingerprint"])
    if len(val_fps) > 1:
        raise SystemExit(f"[experiment] STALE: val views for stem '{val_stem}' mix manifest fingerprints.")
    if "clean" not in val_arrays:
        raise SystemExit(f"[experiment] no cached '{val_stem}' clean view to validate against.")

    print(f"[experiment] backbone={backbone_key} views={len(views)} train stem={train_pool.stem} "
          f"val stem={val_stem} ({len(val_arrays)} cached views)")

    clean_emb, _ = train_pool.arrays["clean"]
    pipeline = _build_pipeline(resolved, clean_emb)

    all_pools = [train_pool, *extra_pools]

    exclude = {g.strip().lower() for g in resolved["exclude_generators"]}
    if exclude:
        stems_manifests = [(resolved["manifest"], None), *[(e["manifest"], None) for e in resolved["extra_train"]]]
        for (m_name, _sr), pool in zip(stems_manifests, all_pools, strict=True):
            df = select_rows(_manifest_path(m_name)).reset_index(drop=True)
            n_cached = len(pool.arrays["clean"][0])
            if len(df) != n_cached:
                raise SystemExit(f"[experiment] STALE: manifest '{m_name}' has {len(df):,} rows but its "
                                  f"cache has {n_cached:,}; re-run embed-views.")
            if "generator" not in df.columns:
                raise SystemExit(f"[experiment] manifest '{m_name}' has no generator column.")
            keep = ~df["generator"].astype(str).str.strip().str.lower().isin(exclude).to_numpy()
            n_dropped = int((~keep).sum())
            if n_dropped:
                for v in views:
                    emb, lab = pool.arrays[v]
                    pool.arrays[v] = (emb[keep], lab[keep])
                print(f"[experiment] excluded from '{m_name}': {n_dropped:,} images")

    x = np.concatenate([pipeline.transform({backbone_key: pool.arrays[v][0]}) for v in views for pool in all_pools])
    y = np.concatenate([pool.arrays[v][1] for v in views for pool in all_pools]).astype(np.float32)
    n_pos = int((y == 1.0).sum())
    n_neg = len(y) - n_pos
    n_images = sum(len(pool.arrays["clean"][0]) for pool in all_pools)
    print(f"[experiment] {n_images:,} images x {len(views)} views -> {len(x):,} training rows "
          f"({n_neg:,} real / {n_pos:,} aigc)")

    train_cfg = resolved["train"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=train_cfg["batch_size"], shuffle=True, generator=gen,
    )
    head_spec = dict(resolved["head"])
    head_kind = head_spec.pop("kind")
    head = build_head(head_kind, x.shape[1], **head_spec).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    if resolved["balance"] and n_pos:
        pos_weight = torch.tensor(n_neg / n_pos, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[experiment] --balance: pos_weight={pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    loss_rows: list[dict] = []
    val_rows: list[dict] = []
    trail: deque = deque(maxlen=100)
    step = 0
    eval_every = train_cfg["eval_every"]
    t0 = time.time()
    auc_clean = auc_robust = float("nan")
    for epoch in range(1, train_cfg["epochs"] + 1):
        head.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            step += 1
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
            if log_dir is not None:
                trail.append(float(loss.item()))
                loss_rows.append({"step": step, "epoch": epoch, "batch_loss": round(float(loss.item()), 6),
                                   "running_mean": round(running / seen, 6),
                                   "trailing_mean": round(sum(trail) / len(trail), 6)})
                if step % eval_every == 0:
                    ac, ar = _grid_auc(head, device, val_arrays, pipeline, backbone_key)
                    val_rows.append({"step": step, "epoch": epoch, "auc_clean": round(ac, 6),
                                      "auc_robust": round(ar, 6), "score": round(0.5 * ac + 0.5 * ar, 6)})
        auc_clean, auc_robust = _grid_auc(head, device, val_arrays, pipeline, backbone_key)
        if log_dir is not None:
            val_rows.append({"step": step, "epoch": epoch, "auc_clean": round(auc_clean, 6),
                              "auc_robust": round(auc_robust, 6), "score": round(0.5 * auc_clean + 0.5 * auc_robust, 6)})
        print(f"[experiment] epoch {epoch}/{train_cfg['epochs']}  train_loss={running / seen:.4f}  "
              f"val AUC_clean={auc_clean:.4f}  AUC_robust(pooled)={auc_robust:.4f}  "
              f"score={0.5 * auc_clean + 0.5 * auc_robust:.4f}")
    elapsed = time.time() - t0

    # -- calibrate: last step of every run, not a separate manual one ---------
    calib = calibrate_threshold(head.cpu(), pipeline, embeddings_dir=EMBEDDINGS_DIR)
    print(f"[experiment] calibrated threshold={calib['threshold']:.3f} "
          f"held-out FPR={calib['fpr']:.4f} TPR={calib['tpr']:.4f}")

    # head stays on CPU from here on: calibration, the eval-grid CSV and the
    # threshold sweep all feed it plain `torch.from_numpy` tensors, and a
    # bundle's state_dict is loaded with map_location="cpu" regardless of
    # where it was saved from -- no reason to move it back to `device`.
    backbone_ref = _resolve_backbone_ref(backbone_key, x.shape[1])
    metrics = {
        "final_val_auc_clean": auc_clean,
        "final_val_auc_robust_pooled": auc_robust,
        "final_train_loss": running / seen,
        "n_images": n_images,
        "n_rows": len(x),
        "n_real_rows": n_neg,
        "n_aigc_rows": n_pos,
        "elapsed_seconds": round(elapsed, 2),
        "views": list(views),
        "extra_train": [e["manifest"] for e in resolved["extra_train"]],
    }

    run_dir = _new_run_dir(cfg["name"])
    (run_dir / "config.json").write_text(
        json.dumps({"name": cfg["name"], "config_hash": cfg["config_hash"], "resolved": resolved}, indent=2),
        encoding="utf-8",
    )
    bundle_path = save_bundle(
        run_dir / "bundle.pt",
        backbone=backbone_ref,
        features=pipeline,
        head_kind=head_kind,
        head_state_dict=head.state_dict(),
        threshold=calib["threshold"],
        threshold_source=calib["source"],
        config_hash=cfg["config_hash"],
        metrics=metrics,
    )
    (run_dir / "threshold.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_eval_grid_csv(run_dir / "eval_grid.csv", head, pipeline, backbone_key, val_arrays,
                          trained_views=set(views), threshold=calib["threshold"])
    _write_threshold_sweep_csv(run_dir / "threshold_sweep.csv", head, pipeline, EMBEDDINGS_DIR)

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copyfile(bundle_path, out)
        print(f"[experiment] bundle also copied -> {out}")

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(log_dir / "train_loss_steps.csv", loss_rows,
                   ["step", "epoch", "batch_loss", "running_mean", "trailing_mean"])
        _write_csv(log_dir / "val_curve.csv", val_rows, ["step", "epoch", "auc_clean", "auc_robust", "score"])
        print(f"[experiment] instrumentation -> {log_dir}")

    print(f"[experiment] run directory -> {run_dir}")
    return {"run_dir": str(run_dir), "bundle": str(bundle_path), "metrics": metrics, "threshold": calib}


def _grid_auc(head, device, view_arrays, pipeline: FeaturePipeline, backbone_key: str) -> tuple[float, float]:
    """Mirrors `train.probe._grid_auc` exactly (same delegation to
    `FeaturePipeline`) -- kept as a separate copy because this module owns the
    run-directory/bundle contract and must not import training internals from
    a module that owns a different, legacy on-disk contract. See module
    docstring, reason 2."""
    head.eval()
    probs_by_view = {}
    with torch.no_grad():
        for name, (emb, _lab) in view_arrays.items():
            x = torch.from_numpy(pipeline.transform({backbone_key: emb})).to(device)
            probs_by_view[name] = torch.sigmoid(head(x).squeeze(-1)).cpu().numpy()
    clean_labels = view_arrays["clean"][1]
    auc_clean = roc_auc_score(clean_labels, probs_by_view["clean"])
    degraded = [n for n in view_arrays if n != "clean"]
    if not degraded:
        return float(auc_clean), float("nan")
    pooled_p = np.concatenate([probs_by_view[n] for n in degraded])
    pooled_y = np.concatenate([view_arrays[n][1] for n in degraded])
    return float(auc_clean), float(roc_auc_score(pooled_y, pooled_p))


def _write_eval_grid_csv(path: Path, head, pipeline, backbone_key, val_arrays, *, trained_views, threshold) -> None:
    import csv

    head.eval()
    rows = []
    with torch.no_grad():
        for view, (emb, labels) in sorted(val_arrays.items()):
            x = torch.from_numpy(pipeline.transform({backbone_key: emb}))
            probs = torch.sigmoid(head(x).squeeze(-1)).numpy()
            auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
            preds = probs >= threshold
            bacc = float(0.5 * ((preds[labels == 1]).mean() + (~preds[labels == 0]).mean())) if len(set(labels.tolist())) > 1 else float("nan")
            rows.append([view, "trained" if view in trained_views else "held-out", f"{auc:.6f}", f"{bacc:.6f}"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "trained", "auc", "bacc_at_threshold"])
        w.writerows(rows)


def _write_threshold_sweep_csv(path: Path, head, pipeline, embeddings_dir: Path) -> None:
    """The `scripts/export_eval_stats.py` threshold-sweep table, per run --
    folded in here rather than kept as a separate script (see module
    docstring / the tier's task list)."""
    import csv

    head.eval()
    try:
        probs, labels, _idx = pooled_wildrf_scores(head, pipeline, embeddings_dir)
    except SystemExit as exc:
        logger.info("skipping threshold_sweep.csv: %s", exc)
        return
    rows = []
    for t in GRID:
        pred = probs >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        rows.append([f"{t:.4f}",
                     f"{float((probs[labels == 0] >= t).mean()):.6f}",
                     f"{float((probs[labels == 1] >= t).mean()):.6f}",
                     f"{2 * tp / max(1, 2 * tp + fp + fn):.6f}"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "fpr", "tpr", "f1"])
        w.writerows(rows)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _new_run_dir(name: str) -> Path:
    run_id = f"{name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# -- finetune seam (scaffold only) --------------------------------------------


def _run_finetune(cfg: dict, *, out: Path | None, log_dir: Path | None) -> dict:
    """The seam for `trainer: {kind: finetune}`. Not implemented.

    A real implementation reads images (`data.dataset.ManifestImageDataset` +
    `data.transforms.build_backbone_transform`, the same modules `embed.views`
    already uses for the clean pass), unfreezes the backbone returned by
    `registry.backbones.load_backbone`, and trains end to end -- sharing the
    dataset/transform code with the probe path is what makes the eval side
    (`evaluation.grid`) usable unchanged for either trainer. None of the loop
    itself is written: an unrun training loop that "mostly works" is a worse
    thing to hand off than a clear stop here. See this tier's report for
    exactly how far this went.
    """
    raise NotImplementedError(
        "[experiment] trainer.kind == 'finetune' is a scaffolded seam, not a working trainer. "
        "The probe path (frozen backbone + FeaturePipeline + head) is what this project ships; "
        "see train.experiment's module docstring for what a finetune implementation would need."
    )


# -- entry point ----------------------------------------------------------------


def run_experiment(name_or_path: str | Path, *, out: str | Path | None = None,
                    log_dir: str | Path | None = None) -> dict:
    """Load `name_or_path`, dispatch on `trainer.kind`, return the result dict
    `_run_probe`/`_run_finetune` produce. `out`, if given, additionally copies
    the bundle there (e.g. `models/<name>.pt`) on top of the run directory's
    own copy. `log_dir`, if given, additionally writes the per-step training
    curves (folds in `scripts/train_instrumented.py`)."""
    cfg = load_experiment(name_or_path)
    print(f"[experiment] {cfg['name']}  config_hash={cfg['config_hash']}  "
          f"trainer={cfg['resolved']['trainer']['kind']}")
    kind = cfg["resolved"]["trainer"]["kind"]
    out_path = Path(out) if out is not None else None
    log_path = Path(log_dir) if log_dir is not None else None
    if kind == "probe":
        return _run_probe(cfg, out=out_path, log_dir=log_path)
    return _run_finetune(cfg, out=out_path, log_dir=log_path)
