"""Score a trained head across every cached robustness view (deliverable 5.5.4).

Consumes the per-view embedding caches written by `embed_views.py` and reports,
for one (backbone, head, manifest) triple:

  * per-view ROC-AUC and balanced accuracy, for all 18 views
  * the chained views broken out separately from the single-transform ones
  * three candidate definitions of AUC_robust, and the competition-style
    `0.5*AUC_clean + 0.5*AUC_robust` under each
  * the robustness gap: clean balanced accuracy minus mean degraded balanced
    accuracy, at a single fixed threshold

No GPU work: everything here is a few matrix multiplies over cached vectors.

TWO REPORTING CHOICES, both deliberate, because both are easy to get wrong in
the flattering direction:

1. **One fixed threshold, chosen on clean, applied to every view.** Re-tuning
   the threshold per view is the standard way to make a fragile detector look
   robust: AUC (a ranking measure) can hold up perfectly while the decision
   boundary drifts far enough that accuracy at any *fixed* operating point
   collapses. This project has already seen exactly that failure -- FINDINGS
   section 2 records AUC 0.7435 alongside balanced accuracy 0.5047, i.e. real
   ranking signal with the threshold in completely the wrong place. A deployed
   detector has one threshold, so the report uses one. `bacc@0.5` is shown
   beside it for reference.

2. **AUC_robust is reported three ways, not one.** They disagree by
   construction and the disagreement is informative:

     pooled  one AUC over every degraded view's scores concatenated. The
             strictest: it additionally penalizes score-scale drift *between*
             views, since a clean-confident score and a blurred score must be
             mutually rankable.
     mean    the average of per-view AUCs. Each view is internally ranked, so
             cross-view drift is invisible to it. Most forgiving.
     worst   the minimum per-view AUC. What an adversary who picks the
             transform gets.

   Pooled is the recommended primary (see FINDINGS section 7), but publishing
   one number computed three ways is what keeps the choice honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from aigc_detect.config import RANDOM_SEED, ROOT_DIR
from aigc_detect.embed import fingerprint_paths
from aigc_detect.embed_views import cache_stem, select_rows, view_embeddings_path, view_fingerprint
from aigc_detect.heads import build_head
from aigc_detect.transforms import build_robustness_views, chain_view_names


def _best_balanced_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on the CLEAN view, then frozen for
    every other view (module docstring, choice 1).

    Returns the MIDPOINT OF THE DECISION MARGIN, not `thresholds[argmax]`.
    This matters precisely here, because clean is near-perfectly separable on
    this data (val AUC 0.9996). `roc_curve` only emits thresholds at observed
    score values, so under perfect separation the whole open interval
    (highest real score, lowest AIGC score] is equally optimal but is
    represented by a single index -- and `argmax` returns its top end, landing
    the boundary flush against the lowest-scoring AIGC image. Every degraded
    view would then be measured at an operating point where *any* downward
    score drift is an immediate false negative, reporting a robustness cliff
    manufactured by the threshold rule rather than by the model.

    Taking the midpoint of that interval is the same optimum on clean, with
    the margin split evenly between the classes.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    finite = np.isfinite(thresholds)  # roc_curve prepends inf
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    bacc = (tpr + (1.0 - fpr)) / 2.0

    # thresholds are descending, so the optimal interval runs from the highest
    # equally-optimal threshold down to (exclusive) the next distinct score.
    optimal = np.flatnonzero(bacc >= bacc.max() - 1e-12)
    upper = float(thresholds[optimal[0]])
    j = int(optimal[-1]) + 1
    lower = float(thresholds[j]) if j < len(thresholds) else float(min(probs.min(), upper))
    return (upper + lower) / 2.0


def _metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    pos, neg = labels == 1, labels == 0
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "bacc_half": float(balanced_accuracy_score(labels, (probs >= 0.5).astype(np.int64))),
        "bacc": float(balanced_accuracy_score(labels, preds)),
        "tpr": float(preds[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(preds[neg].mean()) if neg.any() else float("nan"),
    }


def evaluate_grid(
    backbone_key: str,
    manifest_path: str | Path,
    head_path: str | Path,
    limit: int | None = None,
    sample_rows: int | None = None,
    sample_seed: int = RANDOM_SEED,
    out_csv: str | Path | None = None,
) -> dict:
    manifest_path, head_path = Path(manifest_path), Path(head_path)
    stem = cache_stem(manifest_path, limit=limit, sample_rows=sample_rows)

    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    if ckpt.get("backbone") != backbone_key:
        raise SystemExit(
            f"[eval-grid] head {head_path.name} was trained on backbone "
            f"'{ckpt.get('backbone')}', not '{backbone_key}'. Embeddings from different "
            f"backbones are not interchangeable."
        )
    head = build_head(ckpt["head_kind"], ckpt["in_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    # The scaler is part of the model: it is the TRAIN-set mean/std the head was
    # fitted against. Recomputing it per view would standardize each view by its
    # own statistics, which silently re-centres every degradation away and makes
    # the grid report a robustness the deployed model does not have.
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)

    print(f"[eval-grid] head={head_path.name} kind={ckpt['head_kind']} in_dim={ckpt['in_dim']}")
    print(f"[eval-grid] backbone={backbone_key} manifest={manifest_path.name} cache_stem={stem}")

    # Expected fingerprint of the row selection, recomputed from the manifest as
    # it stands right now -- `main.py split` rewrites manifests in place
    # (FINDINGS trap 7), so a cache can belong to a different set of images.
    df = select_rows(manifest_path, limit=limit, sample_rows=sample_rows, sample_seed=sample_seed)
    expected_m_fp = fingerprint_paths(df["image_path"])

    _, specs = build_robustness_views()  # specs are resolution-independent
    chains = set(chain_view_names())

    rows, missing, ref_labels = [], [], None
    for name, spec in specs.items():
        path = view_embeddings_path(backbone_key, stem, name)
        if not path.exists():
            missing.append(name)
            continue
        with np.load(path, allow_pickle=True) as d:
            if str(d["manifest_fingerprint"]) != expected_m_fp:
                raise SystemExit(
                    f"[eval-grid] STALE: {path.name} was computed from a different row "
                    f"selection than {manifest_path.name} yields now. Re-run embed-views."
                )
            if "view_fingerprint" not in d or str(d["view_fingerprint"]) != view_fingerprint(spec):
                raise SystemExit(
                    f"[eval-grid] STALE: {path.name} was computed under a different definition "
                    f"of view '{name}' (spec is now {spec!r}). Re-run embed-views."
                )
            emb = d["embeddings"].astype(np.float32)
            labels = d["labels"].astype(np.int64)

        if ref_labels is None:
            ref_labels = labels
        elif not np.array_equal(labels, ref_labels):
            raise SystemExit(
                f"[eval-grid] view '{name}' has a different label vector than the other views -- "
                f"the caches are not row-aligned and nothing computed from them would be comparable."
            )

        with torch.no_grad():
            x = torch.from_numpy((emb - scaler_mean) / scaler_std)
            probs = torch.sigmoid(head(x).squeeze(-1)).numpy()
        rows.append({"view": name, "spec": spec, "kind": "chain" if name in chains else "single",
                     "labels": labels, "probs": probs})

    if missing:
        print(f"[eval-grid] {len(missing)} view(s) not cached, skipped: {', '.join(missing)}")
    by_name = {r["view"]: r for r in rows}
    if "clean" not in by_name:
        raise SystemExit("[eval-grid] the 'clean' view is not cached; it defines the threshold and AUC_clean.")

    clean = by_name["clean"]
    threshold = _best_balanced_threshold(clean["labels"], clean["probs"])
    print(f"[eval-grid] fixed threshold {threshold:.4f} (balanced-accuracy optimum on the clean view)")
    print(f"[eval-grid] n={len(clean['labels'])} rows, label counts "
          f"{dict(zip(*np.unique(clean['labels'], return_counts=True)))}\n")

    for r in rows:
        r.update(_metrics(r["labels"], r["probs"], threshold))

    degraded = [r for r in rows if r["view"] != "clean"]
    singles = [r for r in degraded if r["kind"] == "single"]
    chain_rows = [r for r in degraded if r["kind"] == "chain"]

    header = f"{'view':<18} {'kind':<7} {'AUC':>7} {'BAcc@t':>8} {'BAcc@0.5':>9} {'TPR':>7} {'FPR':>7}"
    print(header)
    print("-" * len(header))

    def _print(r):
        print(f"{r['view']:<18} {r['kind']:<7} {r['auc']:>7.4f} {r['bacc']:>8.4f} "
              f"{r['bacc_half']:>9.4f} {r['tpr']:>7.4f} {r['fpr']:>7.4f}")

    _print(clean)
    print("-" * len(header))
    for r in singles:
        _print(r)
    if chain_rows:
        print("-" * len(header))
        for r in chain_rows:
            _print(r)
    print("-" * len(header))

    pooled_probs = np.concatenate([r["probs"] for r in degraded]) if degraded else np.array([])
    pooled_labels = np.concatenate([r["labels"] for r in degraded]) if degraded else np.array([])
    per_view_auc = [r["auc"] for r in degraded]

    auc_clean = clean["auc"]
    robust = {
        "pooled": float(roc_auc_score(pooled_labels, pooled_probs)) if degraded else float("nan"),
        "mean": float(np.mean(per_view_auc)) if degraded else float("nan"),
        "worst": float(np.min(per_view_auc)) if degraded else float("nan"),
    }
    worst_view = min(degraded, key=lambda r: r["auc"])["view"] if degraded else None
    gap = auc_clean - robust["mean"] if degraded else float("nan")
    bacc_gap = clean["bacc"] - float(np.mean([r["bacc"] for r in degraded])) if degraded else float("nan")

    print(f"\nAUC_clean                       {auc_clean:.4f}")
    for k in ("pooled", "mean", "worst"):
        score = 0.5 * auc_clean + 0.5 * robust[k]
        label = f"AUC_robust ({k})"
        suffix = f"   [worst view: {worst_view}]" if k == "worst" else ""
        print(f"{label:<31} {robust[k]:.4f}   ->  0.5*clean+0.5*robust = {score:.4f}{suffix}")
    print(f"\nrobustness gap (AUC, clean - mean degraded)   {gap:.4f}")
    print(f"robustness gap (BAcc@t, clean - mean degraded) {bacc_gap:.4f}")

    if chain_rows:
        chain_mean = float(np.mean([r["auc"] for r in chain_rows]))
        single_mean = float(np.mean([r["auc"] for r in singles])) if singles else float("nan")
        delta = chain_mean - single_mean
        print(f"\nmean AUC, single-transform views  {single_mean:.4f}")
        print(f"mean AUC, chained views           {chain_mean:.4f}   (delta {delta:+.4f})")
        if delta < 0:
            print("Negative: composition costs more than any single axis -- the cliff a")
            print("single-transform grid cannot see.")
        else:
            print("Positive: chains score BETTER than the mean single transform. Do not read")
            print("this as robustness. It means some degradation in the chain is moving inputs")
            print("TOWARD the training domain -- check whether the clean view is the outlier")
            print("(out-of-distribution resolution or compression) rather than the chains.")

    if out_csv is None:
        out_csv = ROOT_DIR / "reports" / f"grid__{backbone_key}__{stem}__{head_path.stem}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "kind", "spec", "auc", "bacc_at_fixed_threshold", "bacc_at_0.5", "tpr", "fpr"])
        for r in rows:
            w.writerow([r["view"], r["kind"], r["spec"], f"{r['auc']:.6f}", f"{r['bacc']:.6f}",
                        f"{r['bacc_half']:.6f}", f"{r['tpr']:.6f}", f"{r['fpr']:.6f}"])
    print(f"\n[eval-grid] per-view table -> {out_csv}")

    return {
        "auc_clean": auc_clean,
        "auc_robust": robust,
        "threshold": threshold,
        "worst_view": worst_view,
        "rows": [{k: v for k, v in r.items() if k not in ("labels", "probs")} for r in rows],
        "csv": str(out_csv),
    }
