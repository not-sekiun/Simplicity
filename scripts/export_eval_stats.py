"""Export every evaluation number the presentation needs as tidy CSVs.

Companion to scripts/train_instrumented.py: that one records the inside of a
training run, this one records the outside -- what the finished heads do on each
tier. Everything is computed from the cached embeddings and the saved
checkpoints, so it is seconds of CPU and reproduces exactly.

Deliberately writes LONG/tidy CSVs (one observation per row) rather than the
wide tables that appear in DEMO.md. Wide tables are for reading; tidy is for
plotting, and scripts/plot_stats.py consumes these directly.

Usage:
    uv run python scripts/export_eval_stats.py

Outputs (CSV, under stats/):
    per_view_auc.csv     view, kind, trained, tier, auc
    threshold_sweep.csv  threshold, fpr, tpr, f1          (WildRF, CDN views)
    generator_recall.csv generator, family, era, recall, n
    platform_fpr.csv     platform, threshold, fpr, n
    ablation_arms.csv    arm, tier, metric, value
    robustness_summary.csv  tier, condition, n_views, auc, bacc, worst_view
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_detect.config import (  # noqa: E402
    DEMO_VAL_MANIFEST,
    EMBEDDINGS_DIR,
    OOD_MANIFEST,
    WILDRF_TEST_MANIFEST,
)
from aigc_detect.embed_views import eval_view_names, select_rows, train_chain_view_names  # noqa: E402
from aigc_detect.heads import build_head  # noqa: E402

STATS_DIR = ROOT / "stats"
BACKBONE = "pe-core-l"
SHIP = "pe-core-l__linear__allsev_e1.pt"
# Derived, not chosen: scripts/derive_threshold.py --head models/<SHIP>. Keep in
# lockstep with predict.DECISION_THRESHOLD -- every number below is read at it.
SHIP_THR = 0.980

# The views the shipping head trains on -- everything else in the 18-view grid
# is a held-out severity, and the charts say so.
# The shipping head trains on ALL severities of every degradation family (that is
# what "allsev" means); only the three SCORED CHAINS are held out. That makes the
# trained/held-out split on these charts much more lopsided than it was for the
# previous 11-view head, and the chains are now the only honest generalisation
# evidence on the grid -- which is exactly why they are flagged.
TRAINED_VIEWS = {v for v in eval_view_names() if not v.startswith("chain_")} | set(train_chain_view_names())
# The CDN-like views a browser extension actually sees. predict.py sweeps the
# threshold over exactly this set; keep them in sync.
CDN_VIEWS = ["clean", "jpeg_q70", "jpeg_q90", "resize_0.5x", "chain_light"]

TIERS = {"ood": ("ood-s4000", OOD_MANIFEST, 4000),
         "wildrf": ("wildrf_test", WILDRF_TEST_MANIFEST, None),
         "demo_val": ("demo_val-s2000", DEMO_VAL_MANIFEST, 2000)}

# Release year, so the charts can show the age/recall relationship. The point
# the error analysis makes is that our weakest generators are the OLDEST.
GEN_ERA = {"ProGAN": 2018, "StyleGAN": 2019, "BigGAN": 2019, "StarGAN": 2018,
           "GauGAN": 2019, "CycleGAN": 2017, "StyleGAN2": 2020, "WhichFaceIsReal": 2019,
           "ADM": 2021, "GLIDE": 2021, "DALLE2": 2022, "Midjourney": 2022,
           "SD14": 2022, "SD15": 2022, "VQDM": 2022, "Wukong": 2022, "SDXL": 2023}
GAN = {"ProGAN", "StyleGAN", "BigGAN", "StarGAN", "GauGAN", "CycleGAN", "StyleGAN2", "WhichFaceIsReal"}

# Arms are compared at a MATCHED OPERATING POINT, never at their own thresholds.
# Each head has a different score distribution, so "recall at each head's own
# threshold" is not a comparison -- a head that simply flags more will look best
# on recall while quietly costing false positives. For every arm we solve for
# the threshold giving the same WildRF false-positive rate, then read recall
# there. Same FPR for everyone; recall is then the only free variable.
MATCHED_FPR = 0.025
ARMS = {  # label -> checkpoint
    "trainext (previous)": "pe-core-l__linear__trainext.pt",
    "+modern, with SD3": "pe-core-l__linear__aigcmodern.pt",
    "+modern, SD3 removed": "pe-core-l__linear__aigcmodern_nosd3.pt",
    "11 views (previous ship)": "pe-core-l__linear__aigcmodern_nosd3_e1.pt",
    "22 views (all transforms)": "pe-core-l__linear__alltransforms_e1.pt",
    "SHIPPED (19 views, 1 epoch)": SHIP,
    # Kept in the CSV, deliberately absent from chart 06's `order`: it is not an
    # arm anyone considered shipping, it is the evidence for --epochs 1. The val
    # curve prefers it; every held-out tier does not. See chart 02.
    "same, 2 epochs (rejected)": "pe-core-l__linear__allsev_e2.pt",
}


def load(name):
    ck = torch.load(ROOT / "models" / name, map_location="cpu", weights_only=False)
    h = build_head(ck["head_kind"], ck["in_dim"])
    h.load_state_dict(ck["state_dict"])
    h.eval()
    return h, ck["scaler_mean"], ck["scaler_std"]


def probs(h, mean, std, stem, view):
    f = EMBEDDINGS_DIR / f"{BACKBONE}__{stem}__{view}.npz"
    if not f.exists():
        return None, None
    d = np.load(f)
    x = (d["embeddings"] - mean) / std
    with torch.no_grad():
        return torch.sigmoid(h(torch.from_numpy(x.astype(np.float32))).squeeze(-1)).numpy(), d["labels"]


def main() -> None:
    STATS_DIR.mkdir(exist_ok=True)
    h, mean, std = load(SHIP)
    views = eval_view_names()

    # 1. per-view AUC across tiers (the robustness profile) --------------------
    rows = []
    for tier, (stem, _m, _sr) in TIERS.items():
        for v in views:
            p, y = probs(h, mean, std, stem, v)
            if p is None:
                continue
            rows.append({"view": v, "kind": "chain" if "chain" in v else "single",
                         "trained": v in TRAINED_VIEWS, "tier": tier,
                         "auc": round(roc_auc_score(y, p), 6)})
    # DALLE3 is single-class; its "auc" pairs its fakes against WildRF reals.
    wr = select_rows(WILDRF_TEST_MANIFEST, sample_rows=None).reset_index(drop=True)["label"].to_numpy() == 0
    for v in views:
        pf, _ = probs(h, mean, std, "dalle3_holdout", v)
        pw, _ = probs(h, mean, std, "wildrf_test", v)
        if pf is None or pw is None:
            continue
        r = pw[wr]
        rows.append({"view": v, "kind": "chain" if "chain" in v else "single",
                     "trained": v in TRAINED_VIEWS, "tier": "dalle3",
                     "auc": round(roc_auc_score(np.r_[np.ones(len(pf)), np.zeros(len(r))], np.r_[pf, r]), 6)})
    pd.DataFrame(rows).to_csv(STATS_DIR / "per_view_auc.csv", index=False)

    # 2. threshold sweep on WildRF over the CDN views -------------------------
    ps, ys = [], []
    for v in CDN_VIEWS:
        p, y = probs(h, mean, std, "wildrf_test", v)
        ps.append(p)
        ys.append(y)
    p, y = np.concatenate(ps), np.concatenate(ys)
    sweep = []
    for t in np.arange(0.50, 0.9991, 0.005):
        pr = p >= t
        tp = int((pr & (y == 1)).sum())
        fp = int((pr & (y == 0)).sum())
        fn = int((~pr & (y == 1)).sum())
        sweep.append({"threshold": round(float(t), 4),
                      "fpr": round(float((p[y == 0] >= t).mean()), 6),
                      "tpr": round(float((p[y == 1] >= t).mean()), 6),
                      "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 6)})
    pd.DataFrame(sweep).to_csv(STATS_DIR / "threshold_sweep.csv", index=False)

    # 3. per-generator recall at the shipping threshold -----------------------
    o = select_rows(OOD_MANIFEST, sample_rows=4000).reset_index(drop=True)
    po, yo = probs(h, mean, std, "ood-s4000", "clean")
    gen = o["generator"].to_numpy()
    grows = []
    for g in sorted(set(gen[yo == 1])):
        m = (gen == g) & (yo == 1)
        grows.append({"generator": g, "family": "gan" if g in GAN else "diffusion",
                      "era": GEN_ERA.get(g), "recall": round(float((po[m] >= SHIP_THR).mean()), 6),
                      "n": int(m.sum())})
    pf, _ = probs(h, mean, std, "dalle3_holdout", "clean")
    grows.append({"generator": "DALLE3 (held out)", "family": "diffusion", "era": 2023,
                  "recall": round(float((pf >= SHIP_THR).mean()), 6), "n": int(len(pf))})
    pd.DataFrame(grows).to_csv(STATS_DIR / "generator_recall.csv", index=False)

    # 4. per-platform false positives on real photographs ---------------------
    w = select_rows(WILDRF_TEST_MANIFEST, sample_rows=None).reset_index(drop=True)
    pw, yw = probs(h, mean, std, "wildrf_test", "clean")
    prows = []
    for plat in ["reddit", "twitter", "facebook"]:
        m = w["generator"].astype(str).str.contains(plat).to_numpy() & (yw == 0)
        for t in [0.5, SHIP_THR]:
            prows.append({"platform": plat, "threshold": t,
                          "fpr": round(float((pw[m] >= t).mean()), 6), "n": int(m.sum())})
    for t in [0.5, SHIP_THR]:
        prows.append({"platform": "all", "threshold": t,
                      "fpr": round(float((pw[yw == 0] >= t).mean()), 6), "n": int((yw == 0).sum())})
    pd.DataFrame(prows).to_csv(STATS_DIR / "platform_fpr.csv", index=False)

    # 5. the ablation, at a matched false-positive rate -----------------------
    arows = []
    for label, ckname in ARMS.items():
        if not (ROOT / "models" / ckname).exists():
            print(f"[eval-stats] skipping missing arm: {ckname}")
            continue
        hh, mm, ss = load(ckname)
        # threshold that puts THIS head at MATCHED_FPR on WildRF reals, pooled
        # over the CDN views -- the same pool predict.py sweeps over.
        rp = np.concatenate([probs(hh, mm, ss, "wildrf_test", v)[0][
            probs(hh, mm, ss, "wildrf_test", v)[1] == 0] for v in CDN_VIEWS])
        thr = float(np.quantile(rp, 1.0 - MATCHED_FPR))
        arows.append({"arm": label, "tier": "wildrf", "metric": "matched_threshold",
                      "value": round(thr, 6)})
        for tier, (stem, _m, _sr) in TIERS.items():
            p, y = probs(hh, mm, ss, stem, "clean")
            arows.append({"arm": label, "tier": tier, "metric": "clean_auc",
                          "value": round(roc_auc_score(y, p), 6)})
        pfk, _ = probs(hh, mm, ss, "dalle3_holdout", "clean")
        arows.append({"arm": label, "tier": "dalle3", "metric": "recall_at_matched_fpr",
                      "value": round(float((pfk >= thr).mean()), 6)})
        d3 = [float((probs(hh, mm, ss, "dalle3_holdout", v)[0] >= thr).mean()) for v in views]
        arows.append({"arm": label, "tier": "dalle3", "metric": "recall_18view_at_matched_fpr",
                      "value": round(float(np.mean(d3)), 6)})
        po2, yo2 = probs(hh, mm, ss, "ood-s4000", "clean")
        arows.append({"arm": label, "tier": "ood", "metric": "recall_at_matched_fpr",
                      "value": round(float((po2[yo2 == 1] >= thr).mean()), 6)})
    pd.DataFrame(arows).to_csv(STATS_DIR / "ablation_arms.csv", index=False)

    # 6. the compact clean-vs-transformed summary (deliverable 5.5.4) ---------
    # AUC is threshold-free and is the headline; balanced accuracy at the SHIPPING
    # threshold is included because a deployed detector has one threshold and the
    # brief asks about performance, not ranking alone.
    def bacc(pr, yy, t):
        pos, neg = pr[yy == 1], pr[yy == 0]
        return float(0.5 * ((pos >= t).mean() + (neg < t).mean()))

    srows = []
    tiers = [("ood", "ood-s4000"), ("demo_val", "demo_val-s2000"),
             ("wildrf", "wildrf_test"), ("dalle3", "dalle3_holdout")]
    wr_mask = select_rows(WILDRF_TEST_MANIFEST, sample_rows=None).reset_index(drop=True)["label"].to_numpy() == 0
    for tier, stem in tiers:
        per = {}
        for v in views:
            pv, yv = probs(h, mean, std, stem, v)
            if pv is None:
                continue
            if tier == "dalle3":   # single class -> pair against WildRF reals
                pw, _ = probs(h, mean, std, "wildrf_test", v)
                r = pw[wr_mask]
                yy = np.r_[np.ones(len(pv)), np.zeros(len(r))]
                pp = np.r_[pv, r]
            else:
                yy, pp = yv, pv
            per[v] = (roc_auc_score(yy, pp), bacc(pp, yy, SHIP_THR))
        clean_auc, clean_bacc = per["clean"]
        deg = {k: val for k, val in per.items() if k != "clean"}
        worst_v = min(deg, key=lambda k: deg[k][0])
        single = [val[0] for k, val in deg.items() if "chain" not in k]
        chained = [val[0] for k, val in deg.items() if "chain" in k]
        srows += [
            {"tier": tier, "condition": "clean", "n_views": 1,
             "auc": round(clean_auc, 6), "bacc_at_threshold": round(clean_bacc, 6), "worst_view": ""},
            {"tier": tier, "condition": "transformed_mean", "n_views": len(deg),
             "auc": round(float(np.mean([v2[0] for v2 in deg.values()])), 6),
             "bacc_at_threshold": round(float(np.mean([v2[1] for v2 in deg.values()])), 6), "worst_view": ""},
            {"tier": tier, "condition": "transformed_single_mean", "n_views": len(single),
             "auc": round(float(np.mean(single)), 6), "bacc_at_threshold": "", "worst_view": ""},
            {"tier": tier, "condition": "transformed_chained_mean", "n_views": len(chained),
             "auc": round(float(np.mean(chained)), 6), "bacc_at_threshold": "", "worst_view": ""},
            {"tier": tier, "condition": "transformed_worst", "n_views": 1,
             "auc": round(deg[worst_v][0], 6), "bacc_at_threshold": round(deg[worst_v][1], 6),
             "worst_view": worst_v},
        ]
    pd.DataFrame(srows).to_csv(STATS_DIR / "robustness_summary.csv", index=False)

    for f in sorted(STATS_DIR.glob("*.csv")):
        print(f"[eval-stats] {f.name:<24} {len(pd.read_csv(f)):>6} rows")


if __name__ == "__main__":
    main()
