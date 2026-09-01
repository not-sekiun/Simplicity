"""Render the presentation charts from the CSVs under stats/.

Reads what scripts/train_instrumented.py and scripts/export_eval_stats.py wrote
and exports one PNG per chart to stats/charts/. Nothing is computed here -- if a
number on a chart looks wrong, it is wrong in the CSV, and the CSV is the thing
to fix. That separation is deliberate: the CSVs are the table view of every
chart, which is also what makes the palette's low-contrast slot legible.

Palette: the validated default categorical order, first three slots
(blue / orange / aqua). Validated all-pairs for light mode -- worst CVD dE 9.2,
worst normal-vision dE 24.0. Aqua sits below 3:1 on the light surface, so every
aqua series carries a direct label rather than relying on the legend alone.
Charts are capped at three series for that reason; a fourth would fail the
all-pairs floor.

Usage:
    uv sync --extra viz
    uv run python scripts/plot_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "stats"
OUT = STATS / "charts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
SHIP_THR = 0.980


def style(ax, title, xlabel, ylabel, subtitle=None):
    """Recessive axes, no chartjunk. Grid sits behind the marks."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        pad = 10
        if subtitle:
            pad = 24 + 13 * subtitle.count(chr(10))   # one extra line of headroom per newline
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=pad)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5,
                va="bottom", linespacing=1.5)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] {name}")


def chart_loss():
    d = pd.read_csv(STATS / "train_loss_steps.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(d["step"], d["batch_loss"], color=MUTED, linewidth=0.5, alpha=0.35, label="per-batch loss")
    # trailing_mean, NOT running_mean. running_mean is the cumulative mean within
    # an epoch: it restarts at each boundary and is still weighted by the opening
    # steps, so plotting it draws one line out of two different statistics and
    # manufactures a loss drop at epoch 2 that never happened. See the comment in
    # train_instrumented.py; the window width lives there as TRAILING_WINDOW.
    ax.plot(d["step"], d["trailing_mean"], color=S1, linewidth=2.0, label="100-step trailing mean")
    # Each epoch's final running_mean IS that epoch's reported train loss -- read
    # off the column, not recomputed -- drawn flat across the epoch it summarises.
    for e in sorted(d["epoch"].unique()):
        rows = d[d["epoch"] == e]
        m = float(rows["running_mean"].iloc[-1])
        ax.plot([rows["step"].min(), rows["step"].max()], [m, m], color=S2,
                linewidth=1.4, linestyle=(0, (5, 3)),
                label="epoch mean" if e == 1 else None)
        ax.text(rows["step"].max(), m, f" {m:.4f}", color=S2, fontsize=9, va="center", ha="left")
    for e in sorted(d["epoch"].unique())[1:]:
        x = d[d["epoch"] == e]["step"].min()
        ax.axvline(x, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(x, ax.get_ylim()[1], f"  epoch {e}", color=INK_2, fontsize=9, va="top")
    # Read from run_meta.json, never hardcoded: these three numbers all change
    # when the training recipe does, and a stale caption is indistinguishable
    # from a correct one at a glance.
    meta = json.loads((STATS / "run_meta.json").read_text())
    style(ax, "Training loss", "step", "BCE loss (pos_weight balanced)",
          f"{meta['n_rows']:,} rows = {meta['n_images']:,} images x "
          f"{len(meta['train_views'])} augmentation views. 1,025 trainable parameters.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "01_training_loss.png")


def chart_val_curve():
    d = pd.read_csv(STATS / "val_curve.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(d["step"], d["auc_clean"], color=S1, linewidth=2.0, label="AUC clean")
    ax.plot(d["step"], d["auc_robust"], color=S2, linewidth=2.0, label="AUC robust (18-view pooled)")
    # The first ~50 steps climb from 0.53; plotting them flattens the part that
    # carries the decision. Clip to the informative band and say so.
    lo = min(d["auc_robust"].iloc[2:].min(), d["auc_clean"].iloc[2:].min()) - 0.004
    ax.set_ylim(lo, 1.0015)
    ends = d.groupby("epoch")["step"].max()
    for e, xs in ends.items():
        row = d[d["step"] == xs].iloc[0]
        ax.scatter([xs], [row["auc_robust"]], s=74, facecolor=SURFACE,
                   edgecolor=S2, linewidth=2.0, zorder=5)
        ax.annotate(f"end epoch {e}: {row['auc_robust']:.4f}",
                    (xs, row["auc_robust"]), textcoords="offset points",
                    xytext=(-10, -22), color=INK_2, fontsize=9, ha="right")
    for e in sorted(d["epoch"].unique())[1:]:
        ax.axvline(d[d["epoch"] == e]["step"].min(), color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(f"{d.iloc[-1]['auc_clean']:.4f}", (d.iloc[-1]["step"], d.iloc[-1]["auc_clean"]),
                textcoords="offset points", xytext=(6, 0), color=INK_2, fontsize=9, va="center")
    # The honest caption. On the 11-view head this chart showed epoch 2 lowering
    # AUC_robust, and that WAS the argument for --epochs 1. On the 19-view head it
    # does not: val prefers epoch 2 by +0.0006. The argument still holds, but the
    # evidence for it is no longer in this picture -- it is on the held-out tiers,
    # read from ablation_arms.csv so this caption cannot drift from the data.
    arms = pd.read_csv(STATS / "ablation_arms.csv")

    def _d3(arm):
        row = arms[(arms["arm"] == arm) & (arms["metric"] == "recall_18view_at_matched_fpr")]
        return float(row["value"].iloc[0]) if len(row) else float("nan")

    e1, e2 = _d3("SHIPPED (19 views, 1 epoch)"), _d3("same, 2 epochs (rejected)")
    style(ax, "Validation AUC during training", "step", "AUC",
          "Validation cannot see the reason to stop at 1 epoch -- it prefers epoch 2."
          + chr(10)
          + f"Held-out DALL-E 3 recall disagrees: {e1:.3f} after epoch 1, {e2:.3f} after epoch 2. "
          + "Y-axis clipped; the first ~50 steps climb from 0.53.")
    # Both curves now sit in the top third (clean ~0.999, robust ~0.983), so the
    # empty band is the bottom -- centre-right collided with the epoch-2 marker.
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "02_validation_auc.png")


def chart_per_view():
    d = pd.read_csv(STATS / "per_view_auc.csv")
    d = d[d["tier"].isin(["ood", "wildrf", "dalle3"])]      # 3 series: all-pairs safe
    wide = d.pivot(index="view", columns="tier", values="auc")
    trained = d.drop_duplicates("view").set_index("view")["trained"]
    # ascending puts the weakest at index 0; invert_yaxis then draws index 0 at
    # the TOP. (Descending + invert cancel out and put the strongest on top.)
    wide = wide.sort_values("ood", ascending=True)
    y = list(range(len(wide)))
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    for tier, color, label in [("ood", S1, "OOD (10 unseen generators)"),
                               ("wildrf", S2, "WildRF (real social media)"),
                               ("dalle3", S3, "DALL-E 3 (held out)")]:
        ax.scatter(wide[tier], y, s=62, color=color, label=label, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v}{'' if trained[v] else '  (held out)'}" for v in wide.index], fontsize=9)
    for lbl, v in zip(ax.get_yticklabels(), wide.index, strict=False):
        if not trained[v]:
            lbl.set_color(MUTED)
    # aqua is below 3:1 on this surface -> direct label satisfies the relief rule.
    # Anchored to the weakest row (now the top one), where there is whitespace.
    ax.annotate("DALL-E 3", (wide["dalle3"].iloc[0], 0), textcoords="offset points",
                xytext=(10, 0), color=INK_2, fontsize=9, va="center")
    ax.set_xlim(min(0.85, wide.min().min() - 0.01), 1.004)
    # Counted from the data, not written down: the trained/held-out split is a
    # property of the shipping checkpoint and changed silently when the head did.
    n_held = int((~trained.reindex(wide.index).astype(bool)).sum())
    style(ax, "Robustness profile: AUC per degradation view", "AUC", "",
          f"{n_held} of {len(wide)} views were never trained on (grey, marked). "
          f"Weakest views at the top.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    bbox_to_anchor=(0.0, -0.06), ncol=3)
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "03_robustness_per_view.png")


def chart_threshold():
    d = pd.read_csv(STATS / "threshold_sweep.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(d["threshold"], d["tpr"], color=S1, linewidth=2.0, label="TPR (AI images caught)")
    ax.plot(d["threshold"], d["fpr"], color=S2, linewidth=2.0, label="FPR (real photos wrongly flagged)")
    ax.axvline(SHIP_THR, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    row = d.iloc[(d["threshold"] - SHIP_THR).abs().idxmin()]
    ax.scatter([row["threshold"]] * 2, [row["tpr"], row["fpr"]], s=70, facecolor=SURFACE,
               edgecolor=[S1, S2], linewidth=2.0, zorder=5)
    ax.annotate(f"shipping threshold {SHIP_THR}\nTPR {row['tpr']:.3f} / FPR {row['fpr']:.3f}",
                (SHIP_THR, 0.55), textcoords="offset points", xytext=(-12, 0),
                color=INK, fontsize=9.5, ha="right", fontweight="bold")
    style(ax, "Why the threshold is 0.980, not 0.5", "decision threshold", "rate",
          "WildRF, pooled over the CDN-like views a browser extension actually sees.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="center left")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "04_threshold_sweep.png")


def chart_generators():
    d = pd.read_csv(STATS / "generator_recall.csv").sort_values("recall")
    colors = [S2 if f == "gan" else S1 for f in d["family"]]
    held = d["generator"].str.contains("held out")
    colors = [S3 if h else c for c, h in zip(colors, held, strict=False)]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(range(len(d)), d["recall"], color=colors, height=0.68, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{g}  ({e})" for g, e in zip(d["generator"], d["era"], strict=False)], fontsize=9)
    for i, (v, h) in enumerate(zip(d["recall"], held, strict=False)):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.5,
                color=INK if h else INK_2, fontweight="bold" if h else "normal")
    ax.set_xlim(0, 1.12)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=c, label=lbl)
               for c, lbl in [(S1, "diffusion"), (S2, "GAN (out of scope)"), (S3, "DALL-E 3 -- held out")]]
    style(ax, "Per-generator recall at the shipping threshold", "recall", "",
          "Our weakest generators are the OLDEST. The newest one, never trained on, is at 0.99.")
    leg = ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "05_generator_recall.png")


def chart_ablation():
    d = pd.read_csv(STATS / "ablation_arms.csv")
    order = ["trainext (previous)", "+modern, with SD3", "+modern, SD3 removed",
             "11 views (previous ship)", "22 views (all transforms)", "SHIPPED (19 views, 1 epoch)"]
    order = [a for a in order if a in set(d["arm"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = [("dalle3", "recall_18view_at_matched_fpr", S1,
               "DALL-E 3 recall, 18-view (held out)", "higher is better"),
              ("ood", "recall_at_matched_fpr", S2,
               "OOD legacy-generator recall", "higher is better")]
    for ax, (tier, metric, color, title, hint) in zip(axes, panels, strict=False):
        sub = d[(d["tier"] == tier) & (d["metric"] == metric)].set_index("arm").reindex(order)
        bars = ax.barh(range(len(sub)), sub["value"], color=color, height=0.62, zorder=3)
        best = sub["value"].idxmax()
        for i, (arm, v) in enumerate(sub["value"].items()):
            bars[i].set_alpha(1.0 if arm == best else 0.55)
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=9.5,
                    color=INK if arm == best else INK_2,
                    fontweight="bold" if arm == best else "normal")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=9.5)
        ax.set_xlim(0, 1.16)
        ax.invert_yaxis()
        style(ax, title, hint, "")
    fig.suptitle("What each training arm bought", color=INK, fontsize=14,
                 fontweight="bold", x=0.008, y=1.10, ha="left")
    fig.text(0.008, 1.02,
             "Every arm held at the SAME 2.5% false-positive rate, so recall is the only free variable. "
             "Modern data buys DALL-E 3 recall and costs legacy recall; view coverage buys both, "
             "but only up to a point -- the 22-view arm trains on everything and is worse than 19.",
             color=INK_2, fontsize=9.5, transform=fig.transFigure)
    fig.tight_layout()
    save(fig, "06_ablation_arms.png")


def chart_robustness_summary():
    """Deliverable 5.5.4: clean vs transformed, compact, all four tiers."""
    d = pd.read_csv(STATS / "robustness_summary.csv")
    labels = {"ood": "OOD  (10 unseen generators)",
              "demo_val": "demo_val  (the brief's benchmark)",
              "wildrf": "WildRF  (real social media)",
              "dalle3": "DALL-E 3  (held out, unseen)"}
    order = ["demo_val", "ood", "wildrf", "dalle3"]
    conds = [("clean", S1, "clean"),
             ("transformed_mean", S2, "transformed (mean of 17 views)"),
             ("transformed_worst", S3, "worst single transform")]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    h = 0.26
    for j2, (cond, color, label) in enumerate(conds):
        vals = [d[(d.tier == t) & (d.condition == cond)]["auc"].iloc[0] for t in order]
        ys = [i2 + (j2 - 1) * h for i2 in range(len(order))]
        ax.barh(ys, vals, height=h * 0.92, color=color, label=label, zorder=3)
        for yy, v, t in zip(ys, vals, order, strict=False):
            extra = ""
            if cond == "transformed_worst":
                extra = "  " + d[(d.tier == t) & (d.condition == cond)]["worst_view"].iloc[0]
            ax.text(v + 0.002, yy, f"{v:.4f}{extra}", va="center", fontsize=8.6, color=INK_2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[t] for t in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.85, 1.045)
    style(ax, "Robustness: clean vs transformed images", "AUC", "",
          "18 views per tier; 11 of them never trained on. Worst view is heavy noise everywhere.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, -0.08), ncol=3)
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, "07_robustness_summary.png")


def main() -> None:
    missing = [f for f in ["train_loss_steps.csv", "val_curve.csv", "per_view_auc.csv",
                           "threshold_sweep.csv", "generator_recall.csv", "ablation_arms.csv",
                           "robustness_summary.csv"]
               if not (STATS / f).exists()]
    if missing:
        sys.exit(f"[plot] missing {', '.join(missing)} -- run train_instrumented.py "
                 f"and export_eval_stats.py first.")
    chart_loss()
    chart_val_curve()
    chart_per_view()
    chart_threshold()
    chart_generators()
    chart_ablation()
    chart_robustness_summary()
    print(f"[plot] charts -> {OUT}")


if __name__ == "__main__":
    main()
