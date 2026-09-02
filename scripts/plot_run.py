"""Render presentation charts from ONE `aigc experiment run` run directory.

Replaces `scripts/plot_stats.py`, which read `stats/*.csv` written by two
now-deleted scripts (`train_instrumented.py`, `export_eval_stats.py`). Tier 7
folded both into `train.experiment`: every run directory
(`data/runs/<run_id>/`) already carries `eval_grid.csv` and
`threshold_sweep.csv`, and passing `--log-dir` to `aigc experiment run` adds
`train_loss_steps.csv`/`val_curve.csv` in the same shape those two scripts
used to produce. Nothing is computed here -- if a number on a chart looks
wrong, it is wrong in the CSV, and the run directory is the thing to
regenerate (`aigc experiment run <name> --log-dir <dir>`), not this script.

WHAT DID NOT SURVIVE THE FOLD-IN, AND WHY. `plot_stats.py` drew three charts
this script does not: per-generator recall, the multi-arm ablation comparison,
and the four-tier robustness summary. All three are inherently CROSS-run
(they compare several checkpoints, or several evaluation tiers side by side)
where a `data/runs/<run_id>/` directory is deliberately ONE run's record.
Reconstructing them would mean either reading several run directories at once
(a different tool from "plot one run") or keeping `export_eval_stats.py`'s
multi-checkpoint ARMS table alive as a separate script -- out of scope for
this tier; see the Tier 7 report for the explicit call. What DOES survive:
the training curves (chart 1-2, when `--log-dir` was used) and a single-run
version of the per-view robustness profile and the threshold sweep (chart
3-4), which is everything a single `aigc experiment run` invocation actually
measures.

Palette: the validated default categorical order, first three slots
(blue / orange / aqua). Validated all-pairs for light mode -- worst CVD dE 9.2,
worst normal-vision dE 24.0. Aqua sits below 3:1 on the light surface, so every
aqua series carries a direct label rather than relying on the legend alone.

Usage:
    uv sync --extra viz
    uv run python scripts/plot_run.py data/runs/<run_id> [--log-dir DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e6e5e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua


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
            pad = 24 + 13 * subtitle.count(chr(10))
        ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=pad)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5,
                va="bottom", linespacing=1.5)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot-run] {name}")


def chart_loss(log_dir: Path, out_dir: Path, run_name: str) -> None:
    f = log_dir / "train_loss_steps.csv"
    if not f.exists():
        print(f"[plot-run] skip 01_training_loss.png: no {f}")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(d["step"], d["batch_loss"], color=MUTED, linewidth=0.5, alpha=0.35, label="per-batch loss")
    ax.plot(d["step"], d["trailing_mean"], color=S1, linewidth=2.0, label="100-step trailing mean")
    for e in sorted(d["epoch"].unique()):
        rows = d[d["epoch"] == e]
        m = float(rows["running_mean"].iloc[-1])
        ax.plot([rows["step"].min(), rows["step"].max()], [m, m], color=S2,
                linewidth=1.4, linestyle=(0, (5, 3)), label="epoch mean" if e == 1 else None)
        ax.text(rows["step"].max(), m, f" {m:.4f}", color=S2, fontsize=9, va="center", ha="left")
    for e in sorted(d["epoch"].unique())[1:]:
        x = d[d["epoch"] == e]["step"].min()
        ax.axvline(x, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(x, ax.get_ylim()[1], f"  epoch {e}", color=INK_2, fontsize=9, va="top")
    style(ax, f"Training loss -- {run_name}", "step", "BCE loss", f"{len(d):,} logged steps.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, out_dir, "01_training_loss.png")


def chart_val_curve(log_dir: Path, out_dir: Path, run_name: str) -> None:
    f = log_dir / "val_curve.csv"
    if not f.exists():
        print(f"[plot-run] skip 02_validation_auc.png: no {f}")
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(d["step"], d["auc_clean"], color=S1, linewidth=2.0, label="AUC clean")
    ax.plot(d["step"], d["auc_robust"], color=S2, linewidth=2.0, label="AUC robust (pooled)")
    lo = min(d["auc_robust"].iloc[2:].min(), d["auc_clean"].iloc[2:].min()) - 0.004 if len(d) > 2 else 0.0
    ax.set_ylim(lo, 1.0015)
    for e in sorted(d["epoch"].unique())[1:]:
        ax.axvline(d[d["epoch"] == e]["step"].min(), color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    style(ax, f"Validation AUC during training -- {run_name}", "step", "AUC")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, out_dir, "02_validation_auc.png")


def chart_per_view(run_dir: Path, out_dir: Path, run_name: str) -> None:
    f = run_dir / "eval_grid.csv"
    if not f.exists():
        print(f"[plot-run] skip 03_per_view_auc.png: no {f}")
        return
    d = pd.read_csv(f).sort_values("auc", ascending=True)
    y = list(range(len(d)))
    colors = [S1 if t == "trained" else MUTED for t in d["trained"]]
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.32 * len(d) + 1.2)))
    ax.scatter(d["auc"], y, s=62, color=colors, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{v}{'' if t == 'trained' else '  (held out)'}" for v, t in zip(d["view"], d["trained"], strict=True)],
        fontsize=9,
    )
    for lbl, t in zip(ax.get_yticklabels(), d["trained"], strict=True):
        if t != "trained":
            lbl.set_color(MUTED)
    ax.set_xlim(min(0.85, float(d["auc"].min()) - 0.01), 1.004)
    n_held = int((d["trained"] != "trained").sum())
    style(ax, f"Per-view AUC -- {run_name}", "AUC", "",
          f"{n_held} of {len(d)} views held out of training (grey). Weakest at the top.")
    save(fig, out_dir, "03_per_view_auc.png")


def chart_threshold(run_dir: Path, out_dir: Path, run_name: str) -> None:
    f = run_dir / "threshold_sweep.csv"
    thr_f = run_dir / "threshold.json"
    if not f.exists():
        print(f"[plot-run] skip 04_threshold_sweep.png: no {f}")
        return
    d = pd.read_csv(f)
    ship_thr = json.loads(thr_f.read_text())["threshold"] if thr_f.exists() else float(d["threshold"].median())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(d["threshold"], d["tpr"], color=S1, linewidth=2.0, label="TPR (AI images caught)")
    ax.plot(d["threshold"], d["fpr"], color=S2, linewidth=2.0, label="FPR (real photos wrongly flagged)")
    ax.axvline(ship_thr, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    row = d.iloc[(d["threshold"] - ship_thr).abs().idxmin()]
    ax.scatter([row["threshold"]] * 2, [row["tpr"], row["fpr"]], s=70, facecolor=SURFACE,
               edgecolor=[S1, S2], linewidth=2.0, zorder=5)
    ax.annotate(f"chosen threshold {ship_thr}\nTPR {row['tpr']:.3f} / FPR {row['fpr']:.3f}",
                (ship_thr, 0.55), textcoords="offset points", xytext=(-12, 0),
                color=INK, fontsize=9.5, ha="right", fontweight="bold")
    style(ax, f"Threshold sweep -- {run_name}", "decision threshold", "rate",
          "WildRF, pooled over the CDN-like views a browser extension actually sees.")
    leg = ax.legend(frameon=False, fontsize=9.5, loc="center left")
    for t in leg.get_texts():
        t.set_color(INK_2)
    save(fig, out_dir, "04_threshold_sweep.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="A data/runs/<run_id> directory written by `aigc experiment run`.")
    ap.add_argument("--log-dir", default=None,
                    help="Where --log-dir wrote train_loss_steps.csv/val_curve.csv (default: run_dir itself).")
    ap.add_argument("--out", default=None, help="Output directory for PNGs (default: <run_dir>/charts).")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"[plot-run] not a directory: {run_dir}")
    log_dir = Path(a.log_dir) if a.log_dir else run_dir
    out_dir = Path(a.out) if a.out else run_dir / "charts"
    run_name = run_dir.name

    chart_loss(log_dir, out_dir, run_name)
    chart_val_curve(log_dir, out_dir, run_name)
    chart_per_view(run_dir, out_dir, run_name)
    chart_threshold(run_dir, out_dir, run_name)
    print(f"[plot-run] charts -> {out_dir}")


if __name__ == "__main__":
    main()
