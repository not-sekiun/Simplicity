"""Instrumented replica of the SHIPPING training run, for presentation stats.

WHY THIS EXISTS. `train_head_on_views` prints one line per epoch. That is the
right amount of noise for a training loop and the wrong amount of data for a
chart: two points do not make a curve. This runs the identical recipe and
records the inside of it -- per-step loss, and validation AUC re-measured every
`--eval-every` steps -- so the loss curve and the AUC curve are real measured
series rather than two endpoints joined by a line.

IT IS A REPLICA, NOT A REIMPLEMENTATION. Everything that defines the run is
imported from the same modules the real trainer uses: `load_view_cache` (so the
same fingerprint checks fire), `build_head`, `RANDOM_SEED`, and the same
optimizer / loss / standardizer construction copied verbatim from
`train_head.train_head_on_views`. If the two ever drift, the fingerprint checks
and the final-epoch numbers printed here will disagree with the shipping run,
which is the signal to re-sync.

The checkpoint it writes goes to stats/ and is NOT a ship candidate -- it exists
so the numbers on the charts are attributable to a specific set of weights.

Usage:
    uv run python scripts/train_instrumented.py
    uv run python scripts/train_instrumented.py --epochs 2 --eval-every 25

Outputs (CSV, under stats/):
    train_loss_steps.csv   step, epoch, batch_loss, running_mean, trailing_mean
    val_curve.csv          step, epoch, auc_clean, auc_robust, score
    run_meta.json          the exact configuration these curves came from
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigc_detect.config import (  # noqa: E402
    EMBEDDINGS_DIR,
    LABEL_AIGC,
    MIDJOURNEY_V6_MANIFEST,
    NANO_BANANA_MANIFEST,
    RANDOM_SEED,
    SID_REAL_MANIFEST,
    TRAIN_EXT_MANIFEST,
    UNSPLASH_REAL_MANIFEST,
    VAL_MANIFEST,
)
from aigc_detect.embed import fingerprint_paths  # noqa: E402
from aigc_detect.embed_views import (  # noqa: E402
    eval_view_names,
    load_view_cache,
    select_rows,
    train_chain_view_names,
)
from aigc_detect.heads import build_head  # noqa: E402
from aigc_detect.train_head import _grid_auc  # noqa: E402
from aigc_detect.transforms import build_robustness_views  # noqa: E402

STATS_DIR = ROOT / "stats"

# Width of the plotted trailing-mean window, in steps (~2.8% of an epoch here).
# Wide enough to kill the per-batch noise, short enough that the curve still
# follows the loss instead of the run's history.
TRAILING_WINDOW = 100

# The shipping recipe, spelled out. Mirrors the --extra-train-manifest list in
# HANDOFF section A; each entry is (cache stem, manifest) exactly as main.py
# resolves it.
BACKBONE = "pe-core-l"
TRAIN_STEM = "train_ext"
VAL_STEM = "val-s2000"
EXTRA_TRAIN = [
    ("sid_real", SID_REAL_MANIFEST),
    ("unsplash_real", UNSPLASH_REAL_MANIFEST),
    ("nano_banana", NANO_BANANA_MANIFEST),
    ("midjourney_v6", MIDJOURNEY_V6_MANIFEST),
]
# EVERY severity of every degradation family + the four composed train chains,
# i.e. --with-chains --all-severities. The 19 views the shipping head actually
# trains on; only the three SCORED chains stay held out. This tracked the 11-view
# recipe until the allsev head shipped -- if it ever disagrees with the shipping
# checkpoint's own `train_views`, this file is the one that is stale, and every
# curve drawn from it describes a run nobody ships.
TRAIN_VIEWS = (
    *[v for v in eval_view_names() if not v.startswith("chain_")],
    *train_chain_view_names(),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=2,
                    help="Ship recipe is 1; default 2 here so the curve SHOWS the epoch-2 "
                         "robustness overfit that motivated shipping 1.")
    ap.add_argument("--eval-every", type=int, default=50, help="Steps between validation measurements.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-sample-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()

    STATS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, all_specs = build_robustness_views()

    def expected_fp(manifest, sample_rows=None):
        return fingerprint_paths(select_rows(manifest, sample_rows=sample_rows)["image_path"])

    # --- load exactly what the shipping run loads, with the same checks -------
    train_fp = expected_fp(TRAIN_EXT_MANIFEST)
    train_arrays = {
        v: load_view_cache(BACKBONE, TRAIN_STEM, v, all_specs[v], expected_manifest_fp=train_fp)[:2]
        for v in TRAIN_VIEWS
    }
    extra_arrays = []
    for stem, manifest in EXTRA_TRAIN:
        fp = expected_fp(manifest)
        arrays = {
            v: load_view_cache(BACKBONE, stem, v, all_specs[v], expected_manifest_fp=fp)[:2]
            for v in TRAIN_VIEWS
        }
        extra_arrays.append(arrays)
        n = len(arrays["clean"][0])
        n_real = int((arrays["clean"][1] == 0).sum())
        print(f"[instrumented] extra stem={stem} images={n:,} ({n_real:,} real / {n - n_real:,} aigc) OK")

    scored = set(eval_view_names())
    val_fp = expected_fp(VAL_MANIFEST, a.val_sample_rows)
    val_arrays = {}
    for p in sorted(EMBEDDINGS_DIR.glob(f"{BACKBONE}__{VAL_STEM}__*.npz")):
        view = p.name[len(f"{BACKBONE}__{VAL_STEM}__"): -len(".npz")]
        if view in scored:
            val_arrays[view] = load_view_cache(
                BACKBONE, VAL_STEM, view, all_specs[view], expected_manifest_fp=val_fp
            )[:2]

    # --- assemble, standardize (scaler from the BASE clean view only) --------
    clean_emb = train_arrays["clean"][0]
    mean, std = clean_emb.mean(axis=0), clean_emb.std(axis=0)
    std[std == 0] = 1.0
    all_sets = [train_arrays, *extra_arrays]
    x = np.concatenate([(arrays[v][0] - mean) / std for v in TRAIN_VIEWS for arrays in all_sets])
    y = np.concatenate([arrays[v][1] for v in TRAIN_VIEWS for arrays in all_sets]).astype(np.float32)

    n_pos = int((y == float(LABEL_AIGC)).sum())
    n_neg = len(y) - n_pos
    n_images = sum(len(s["clean"][0]) for s in all_sets)
    print(f"[instrumented] {n_images:,} images x {len(TRAIN_VIEWS)} views -> {len(x):,} rows "
          f"({n_neg:,} real / {n_pos:,} aigc)")

    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=a.batch_size, shuffle=True, generator=gen)
    head = build_head("linear", x.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    pos_weight = torch.tensor(n_neg / n_pos, device=device)  # --balance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[instrumented] --balance: pos_weight={pos_weight.item():.4f}  device={device}")

    loss_rows, val_rows = [], []
    step = 0
    # Two smoothings of the same loss, because they answer different questions.
    # `running_mean` is the CUMULATIVE mean WITHIN an epoch -- its final value is
    # the epoch's reported train loss (the 0.1275 that anchors the reproduction
    # check in HANDOFF_2 / FINDINGS 2k), so it is kept exactly as it was. It is
    # the wrong thing to plot: it restarts at every epoch boundary and still
    # carries the 0.885 opening steps, so it flattens out ABOVE the current loss
    # and then falls off a cliff when the accumulator resets -- one line drawn
    # from two different statistics, which reads as an epoch-2 improvement that
    # did not happen. `trailing_mean` is a fixed-width window over the last
    # TRAILING_WINDOW steps, continuous across epochs, so it tracks where the
    # loss ACTUALLY is at each step. That is what chart 01 plots.
    trail = deque(maxlen=TRAILING_WINDOW)
    for epoch in range(1, a.epochs + 1):
        running, seen = 0.0, 0
        for xb, yb in loader:
            head.train()
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(head(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            step += 1
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
            trail.append(float(loss.item()))
            loss_rows.append({"step": step, "epoch": epoch,
                              "batch_loss": round(float(loss.item()), 6),
                              "running_mean": round(running / seen, 6),
                              "trailing_mean": round(sum(trail) / len(trail), 6)})
            if step % a.eval_every == 0 or step == 1:
                ac, ar = _grid_auc(head, device, val_arrays, mean, std)
                val_rows.append({"step": step, "epoch": epoch,
                                 "auc_clean": round(ac, 6), "auc_robust": round(ar, 6),
                                 "score": round(0.5 * ac + 0.5 * ar, 6)})
        ac, ar = _grid_auc(head, device, val_arrays, mean, std)
        val_rows.append({"step": step, "epoch": epoch, "auc_clean": round(ac, 6),
                         "auc_robust": round(ar, 6), "score": round(0.5 * ac + 0.5 * ar, 6)})
        print(f"[instrumented] epoch {epoch}/{a.epochs}  train_loss={running / seen:.4f}  "
              f"val AUC_clean={ac:.4f}  AUC_robust={ar:.4f}  score={0.5 * ac + 0.5 * ar:.4f}")

    pd.DataFrame(loss_rows).to_csv(STATS_DIR / "train_loss_steps.csv", index=False)
    pd.DataFrame(val_rows).to_csv(STATS_DIR / "val_curve.csv", index=False)
    (STATS_DIR / "run_meta.json").write_text(json.dumps({
        "backbone": BACKBONE, "head": "linear", "train_stem": TRAIN_STEM,
        "extra_train_stems": [s for s, _ in EXTRA_TRAIN], "val_stem": VAL_STEM,
        "train_views": list(TRAIN_VIEWS), "epochs": a.epochs, "lr": a.lr,
        "batch_size": a.batch_size, "weight_decay": a.weight_decay, "seed": a.seed,
        "balance_pos_weight": round(float(pos_weight.item()), 6),
        "n_images": int(n_images), "n_rows": len(x),
        "n_real_rows": int(n_neg), "n_aigc_rows": int(n_pos),
        "steps_per_epoch": len(loader), "eval_every": a.eval_every,
        "note": "Instrumented replica for presentation stats. NOT a ship candidate.",
    }, indent=2), encoding="utf-8")

    torch.save({"state_dict": head.state_dict(), "head_kind": "linear", "backbone": BACKBONE,
                "in_dim": int(x.shape[1]), "scaler_mean": mean, "scaler_std": std,
                "note": "instrumented replica -- not a ship candidate"},
               STATS_DIR / "instrumented_head.pt")
    print(f"[instrumented] wrote {len(loss_rows):,} loss rows + {len(val_rows)} val points -> {STATS_DIR}")


if __name__ == "__main__":
    main()
