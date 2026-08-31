# stats/ — presentation data and charts

Every number on a slide should be traceable to a file here, and every file here
is regenerable from cached embeddings in under two minutes. Nothing in this
folder is used for training or model selection.

## Regenerate everything

```
uv sync --extra viz
uv run python scripts/train_instrumented.py    # training curves  (~3 min, GPU)
uv run python scripts/export_eval_stats.py     # evaluation tables (~40 s, CPU)
uv run python scripts/plot_stats.py            # charts -> stats/charts/
```

## The CSVs

| file | rows | what it holds |
|---|---|---|
| `train_loss_steps.csv` | 12,446 | per-step BCE loss + two smoothings, both epochs. `running_mean` is cumulative WITHIN an epoch (its last value is that epoch's reported train loss); `trailing_mean` is a 100-step window, continuous across the boundary, and is what chart 01 plots |
| `val_curve.csv` | 251 | val AUC (clean + 18-view pooled) measured every 50 steps |
| `per_view_auc.csv` | 72 | AUC per degradation view x 4 tiers, with a `trained` flag |
| `threshold_sweep.csv` | 100 | FPR / TPR / F1 across thresholds 0.50-0.999 on WildRF |
| `generator_recall.csv` | 18 | per-generator recall at the shipping threshold, with release year |
| `platform_fpr.csv` | 8 | false positives on real photos by platform, at 0.5 and 0.980 |
| `ablation_arms.csv` | 49 | what each training arm bought, **at a matched false-positive rate** |
| `robustness_summary.csv` | 20 | **deliverable 5.5.4** — clean vs transformed per tier, + single-vs-chained |
| `run_meta.json` | — | the exact configuration the curves came from |

These are the **table view** for every chart, which is also how the charts stay
accessible: one palette slot sits below 3:1 contrast on the light surface, so
the numbers must be readable somewhere that is not the picture.

## The charts

| file | the point it makes |
|---|---|
| `01_training_loss.png` | 796,461 rows, 1,025 trainable parameters, flat well before epoch 1 ends |
| `02_validation_auc.png` | **validation prefers epoch 2; every held-out tier does not** — why we ship 1 epoch |
| `03_robustness_per_view.png` | 3 of 18 views never trained on (the scored chains); weakest is heavy noise |
| `04_threshold_sweep.png` | why the threshold is 0.980 and not 0.5 |
| `05_generator_recall.png` | **our weakest generators are the oldest; the newest, held out, is at 0.99** |
| `06_ablation_arms.png` | what each data arm bought, at a matched 2.5% FPR |
| `07_robustness_summary.png` | **deliverable 5.5.4** — clean vs transformed vs worst, all four tiers |

## One methodological note, because it changes a conclusion

`ablation_arms.csv` compares training arms **at a matched false-positive rate**,
not at each head's own threshold. The heads have different score distributions,
so reading recall at different thresholds is not a comparison: an arm that
simply flags more looks best on recall while quietly spending false positives.
Scored that way, the SD3-poisoned arm appeared to have the *highest* DALL-E 3
recall of any arm. Held at a common 2.5% FPR it does not, and the real
trade-off becomes visible instead — modern generator data buys DALL-E 3 recall
(0.874 -> 0.958 over 18 views) and costs
legacy OOD recall (0.918 -> 0.899).

## `instrumented_head.pt`

The checkpoint the curves came from. It reproduces the shipping run's numbers
exactly (epoch 1: `loss=0.1383`, `AUC_clean=0.9986`, `AUC_robust=0.9832`), which
is what makes the curves trustworthy — but it is **not a ship candidate**, and
the deployed head remains `models/pe-core-l__linear__allsev_e1.pt`.

## The threshold

`0.980`, and it is derived rather than chosen. `scripts/derive_threshold.py`
holds the protocol as code with the split pinned, and `--verify` asserts it
still reproduces FINDINGS 2j's recorded table. Re-run it on every head swap:

```
uv run python scripts/derive_threshold.py --head models/pe-core-l__linear__allsev_e1.pt
uv run python scripts/derive_threshold.py --verify
```
