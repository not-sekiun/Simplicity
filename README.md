# AIGC Image Detection — TikTok TechJam 2026, Track 5

Robust detection of AI-generated images under real-world redistribution —
JPEG re-encoding, blur, thumbnail resize, sensor noise, color jitter,
center-cropping, and realistic chains of those. Solo submission.

> **Numbers in this README are current, not final.** The shipping head is
> trained on 6,000 of the 23,800 available training images (25% of the
> pool) — a full-pool retrain is in progress and expected to move the
> headline OOD score by roughly +0.015–0.025 (see [Status](#status) and
> [HANDOFF.md](HANDOFF.md)). The **backbone choice (`pe-core-l`) is
> locked** — decided by a controlled race against two challengers on a
> held-out OOD benchmark, see [Model](#model-frozen-backbone--linear-probe)
> — but the **checkpoint it ships with will be replaced** as more training
> data lands. Re-run `error-analysis` / `eval-grid` after that to refresh
> this file's tables.

## Status

| Piece | State |
|---|---|
| Data pipeline (download, index, split, 3 held-out eval tiers) | done |
| Robustness transform pipeline (brief's exact 5.2 table + 3 realistic chains) | done |
| Shortcut audit (blind-probe canary) | done — caught and removed two label shortcuts, see [FINDINGS.md](FINDINGS.md) |
| Frozen-backbone + linear-probe model pipeline | done |
| Backbone race (`pe-core-l` vs `dinov3-l` vs `metaclip2-h`) | **done — `pe-core-l` wins**, see [Model](#model-frozen-backbone--linear-probe) |
| Inference script, `{image_path, pred}` JSON (deliverable 5.5.2) | done — `predict.py` |
| Robustness evaluation summary, 18 views (deliverable 5.5.4) | done — `main.py eval-grid` |
| Error analysis: concrete false positives/negatives (deliverable 5.5.5) | done — `main.py error-analysis` |
| Full-pool retrain (25% → 100% of training images) | **in progress**, largest known remaining lever |
| Cross-generator training slice (`data/train_ext/`, 9 unseen generators) | **in progress** |
| `metaclip2-giant` backbone (legal, registered) | blocked on a micro-batching change, deprioritized — see [HANDOFF.md](HANDOFF.md) §0 |

Full state, open jobs, and the reasoning behind every decision above:
**[HANDOFF.md](HANDOFF.md)** (start there for anyone picking this project
back up) → [NARRATIVE.md](NARRATIVE.md) (numbered experiment log) →
[FINDINGS.md](FINDINGS.md) (forensic detail on data shortcuts and traps).

> ⚠ **A high AUC on this project is evidence of a leak, not success.**
> Two label shortcuts were found and removed from the training data (see
> [FINDINGS.md](FINDINGS.md) §1) — always cross-check any headline number
> against the out-of-distribution tier described below, not just clean
> validation accuracy.

## How this solution addresses the problem statement

- **Frozen vision-foundation-model backbone + a single linear probe**, not a
  from-scratch CNN — under the <2B-parameter cap, a large pretrained
  backbone's features already encode most of what separates real from
  generated images; a from-scratch model at hackathon scale would have to
  relearn that from a much smaller dataset. This follows *Simplicity
  Prevails* (arXiv:2602.01738), whose whole thesis is that this recipe is
  both simpler and more robust than task-specific architectures.
- **Robustness is trained for, not hoped for.** The brief's transform table
  (5.2) is implemented exactly and used two ways: as a *stochastic*
  augmentation during training (so the probe sees degraded inputs, not just
  clean ones) and as a *deterministic* 18-view grid at eval time (so
  robustness is measured per transform, per severity, not averaged away).
- **Composition, not just single transforms.** A real repost has usually
  been through several transforms at once (filtered, resized, re-uploaded).
  Three chained views (`chain_light/medium/heavy`) test that directly and
  it is the binding failure mode — see [Robustness evaluation](#robustness-evaluation-deliverable-554).
- **Generalization is measured on data the model never trained on**, not
  just held-out rows from the same pool — a dedicated out-of-distribution
  tier with 10 of 18 generators absent from training (see
  [Evaluation tiers](#evaluation-tiers)) is what actually chose the backbone
  and is what the error analysis below is built from.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU + driver (CUDA
13.x compatible; developed on an RTX 3080 / driver reporting CUDA 13.3).
CPU-only inference works but is slow for anything beyond a handful of images.

```bash
uv sync                    # creates .venv, installs torch+cu130 and all deps
uv run main.py check-env   # verify GPU is visible to PyTorch, report dataset status
```

`pyproject.toml` pins PyTorch to the `cu130` wheel index via
`[tool.uv.sources]` / `[[tool.uv.index]]`. If your GPU/driver only supports
an older CUDA, edit those two blocks to `cu126` (or your version) and
re-run `uv sync`.

**Always run code through uv** (`uv run main.py ...` / `uv run python ...`),
never a bare `python`/`pip` — this is a uv-managed project and the pinned
CUDA wheel only resolves inside the uv environment.

## Quick start: run inference on your own images

This is the required deliverable (5.5.2) — a script that takes a directory
of images and emits a confidence score per image.

```bash
uv run python predict.py --input_dir path/to/images --output preds.json
# or, equivalently:
uv run main.py predict --input_dir path/to/images --output preds.json
```

Output (`preds.json`) is a JSON array, deterministically ordered by
POSIX-relative path:

```json
[
  {"image_path": "cat.jpg", "pred": 0.0421},
  {"image_path": "subfolder/generated.png", "pred": 0.9731}
]
```

`pred` is `P(AIGC)` in `[0, 1]` — closer to 1 means more likely
AI-generated. Recurses subdirectories; accepts jpg/jpeg/png/webp/bmp;
unreadable files are skipped with a warning rather than crashing the run.
No `--head` needed — it defaults to the current shipping checkpoint
(`models/pe-core-l__linear__augchain.pt`); pass `--head <path>` to score
with a different one (e.g. after the full-pool retrain finishes).

## Model: frozen backbone + linear probe

Per *Simplicity Prevails* (arXiv:2602.01738): a single linear layer trained
on the pooled output of a **frozen** vision foundation model (AdamW, lr
1e-3, batch 128, 2 epochs). The backbone is never fine-tuned — freezing is
the mechanism the paper's robustness claim rests on, not a compute
shortcut, and it's what keeps a hackathon-scale training run feasible at
all under real GPU/time limits.

### Backbone race

Four candidate backbones were registered (ungated on HuggingFace, vision
tower only, each asserted under the 2B-parameter cap):

| backbone | native res | vision-tower params | status |
|---|---:|---:|---|
| **`pe-core-l`** | 336px | 316.1M | **ships** |
| `dinov3-l` | 256px | 303M | raced, lost |
| `metaclip2-h` | 224px | 630.8M | raced, lost |
| `metaclip2-giant` | 378px | 1,843.6M (92.2% of cap) | registered, legal, blocked on a code change (see [HANDOFF.md](HANDOFF.md) §0) |

`dinov2-g` (DINOv2-giant, 518px) is registered but was **not** raced — the
paper ranks DINOv2 second-from-bottom (0.852 GenImage / 0.636 in-the-wild)
and at 518px it is by far the most expensive to embed (~3h vs ~30min each
for the others). It was skipped on expected value, not on the parameter cap.

The race was decided on the **out-of-distribution tier** (below), not
clean validation accuracy — val is saturated (11/18 robustness views at or
above AUC 0.99, differences inside their own standard error) and cannot
rank two backbones. A decision rule was fixed *before* the numbers were
seen (switch only if a challenger beats the incumbent by more than +0.010
on `0.5·AUC_clean + 0.5·AUC_robust(pooled)`):

| backbone | OOD score | Δ vs `pe-core-l` |
|---|---:|---:|
| **`pe-core-l`** | **0.9366** | — |
| `dinov3-l` | 0.9041 | −0.0325 |
| `metaclip2-h` | 0.8961 | −0.0405 |

Same ordering held on the in-scope diffusion-only metric, on worst-case
AUC, and on clean val. Full protocol and race artifacts:
[HANDOFF.md](HANDOFF.md) §§1–2, `reports/race/`.

## Data

**Training pool: [`TheKernel01/Tiny-GenImage`](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) only** —
7 generators (ADM, BigGAN, GLIDE, Midjourney, SD1.5, VQDM, Wukong) + real
images, content-matched real/fake pairs. `train.csv` (23,800 rows) /
`val.csv` (4,200 rows), stratified 85/15.

**CIFAKE and SID_Set (both listed in the brief's dataset options) were
evaluated and dropped from training.** SID_Set carries a composition
shortcut — its real and AI halves are separable at 0.93 balanced accuracy
from an 8×8 greyscale thumbnail alone, and a head trained on it transfers
to CIFAKE at chance. CIFAKE measured as actively harmful once mixed in.
Full forensics: [FINDINGS.md](FINDINGS.md) §§1, 2d. This is a documented,
deliberate scope decision, not missing work — the brief permits choosing
among its listed datasets and states them as options, not requirements.

```bash
uv run main.py download tiny-genimage --limit-per-split 40000   # HF, streamed + capped
uv run main.py split --val-fraction 0.15 --seed 42               # stratified train/val manifests
uv run main.py audit-data --transform                             # shortcut audit + blind-probe canary
```

Every downloader writes `data/raw/<source>_index.csv`
(`image_path,label,source,generator`; label 0=real, 1=AIGC). `main.py
split` merges every `*_index.csv` under `data/raw/` into a single
stratified `data/processed/{train,val}.csv`. `data/` is gitignored — run
`uv run main.py check-env` to see what's actually on disk before assuming
anything is there.

### Evaluation tiers

Four tiers, increasingly hard, **none of them ever globbed by `split`** —
so "don't train on the eval set" is structural, not a convention that can
be forgotten:

| Tier | What it is | Discriminates? |
|---|---|---|
| `val` (internal, 15%) | Held-out split of the training pool, same 7 generators | No — saturated, 11/18 views ≥ 0.99 |
| `heldout` | Tiny-GenImage's own HF "validation" split, same 7 generators, never touched during training | in-distribution sanity check only |
| `demo-val` | The brief's self-reported benchmark (5.4): COCO val2017 (real) + WildFake "DALL·E Advanced" (AIGC). **Never trained on, never tuned against** — the brief says explicitly not to use it for training, and it doesn't score, so it's used only for periodic checkpoint sanity checks | No — saturated, 16/18 views ≥ 0.99 |
| `ood` | A deliberately hard tier built from `TheKernel01/AIGC-Detection-Benchmark`, generator-balanced across 18 classes, **10 of which are absent from training** (5 GAN families + DALL·E 2, SD1.4, SDXL, StarGAN, WhichFaceIsReal) | **Yes — 0/18 views ≥ 0.99, range 0.81–0.95.** The only tier that can currently rank a model change |

`val`/`heldout`/`demo-val` all being saturated is itself a finding worth
stating in a write-up: on in-distribution or near-in-distribution data this
approach is close to ceiling, and the interesting signal only shows up
once the generator distribution actually shifts.

```bash
uv run main.py build-heldout                                 # merge heldout index
uv run main.py download-demo coco-val2017                    # real half, auto
uv run main.py download-demo wildfake-dalle-advanced         # AIGC half, needs a manual fetch first — see --help
uv run main.py build-demo-val                                # merge + leakage guard
uv run main.py download-ood --per-generator 250               # generator-balanced OOD slice
uv run main.py build-ood                                      # merge OOD index
```

## Robustness pipeline

`src/aigc_detect/transforms.py` implements the brief's transform table
(5.2) exactly:

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG Compression | quality = 90, 70, 50, 30 | social re-encode, messaging |
| Gaussian Blur | σ = 0.5, 1.0, 2.0 | out-of-focus |
| Resize | scale 0.5× / 0.25× then upscale | thumbnail generation |
| Gaussian Noise | σ = 0.02, 0.05, 0.10 | low-light sensor noise |
| Color Jitter | brightness/contrast/sat. ±20% | filter apps, auto-enhance |
| Center Crop | crop 80% | profile-pic cropping, framing |

Plus **three chained views** the table alone can't test — nothing reaches a
real detector having survived exactly one transform:

| View | Ops | Scenario |
|---|---|---|
| `chain_light` | resize 0.5× → JPEG q70 | a single re-upload |
| `chain_medium` | crop 80% → jitter → resize 0.5× → JPEG q50 | screenshot → filter app → re-upload |
| `chain_heavy` | blur σ1.0 → resize 0.25× → noise σ0.05 → JPEG q30 | a repost of a repost |

18 evaluation views total: `clean` + 14 single-transform + 3 chained.
Training uses a separate stochastic mix of the same table (1–2 random ops,
random order, random severity) plus 4 training-only chains disjoint from
the 3 scored ones, so training on composition never contaminates the
columns that measure it.

## Robustness evaluation (deliverable 5.5.4)

```bash
# cache all 18 views for a seeded, label-balanced subsample (one decode per image)
uv run main.py embed-views --backbone pe-core-l --manifest ood --sample-rows 4000
# score the shipping head across them (no GPU work — reads the cache)
uv run main.py eval-grid --backbone pe-core-l --manifest ood --sample-rows 4000 \
    --head models/pe-core-l__linear__augchain.pt --by-generator
```

Current shipping head, clean vs. transformed, on the OOD tier
(the tier that discriminates — see [Evaluation tiers](#evaluation-tiers)):

| view | AUC | BAcc @ fixed threshold | FPR | FNR |
|---|---:|---:|---:|---:|
| clean | 0.9532 | 0.9358 | 0.088 | 0.041 |
| mean, single-transform views | 0.9219 | — | — | — |
| mean, chained views | 0.8718 | — | — | — |
| **chain_heavy (worst)** | **0.8099** | 0.7548 | 0.292 | 0.199 |

`AUC_robust` (pooled over all 14 single + 3 chained views) = **0.9200** →
`0.5·AUC_clean + 0.5·AUC_robust` = **0.9366**. Full per-view table:
`reports/race/pe-core-l/grid_ood.csv`; console output with the
per-generator breakdown: `reports/race/pe-core-l/run.log`.

**Composition compounds, it doesn't average.** `chain_heavy` scores below
every one of its own component transforms taken alone (composition penalty
−0.076 vs. its weakest single component) — the single-transform grid alone
would have missed this failure mode entirely. This is the reason chained
views are reported, not just the brief's 14 base rows.

## Error analysis (deliverable 5.5.5)

```bash
uv run main.py error-analysis --backbone pe-core-l --manifest ood --sample-rows 4000 \
    --head models/pe-core-l__linear__augchain.pt --top-k 8
```

Reads the same view caches `eval-grid` uses (no extra GPU work) and, at
**the same one fixed threshold** eval-grid reports at, writes:
`reports/error_analysis/report__*.md` (readable summary),
`examples__*.csv` + copied image files (the model's most confident
mistakes, per view), and `by_generator__*.csv` (worst-collapsing
generators first). Findings from the current shipping head on `ood`:

- **The single largest generator-level collapse is `DALLE2`** (unseen,
  diffusion, in scope): clean AUC 0.928 → 0.802 pooled-degraded, a 12.6-point
  drop — the largest of any generator (next: `StyleGAN`, a GAN, at 8.6
  points; the next diffusion generator is `ADM` at 3.3 points). Its
  most-missed images are the highest-confidence false negatives in the
  whole set (`pred` as low as 0.0015 for an actual AIGC image).
- **Unseen diffusion generators otherwise generalize well** — SD1.4 (0.958)
  and SDXL (0.955) beat even *trained* generators like ADM (0.921) and
  Midjourney (0.930). GAN families did not collapse either (mean clean GAN
  AUC 0.964 > mean clean diffusion 0.947, and the same ordering holds
  degraded) — there is no GAN-specific weakness to fix,
  and DALLE2 looks like a genuine style outlier rather than a
  seen/unseen effect in general.
- **Errors skew toward false positives under degradation.** Going from
  `clean` to `chain_heavy`, FPR moves +0.20 and FNR moves +0.16 — heavy
  degradation makes the model more likely to flag *real* images as AIGC
  than to miss actual fakes. For a deployment that penalizes false
  accusations against real users, this is the direction that matters most
  to calibrate against.
- **A specific false-positive cluster**: every top false positive on both
  `clean` and `chain_heavy` comes from `WhichFaceIsReal` (a real-photo
  benchmark for a human-perception task, labeled `real` here) — the head
  flags these portraits as AIGC with confidence 1.0000 regardless of
  transform. This is a real, unresolved generalization gap worth a
  dedicated future look, not fixable by more robustness augmentation.

## Trade-offs & limitations

- **Numbers above are provisional.** The shipping head trains on 6,000 of
  23,800 available training images; a learning-curve check found the OOD
  score still rising at that point (+0.022 from 3,000 → 6,000 images), so a
  full-pool retrain is expected to move every table in this README before
  final submission. See [HANDOFF.md](HANDOFF.md) §0 for the live status.
- **One fixed threshold, chosen on clean, used everywhere.** This is a
  deliberate choice — re-tuning per transform is the standard way to make a
  fragile detector look robust on paper (AUC can look fine while accuracy
  at any real operating point collapses) — but it means the FPR/FNR numbers
  above are specific to that one operating point; shifting it trades false
  accusations of real content against missed fakes, and which side to favor
  is a deployment decision this project doesn't make on its own.
- **`ood` is a synthetic proxy for "unknown future generators," not a
  guarantee.** It's the best generalization signal available at hackathon
  scale, but n≈4,000 (per-view SE ≈0.005–0.008) and its own generator mix
  is still finite; a genuinely novel generator family could behave
  differently again.
- **The DALLE2 collapse and the WhichFaceIsReal false-positive cluster
  (above) are both unresolved.** Given more time: (1) fold a small amount
  of DALL·E-family data into training via the `data/train_ext/` slice
  already being pulled (9 generators absent from the current pool), (2)
  investigate the WhichFaceIsReal cluster specifically rather than assuming
  it's covered by generic robustness training, (3) calibrate per-degradation
  thresholds instead of one global one, (4) unblock `metaclip2-giant` (needs
  micro-batched forward passes — it currently sends 18 images per forward
  even at batch size 1, well past its throughput cliff).
- **Frozen backbone is the whole robustness strategy.** No adversarial
  training, no explicit forensic/frequency-domain features. This keeps the
  system simple, fast to iterate, and inference-cheap (a single linear
  layer at inference time — no backprop through a foundation model), but
  it means any weakness in the backbone's own pretraining generalization
  is directly a weakness here.

## Project layout

```
main.py                       Entry-point CLI (`uv run main.py --help` for the full list)
predict.py                    Standalone inference entry point (deliverable 5.5.2)
src/aigc_detect/
  config.py                    Paths, label ids, generator/family tables, split fraction
  transforms.py                 Robustness transform pipeline (5.2 table + chains)
  dataset.py                    ManifestImageDataset: CSV manifest -> (tensor, label)
  backbones.py                  Frozen VFM registry + loader (asserts <2B params)
  embed.py / embed_views.py     Cache pooled embeddings, per-view or per-manifest
  heads.py                      LinearHead / MLPHead
  train_head.py                 Paper training recipe on cached embeddings
  eval_grid.py                  Robustness evaluation summary (5.5.4)
  error_analysis.py             False positive/negative + per-generator report (5.5.5)
  predict.py                    Inference logic shared by predict.py and `main.py predict`
scripts/
  download_data.py, download_tiny_genimage.py, download_demo_val.py,
  download_ood_benchmark.py      Dataset downloaders/indexers
  make_splits.py, make_heldout.py, make_demo_val.py, make_ood.py
                                  Manifest builders for each tier
  audit_data.py                  Shortcut audit + blind-probe canary
  run_race.py                    Backbone race runner
data/                           gitignored — check `main.py check-env` before assuming
  raw/ processed/ heldout/ demo_val/ ood/ embeddings/
models/                         Trained head checkpoints (small — a few KB each,
                                 the backbone weights are never saved, only downloaded)
reports/                        Robustness grids, race results, error analysis
```

## Development tools & stack

- Python 3.11, [uv](https://docs.astral.sh/uv/) for environment/dependency
  management.
- PyTorch 2.13 + torchvision (cu130 build), scikit-learn (AUC/balanced
  accuracy), pandas/numpy, Hugging Face `datasets`/`transformers`/`timm`
  for backbone loading and dataset streaming, kagglehub for one dataset
  mirror.
- Windows 11 + an RTX 3080 for development; VS Code + Claude Code as the
  editor/assistant.
- No paid APIs — every backbone is a public checkpoint downloaded once and
  run locally.

## Team

Solo — 1010angusx@gmail.com.
