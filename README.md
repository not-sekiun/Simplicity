# AIGC Image Detection — TikTok TechJam 2026, Track 5

Robust detection of AI-generated images under real-world redistribution —
JPEG re-encoding, blur, thumbnail resize, sensor noise, color jitter,
center-cropping, and realistic chains of those. Solo submission.

**Shipping head:** `models/pe-core-l__linear__allsev_e1.pt` at threshold
**0.980**. A frozen 316M-parameter vision backbone plus a **1,025-parameter
linear probe** — the backbone is never fine-tuned.

| tier | n | clean AUC | transformed (17-view mean) |
|---|---|---|---|
| `demo_val` — the brief's §5.4 benchmark | 13,843 | **0.9999** | 0.9960 |
| `ood` — 10 generators absent from training | 8,200 | **0.9982** | 0.9724 |
| `wildrf_test` — real Reddit/X/Facebook photos | 2,503 | **0.9969** | 0.9875 |
| `dalle3_holdout` — a modern generator held out entirely | 1,500 | **0.9988** | 0.9917 |

On real social-media photographs at the shipping threshold: **FPR 2.15% at TPR
97.97%**, measured on a held-out split of WildRF. The threshold is derived, not
chosen — `src/aigc_detect/train/calibrate.py` holds the protocol as code, runs
as the last step of every training run, and writes the result into the model
bundle, so a head and its operating point travel together.

## Status

Feature-complete. Every §5.5 deliverable is implemented and reported.

| Piece | State |
|---|---|
| Data pipeline (declared sources, resumable pulls, manifest recipes, 6 held-out eval tiers) | done |
| Robustness transform pipeline (brief's exact 5.2 table + 3 realistic chains) | done |
| Shortcut audit (blind-probe canary over every corpus) | done — and a **gate**: a corpus that clears it cannot enter a training manifest. Caught **three** mislabelled corpora, see [docs/findings.md](docs/findings.md) |
| Frozen-backbone + linear-probe model pipeline | done |
| Backbone race (`pe-core-l` vs `dinov3-l` vs `metaclip2-h`) | done — **`pe-core-l` wins** |
| Inference script, `{image_path, pred}` JSON (deliverable 5.5.2) | done — `predict.py` |
| Robustness evaluation summary, 18 views x 4 tiers (deliverable 5.5.4) | done — `main.py eval-grid`, `stats/robustness_summary.csv` |
| Error analysis: false positives/negatives (deliverable 5.5.5) | done — `main.py error-analysis` |
| Modern-generator training data + a held-out modern generator | done — see [Data](#data) |
| Presentation stats + charts | done — [`stats/`](stats/README.md) |
| `metaclip2-giant` backbone (legal, registered) | not pursued — blocked on a micro-batching change, deprioritized |

Full state and reasoning: **[docs/experiments.md](docs/experiments.md)**
(numbered experiment log) → [docs/findings.md](docs/findings.md) (forensic
detail on data faults) → [docs/archive/demo-script.md](docs/archive/demo-script.md)
(the demo/pitch script).

> **A high AUC on this project is evidence of a leak until proven otherwise.**
> Three separate corpora reached training mislabelled — depth maps as
> photographs, GAN faces as real, and real photographs as SD3 output. None was
> caught by a metric; all three were caught by scoring the corpus or looking at
> the pixels. Cross-check any headline number against `ood`, `wildrf_test` and
> `dalle3_holdout`, never clean validation alone.

## How this solution addresses the problem statement

- **Frozen vision-foundation-model backbone + a single linear probe**, not a
  from-scratch CNN — under the <2B-parameter cap, a large pretrained
  backbone's features already encode most of what separates real from
  generated images; a from-scratch model at hackathon scale would have to
  relearn that from a much smaller dataset. This follows *Simplicity
  Prevails* (arXiv:2602.01738), whose whole thesis is that this recipe is
  both simpler and more robust than task-specific architectures.
- **Robustness is trained for, not hoped for.** The brief's transform table
  (5.2) is implemented exactly. Every view is **fixed and deterministic** — the
  head trains on 11 named views and is scored on a disjoint-where-it-matters
  18-view grid, so robustness is measured per transform, per severity, never
  averaged away.
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
uv sync --all-extras --group dev   # .venv, torch+cu130, demo/viz extras, ruff+pytest
cp .env.example .env               # optional: every key has a working default
uv run aigc check-env              # verify GPU is visible, report dataset status
```

`uv sync` on its own installs just the inference dependencies. The extras are
`demo` (the local server behind the browser extension) and `viz` (chart
rendering); `--group dev` adds ruff and pytest.

`.env` is optional — an empty one is valid. It exists so you can point
`AIGC_DATA_ROOT` at another drive (the image corpora are ~24 GB), supply an
`HF_TOKEN` for gated datasets, or change which checkpoint the demo server
loads, without editing code. `aigc_detect.config.settings` is the only thing
that reads it. See `.env.example` for the full list.

`pyproject.toml` pins PyTorch to the `cu130` wheel index via
`[tool.uv.sources]` / `[[tool.uv.index]]`. If your GPU/driver only supports
an older CUDA, edit those two blocks to `cu126` (or your version) and
re-run `uv sync`.

**Always run code through uv** (`uv run aigc ...` / `uv run python ...`),
never a bare `python`/`pip` — this is a uv-managed project and the pinned
CUDA wheel only resolves inside the uv environment.

The CLI has two equivalent forms: `uv run aigc <command>` (the console entry
point) and `uv run main.py <command>` (a shim, kept because the docs and the
command reference below all use it). They dispatch to the same code.

Checks, if you are changing anything:

```bash
uv run ruff check .   # lint; currently clean
uv run pytest         # currently 220 green
```

Both run in CI on every push (`.github/workflows/ci.yml`). A fresh clone has
none of the ~24 GB of images and no trained checkpoint, so the tests that need
either skip themselves by name rather than failing — see `tests/conftest.py`.

## Quick start: run inference on your own images

This is the required deliverable (5.5.2) — a script that takes a directory of
images and emits a confidence score per image.

```bash
uv run python predict.py --input_dir path/to/images --output preds.json
# or, equivalently:
uv run main.py predict --input_dir path/to/images --output preds.json
```

Output (`preds.json`) is a JSON array, deterministically ordered by
POSIX-relative path:

```json
[
  {"image_path": "cat.jpg", "pred": 0.0002},
  {"image_path": "subfolder/generated.png", "pred": 0.9999}
]
```

`pred` is `P(AIGC)` in `[0, 1]` — closer to 1 means more likely AI-generated.
Recurses subdirectories; accepts jpg/jpeg/png/webp/bmp; unreadable files are
skipped with a warning rather than crashing the run. No `--head` needed — it
defaults to the shipping checkpoint. The run also prints how many images cleared
the decision threshold, which is the number a deployment would act on; that
threshold comes from the checkpoint's own bundle (0.980 for the shipping head),
not from a constant in this repo.

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
| `metaclip2-giant` | 378px | 1,843.6M (92.2% of cap) | registered, legal, blocked on a code change |

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

> **These are race-era absolute numbers and are not comparable to the tables
> elsewhere in this README.** The race predates both the `WhichFaceIsReal` label
> correction (which alone moved ood clean AUC 0.9670 → 0.9961) and the extended
> training pool. All three arms raced under identical conditions, so the
> *ordering* is valid and is what the decision rested on; the absolute values
> have since been superseded across the board.

Same ordering held on the in-scope diffusion-only metric, on worst-case
AUC, and on clean val. Full protocol and race artifacts: `reports/race/`.

## Data

**Training pool: 41,919 images.** Deliberately assembled from several
independent sources, because one dataset is one provenance and a detector will
learn "this source" as readily as "this generator".

| corpus | n | role |
|---|---|---|
| `train_ext` | 30,919 | Tiny-GenImage's 7 generators + 6 more, plus ImageNet-register reals |
| SID_Set reals | 4,000 | real photographs, OpenImages register |
| Unsplash | 4,000 | curated photography — a real domain ImageNet does not cover |
| Midjourney v6 | 1,500 | **modern generator (2024)**, art/illustration register |
| nano-banana | 1,500 | **modern generator (2025)**, Gemini 2.5 Flash Image |

**Adding real-image domains beat adding generators, three times running.**
+SID_Set reals +Unsplash took WildRF false positives from 33.0% to 18.3% at
unchanged recall, with no modelling change at all. Generator-side data only
started paying once the generators were ones we were actually failing on — the
two modern corpora above — which is exactly what the DALL·E 3 holdout confirms.

**CIFAKE was evaluated and dropped**; it measured as actively harmful once mixed
in. **SID_Set's paired real/fake split was also dropped** — its halves are
separable at 0.93 balanced accuracy from an 8x8 greyscale thumbnail alone. Its
*real* half was later re-added on its own as a real-domain corpus, once the
aspect-preserving resize + square crop in `transforms.py` closed the shortcut
the pairing exposed. Full forensics: [docs/findings.md](docs/findings.md).

**One corpus is quarantined, not deleted.** `gmongaras/Stable_Diffusion_3_Recaption`
is a *recaptioning* dataset — real photographs with SD3-authored captions, not
SD3 output. Training on 1,500 of them as AIGC cost DALL·E 2 recall 7 points
before it was caught. It lives in `data/quarantine/` with the full evidence and
is removed from the CLI so it cannot be pulled again. See
[docs/findings.md](docs/findings.md) §2k.

```bash
uv run main.py pull run tiny_genimage     # resumable; ends with the shortcut audit
uv run main.py manifest resolve train      # the recipe -> data/manifests/resolved/train.csv
uv run main.py audit-data --transform      # the same audit over EVERY declared corpus
```

### Evaluation tiers

Six tiers, every one flagged `never_train` in its recipe — so "don't train on
the eval set" is enforced rather than remembered. An eval-role corpus raises if
a training recipe includes it, and a trainer handed a flagged manifest raises
too (`aigc_detect.data.corpus.assert_trainable`):

| Tier | n | What it is | Discriminates? |
|---|---|---|---|
| `val` | 4,200 | Held-out split of the training pool | No — saturated |
| `heldout` | 7,000 | Tiny-GenImage's own validation split | in-distribution sanity only |
| `demo_val` | 13,843 | The brief's §5.4 benchmark: COCO val2017 (4,998–5,000 real) + WildFake DALL·E Advanced (8,843 AIGC). **Never trained on, never tuned against** | No — saturated at 0.9999 |
| `ood` | 8,200 | Generator-balanced across 18 classes, **10 absent from training** | Yes — for legacy generators |
| `wildrf_test` | 2,503 | **Real Reddit/X/Facebook images**, already platform-re-encoded | **Yes — the tier that matters** |
| `dalle3_holdout` | 1,500 | **DALL·E 3, held out of training entirely** | **Yes — the generalization test** |

`dalle3_holdout` is the tier that makes the modern-generator claim falsifiable.
Three current generators were pulled and only two trained on, so the third
answers the question no other tier can: does this generalise to a modern model
it has never seen, from a publisher it has never seen? It does — see below.

## Robustness pipeline

`src/aigc_detect/data/transforms.py` implements the brief's transform table
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

**Everything is deterministic.** Because the backbone is frozen, each image is
decoded once and every view it needs is generated, embedded and cached — so an
image contributes a fixed set of rows, not a fresh random draw per epoch. The
views that *are* random by construction (`color_jitter`, the noise views, two of
the chains) are seeded on the image's **content hash**, so the same photo gets
the same degradation in every subsample, on every machine, at any path. The
head trains on 11 fixed views: `clean` + 6 single severities + 4 training-only
chains (`trainchain_a..d`) that are **disjoint from the 3 scored chains**, so
training on composition never contaminates the columns that measure it.

(`RobustnessAugment` / `build_train_transform` implement a stochastic
live-pixel augmenter. They are **not** in the training path — the only caller is
`main.py preview-augment`, which renders a sanity-check grid.)

## Robustness evaluation (deliverable 5.5.4)

```bash
uv run main.py embed-views --backbone pe-core-l --manifest ood --sample-rows 4000
uv run main.py eval-grid  --backbone pe-core-l --manifest ood --sample-rows 4000     --head models/pe-core-l__linear__allsev_e1.pt --by-generator
```

**One fixed threshold (0.980) applied to every view.** Re-tuning per transform is
the standard way to make a fragile detector look robust on paper, so it is not
done here.

| tier | clean AUC | transformed (17-view mean) | worst transform | clean BAcc | transformed BAcc |
|---|---|---|---|---|---|
| **demo_val** | **0.9999** | 0.9960 | 0.9641 | 0.9890 | 0.9628 |
| **ood** | **0.9982** | 0.9724 | 0.8878 | 0.9490 | 0.8542 |
| **wildrf_test** | **0.9969** | 0.9875 | 0.9273 | 0.9796 | 0.9518 |
| **dalle3_holdout** | **0.9988** | 0.9917 | 0.9411 | 0.9840 | 0.9641 |

The worst view is `noise_sigma0.1` on three tiers and `chain_heavy` on `ood`, a
held-out composition. Heavy noise is trained on now — the head sees every
severity — which is why the remaining failure has moved to composition: the one
thing left in the grid that is still genuinely unseen.

**Composition compounds, it doesn't average** — the axis the brief's 14-row
table alone cannot see:

| tier | single-transform mean (14) | chained mean (3) | penalty |
|---|---|---|---|
| ood | 0.9786 | 0.9435 | **-0.0351** |
| wildrf_test | 0.9883 | 0.9836 | -0.0047 |
| demo_val | 0.9958 | 0.9970 | +0.0012 |
| dalle3_holdout | 0.9914 | 0.9932 | +0.0018 |

Negative means composition costs more than any single axis. It is *positive* on
demo_val and DALL·E 3 — there, degradation moves images toward the training
domain, so the clean view is the outlier rather than the chains.

Chart: `docs/assets/charts/07_robustness_summary.png`. Data:
`stats/robustness_summary.csv`, `stats/per_view_auc.csv`. The 18-view grid runs
on a 2,000-row sample of demo_val and a 4,000-row sample of ood; full-tier clean
AUC is unchanged.

## Error analysis (deliverable 5.5.5)

```bash
uv run main.py error-analysis --backbone pe-core-l --manifest ood --sample-rows 4000     --head models/pe-core-l__linear__allsev_e1.pt --top-k 8
```

**False positives — real photographs wrongly flagged.** Measured on WildRF's
clean view, by platform:

| | Facebook | Reddit | Twitter | overall FPR | TPR |
|---|---|---|---|---|---|
| at 0.50 | 36.3% | 14.3% | 22.6% | 19.3% | 99.5% |
| **at 0.980 (shipping)** | 5.6% | 0.8% | 4.4% | **2.4%** | **98.3%** |

This is a *calibration* gap, not a representation one — the ranking stays intact
(AUC 0.9969), the scores just shift. Which is why the threshold is 0.980 and not
0.5: an ~8x reduction in false accusations for 1.2 points of recall. Telling
someone their own photograph is fake costs far more than missing one AI image.
The threshold is derived on a **held-out** WildRF split (split by image, swept in
0.005 steps, F1-optimal on one half, reported on the other): **FPR .0215 / TPR
.9797**. That protocol is `src/aigc_detect/train/calibrate.py`; it runs at the
end of every `aigc experiment run`, and `verify_recorded_table` asserts it still
reproduces the table it was first recorded from — checked by `uv run pytest`,
not by a flag someone has to remember to pass.

**False negatives — and the most interesting result in the project.**
Per-generator recall at the shipping threshold, weakest first:

| generator | family | era | recall |
|---|---|---|---|
| ADM | diffusion | 2021 | **0.472** |
| DALLE2 | diffusion | 2022 | **0.573** |
| Midjourney (v4-era) | diffusion | 2022 | 0.694 |
| GLIDE | diffusion | 2021 | 0.830 |
| SD14 / SD15 / VQDM / Wukong | diffusion | 2022 | 0.973–0.991 |
| SDXL | diffusion | 2023 | 0.992 |
| **DALL·E 3 (held out)** | diffusion | **2023** | **0.992** |
| GAN family (8 generators) | gan | 2018–21 | 0.982 mean |

**Our weakest generators are the oldest; our best are the newest.** The 2022-era
diffusion mean is 0.865 while the modern held-out generator is 0.992. That
inversion matters more than the averages: the deployment question is not whether
this detects four-year-old research models, it is whether it holds against what
people use now and what ships next year.

**The legacy regression is recovered.** Earlier heads bought modern-generator
recall by giving up legacy recall (2022-era diffusion 0.859 → 0.811). Training on
every transform severity gets it back without giving the modern gains away: the
2022-era mean is 0.865, and held at a matched 2.5% WildRF false-positive rate,
`ood` clean recall goes 0.883 → 0.899 against the previous head while DALL·E 3
goes 0.944 → 0.958. The matched rate matters — recall read at two different
thresholds is not a comparison.

**Residual weaknesses, unfixed.** WildRF gains are carried by Reddit (0.8% FPR);
Twitter (4.4%) and Facebook (5.6%) remain materially worse, and Facebook's n=160
is directional only. The worst view on `ood` is now `chain_heavy`, a composition
held out from training — the honest place for a failure to be, but also the
condition a re-uploaded, re-compressed image actually meets.

## Trade-offs & limitations, and what more time would buy

1. **Only two modern generators in training, one held out.** The generalisation
   claim rests on a single held-out modern generator. Three or four would make it
   much stronger, and it is the first thing more time would buy.
2. **Heavy noise and deep composition are the real failure modes.**
   `noise_sigma0.1` costs 3–11 points of AUC per tier, and `chain_heavy` costs 11
   on `ood`. Noise is trained on at every severity now, so that number is no
   longer extrapolation; the chains still are, and they are the gap to close.
3. **One global threshold.** Per-domain thresholds would beat one number — the
   platform table above shows Facebook and Twitter want different operating
   points than Reddit — and temperature calibration is unimplemented.
4. **Legacy-generator recall regressed** at the new operating point. Deliberate,
   but real, and stated above rather than buried.
5. **`ood` is a synthetic proxy for "unknown future generators."** `dalle3_holdout`
   is the genuine test and it is n=1,500 from one publisher.
6. **Real-domain coverage remains a binding constraint.** Residual false
   positives concentrate in photography registers the training reals do not cover.
7. **Frozen backbone is the whole robustness strategy.** No adversarial training,
   no frequency-domain forensics. That keeps inference to a single linear layer
   and is why it generalises — but any weakness in the backbone's own pretraining
   is directly a weakness here.
8. **Compute-bound, not idea-bound.** One RTX 3080. Every ablation is a data
   ablation because that is what fits — which, given that data work produced
   every gain this project made, turned out to be the right constraint to have.

## Project layout

```
main.py                        Shim -> aigc_detect.cli (`uv run main.py --help`)
predict.py                     Standalone inference entry point (deliverable 5.5.2)
.env.example                   Documented environment variables; copy to .env
experiments/<name>.yaml        A declared training run: manifest, backbone, views,
                                 feature pipeline, head, schedule. `aigc experiment
                                 run <name>` is what reproduces a checkpoint.
src/aigc_detect/
  cli/                         One module per command group. Each command's handler
                                 and its argparse registration live together, so
                                 adding a command is one file plus one line.
  config/
    settings.py                  .env + os.environ - the only reader of either
    paths.py                     Directories and manifests, rooted at $AIGC_DATA_ROOT
    labels.py                    Label ids, split fraction, fallback norm stats
    generators.py                Generator -> architecture family tables
  registry/
    backbones.py                 Frozen VFM registry + loader (asserts <2B params)
    heads.py                     LinearHead / MLPHead
    corpora.yaml                 Every corpus, declared with its role
    sources.yaml                 How each corpus is (re)pulled - fetcher, repo, split,
                                   cap, re-encode and quality gate, in YAML not Python
  data/
    transforms.py                Robustness transform pipeline (5.2 table + chains)
    dataset.py                   ManifestImageDataset + resolve_image_path
    corpus.py                    The corpus registry (declared, not globbed)
    manifest.py                  Recipes: include / filter / assign / split
    sources.py                   Reads sources.yaml into Source records
    fetchers/                    One backend per kind (hf, kaggle, manual) behind one
                                   `Fetcher` protocol. Resume is the default: every
                                   batch commits, index before state file, always.
    audit/                       The blind-probe shortcut audit and the gate it feeds.
                                   A corpus that clears 0.70 balanced accuracy cannot
                                   enter a training manifest without a written override.
    relocate.py                  The one-time move into data/corpora/
    prune.py                     Orphan sweep (reports, never deletes)
  cache/                       Content-addressed embedding store
    hashing.py                   Image id = blake2b of the file's bytes (memoised)
    store.py                     Sharded float16 vectors + WAL SQLite index
    identity.py                  Backbone id without loading the checkpoint
    migrate.py verify.py export.py
  embed/
    embeddings.py                Cache pooled embeddings per manifest
    views.py                     Embed every robustness view; the .npz files are
                                   a projection of the store
  train/
    features.py                  FeaturePipeline: gather / l2norm / standardize. The
                                   ONE place that turns embeddings into head input.
    probe.py                     Paper training recipe on cached embeddings
    calibrate.py                 The decision-threshold protocol, run in-loop
    experiment.py                Config in, run directory out (data/runs/<run_id>/)
  evaluation/
    grid.py                      Robustness evaluation summary (5.5.4)
    error_analysis.py            False positive/negative + per-generator report (5.5.5)
  inference/
    predict.py                   Inference logic shared by predict.py and `aigc predict`
    bundle.py                    The model bundle: backbone (with revision), fitted
                                   feature pipeline, head, threshold AND how it was
                                   derived. Legacy checkpoints upgrade in memory.
    detector.py                  The `Detector` protocol + FrozenProbeDetector, so a
                                   caller holds "a model" and never a backbone registry
apps/server/app.py             The demo's FastAPI server (`aigc-serve`). Not graded;
                                 needs `--extra demo`.
demo/extension/                One source tree under src/, built by build.js into
                                 dist/chrome/ and dist/firefox/ from one manifest base
tests/                         Cache invariants, CLI/config contract, manifests,
                                 fetchers, features, calibration, bundles, experiment
                                 configs, and predict-vs-server parity (`uv run pytest`)
.github/workflows/ci.yml       ruff + the full suite, on every push
scripts/
  audit_data.py                  Shortcut audit over the corpus registry (CLI entry)
  audit_corpora.py               Per-corpus health + audit verdict report
  plot_run.py                    Charts from one `aigc experiment run` run directory
  run_race.py                    Backbone race runner
data/                          Relocatable via $AIGC_DATA_ROOT. Images are
                                 gitignored; everything describing them is not.
  corpora/<id>/                  One corpus: images/ + index.csv + corpus.yaml
  manifests/<name>.yaml          The recipe; resolved/<name>.csv its rows
  cache/                         The content-addressed store ($AIGC_CACHE_ROOT)
  embeddings/                    .npz projections of the store (rebuildable)
  runs/<run_id>/                 One experiment run: resolved config, eval grid,
                                   threshold sweep, and the bundle it produced
  quarantine/                    A rejected corpus + the evidence (never train on it)
models/                        Head checkpoints (a few KB each; backbone weights are
                                 downloaded, never saved here). archive/ holds
                                 superseded ablation arms whose numbers the docs cite.
stats/                         Presentation data (see stats/README.md)
docs/                          findings, experiments, data, transforms, assets/, archive/
reports/                       Robustness grids, race results, error analysis, audit log
```

## Development tools & stack

- Python 3.11.15, [uv](https://docs.astral.sh/uv/) for environment/dependency
  management. The project is an installed package (hatchling, `src/` layout),
  so `import aigc_detect` works without any path manipulation. Optional
  extras: `--extra demo` (FastAPI demo server), `--extra viz` (matplotlib).
- `ruff` for linting and `pytest` for the contract tests, in the `dev`
  dependency group, run in GitHub Actions on every push
  (`.github/workflows/ci.yml`). The suite pins the CLI surface, the config
  package's public names, every manifest recipe's resolved rows, the cache
  store's invariants, and that `predict.py` and the demo server agree to
  1e-4 — which is what makes restructuring the codebase safe rather than
  hopeful. Tests that need image bytes or a trained checkpoint skip by name
  on a bare clone instead of failing.
- `python-dotenv` for `.env` loading — see [Setup](#setup).
- PyTorch 2.13.0+cu130 + torchvision, scikit-learn (AUC/balanced
  accuracy), pandas/numpy, Hugging Face `datasets`/`transformers`/`timm`
  for backbone loading and dataset streaming, kagglehub for one dataset
  mirror.
- Windows 11 + an RTX 3080 for development; VS Code + Claude Code as the
  editor/assistant.
- No paid APIs — every backbone is a public checkpoint downloaded once and
  run locally.

## Team

Solo submission — all work (data pipeline, model, evaluation, demo, writeups) by
the repository owner. Claude Code was used as a coding assistant throughout;
every experimental result reported here was produced by the committed scripts and
is reproducible from them.
