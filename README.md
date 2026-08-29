# AIGC Image Detection — TikTok TechJam 2026, Track 5

Robust detection of AI-generated images under real-world transformations
(compression, blur, resize, noise, color jitter, cropping).

## Status

Environment, dataset download/indexing scripts, and the data augmentation /
robustness-eval transform pipeline are built and verified. Local data is
downloaded: **CIFAKE** (full, 120k images) + **SID_Set** (4000/class subset,
8k images) are indexed and split into `data/processed/{train,val}.csv`
(108,800 train / 19,200 val, stratified by source+label). WildFake is not
yet ingested (see below). Model training/inference code is not built yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU + driver (CUDA
13.x compatible; tested on an RTX 3080 / driver reporting CUDA 13.3).

```bash
uv sync            # creates .venv, installs torch+cu130 and all deps
uv run main.py check-env   # verify GPU is visible to PyTorch
```

`pyproject.toml` pins PyTorch to the `cu130` wheel index
(`https://download.pytorch.org/whl/cu130`) via `[tool.uv.sources]` /
`[[tool.uv.index]]`. If your GPU/driver only supports an older CUDA, edit
those two blocks to `cu126` and re-run `uv sync`.

## Project layout

```
main.py                    Entry script (CLI) — see below
src/aigc_detect/
  config.py                 Paths, label ids, image size, split fraction
  transforms.py              Augmentation + robustness-eval transform pipeline
  dataset.py                 ManifestImageDataset (reads a CSV of image_path,label,source)
scripts/
  download_data.py           Dataset download/indexing (CIFAKE, SID_Set)
  make_splits.py              Stratified train/val split builder
data/
  raw/                        Downloaded images + one *_index.csv per source (gitignored)
  processed/                  train.csv / val.csv manifests (gitignored)
```

## Entry script

```bash
uv run main.py check-env                              # torch/CUDA + dataset status
uv run main.py download cifake                        # Kaggle, full (~100MB)
uv run main.py download sid-set --limit-per-class 4000 # HF, streamed + capped subset
uv run main.py split --val-fraction 0.15 --seed 42     # stratified train/val manifests
uv run main.py preview-augment --n 8                   # sanity-check the aug pipeline visually
```

### Dataset credentials

- **CIFAKE** (Kaggle, [birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)):
  needs a free Kaggle API token. Get one at kaggle.com/settings → "Create New
  Token", save it as `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` /
  `KAGGLE_KEY` env vars).
- **SID_Set** (HuggingFace, [saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)):
  public, no login needed normally. If you hit a 401, run
  `uv run huggingface-cli login` first. The full dataset is ~140GB across
  train(210k)/validation(30k)/test(60k, gated); `download sid-set` **streams**
  it and stops after `--limit-per-class` images per label rather than pulling
  everything — matches the hackathon's "limited compute" scope. SID_Set's
  `label=2` ("tampered") is a region-manipulation category, out of scope for
  this binary real-vs-AIGC task, and is skipped by default
  (`--include-tampered` to fold it into the AIGC class instead).
- **WildFake** (ModelScope) as an additional *training* source is in the
  brief's dataset list but not wired into `download_data.py` yet — add it by
  hand under `data/raw/wildfake/` with a `wildfake_index.csv` (columns
  `image_path,label,source`) matching the schema the other two sources
  produce, then re-run `main.py split`. (This is separate from WildFake's
  "DALL·E Advanced" subset used in the self-reported demo-val set below —
  same manual-fetch constraint, different destination directory/purpose.)

Each `download` call writes `data/raw/<source>_index.csv`
(`image_path,label,source`; label 0=real, 1=AIGC). `main.py split` merges
every `*_index.csv` under `data/raw/` and writes a single stratified
`data/processed/{train,val}.csv` (stratified per source+label so each
dataset's class balance is preserved in both splits).

## Demo validation set (self-reported only — never trained on)

The brief (5.4) defines a separate "Validation Dataset (for Demonstration
Purposes Only)": COCO val2017 (non-AIGC, ~4998 imgs) + WildFake's "DALL·E
Advanced" subset (AIGC, ~8843 imgs). It **does not contribute to scoring**
and the brief explicitly says **"do not use the following data during
training."** We treat it as periodic, read-only checkpoint eval — never
part of any train/val split, never a target for hyperparameter tuning
(only the internal `train.csv`/`val.csv` split from CIFAKE+SID_Set drives
actual iteration; see "Which split to use" below).

To keep the "never trained on" guarantee structural rather than just a
convention: this data lives under `data/demo_val/`, a directory
`scripts/make_splits.py` never looks at (it only globs `data/raw/`).

```bash
uv run main.py download-demo coco-val2017              # direct download, no auth
uv run main.py download-demo wildfake-dalle-advanced   # indexes a manual download (see below)
uv run main.py build-demo-val                          # merges both into data/demo_val/demo_val.csv
```

- **COCO val2017** downloads directly from the public COCO S3 bucket
  (~815MB, no auth). Standard val2017 has 5000 images; the brief cites 4998,
  so we index the full standard set as a stated assumption (off by ≤2 of
  ~5000) rather than guess which 2 to exclude.
- **WildFake "DALL·E Advanced"** does **not** download automatically: this
  network cannot reach ModelScope's API or SDK endpoints at all — both
  `curl` against `modelscope.cn/api/v1/...` and the `modelscope` Python
  SDK's `HubApi` hang indefinitely, even though the plain website loads.
  This matches the brief's own note that the ModelScope page needs a
  manual translate-button step. To get this half:
  1. Open https://modelscope.cn/datasets/hy2628982280/WildFake/summary
  2. Use the page's translate button if needed
  3. Find/download the "DALL·E Advanced" generator subset
  4. Extract its images into `data/demo_val/wildfake_dalle_advanced/`
  5. Run `uv run main.py download-demo wildfake-dalle-advanced` to index it
- `build-demo-val` also runs a **leakage guard**: it warns if any demo-val
  image filename collides with one already in `train.csv`/`val.csv`.

### Which split to use, when

Use the internal 85/15 `train.csv`/`val.csv` (CIFAKE + SID_Set) for
everything during development — training itself, early stopping, and
hyperparameter/model selection. Only evaluate against `demo_val.csv`
periodically (e.g. once per saved checkpoint) to produce the "iterative
improvement" numbers the write-up wants; don't let it drive decisions, since
it's your only external benchmark and tuning against it would just mean
overfitting to it by another name.

## Data augmentation / robustness pipeline

`src/aigc_detect/transforms.py` implements exactly the transform table from
the challenge brief (5.2):

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG Compression | quality = 90, 70, 50, 30 | social re-encode, messaging |
| Gaussian Blur | sigma = 0.5, 1.0, 2.0 | out-of-focus |
| Resize | scale 0.5x / 0.25x then upscale | thumbnail generation |
| Gaussian Noise | sigma = 0.02, 0.05, 0.10 | low-light sensor noise |
| Color Jitter | brightness/contrast/sat. ±20% | filter apps, auto-enhance |
| Center Crop | crop 80% | profile-pic cropping, framing |

Three builders:

- `build_train_transform()` — light standard aug (h-flip) + a **stochastic,
  compositional** mix of the table above (1–2 random ops per sample, random
  order, random severity), so the classifier learns robustness rather than
  memorizing one fixed corruption.
- `build_eval_transform()` — deterministic resize+normalize only ("clean"),
  for local validation accuracy.
- `build_robustness_eval_transforms()` — one **deterministic** pipeline per
  (transform, severity) pair in the table, plus `"clean"`, e.g. `jpeg_q50`,
  `blur_sigma1.0`, `resize_0.25x`. Feed each to a separate eval `DataLoader`
  over `val.csv` to produce the clean-vs-transformed comparison table
  required by deliverable 5.5.4 (Robustness Evaluation Summary).

Verified with `uv run main.py preview-augment` against a synthetic manifest;
all 15 pipelines (`clean` + 4 JPEG + 3 blur + 2 resize + 3 noise + color
jitter + center crop) produce correctly-shaped `(3, 224, 224)` float tensors.

## Not yet built

- Model architecture + training loop (constraint: <2B parameters).
- Inference script emitting `{image_path, pred}` JSON per deliverable 5.5.2.
- Robustness evaluation report / error analysis (5.5.4, 5.5.5).
- WildFake ingestion (see note above).

## Team

Solo — 1010angusx@gmail.com.
