# AGENTS.md

## Project

AIGC (AI-generated image) detector for **TikTok TechJam 2026, Track 5**:
"Robust Detection of AI-Generated Images Under Real-World Transformations."
Binary classifier (real vs AI-generated), must stay accurate after JPEG
compression, Gaussian blur, resize/thumbnail, Gaussian noise, color jitter,
and center crop. Model constraint: **<2B parameters**. Solo submission.

Full brief digest, setup steps, and command reference: see `README.md`.
This file is the concise "what's built and why" map for picking the
project back up.

## Stack

- **uv-managed** Python 3.11 project — always run code via `uv run ...`,
  never bare `python`.
- `torch==2.13.0+cu130`, pinned via `[tool.uv.sources]` / `[[tool.uv.index]]`
  in `pyproject.toml` (RTX 3080, driver supports CUDA 13.3). Verified
  working with a real CUDA op — don't assume, but this has been checked.
- Single entry point: `uv run main.py <command>` (run `--help` for the
  full subcommand list). `scripts/*.py` are importable modules main.py
  wraps, not meant to be the primary interface.

## Layout

```
main.py                       Entry-point CLI
src/aigc_detect/
  config.py                    Paths, label ids (0=real, 1=aigc), IMAGE_SIZE=224,
                                 VAL_FRACTION=0.15, RANDOM_SEED=42
  transforms.py                 Augmentation / robustness-eval pipeline
                                 (torchvision.transforms.v2). Implements the
                                 brief's exact transform table (5.2) — don't
                                 change parameter values without checking it.
  dataset.py                    ManifestImageDataset(Dataset): CSV manifest
                                 (image_path,label,source) -> (tensor, label)
scripts/
  download_data.py               CIFAKE (Kaggle) + SID_Set (HF, streamed/capped)
                                  TRAINING data downloaders + per-source indexers
  make_splits.py                  Merges data/raw/*_index.csv, stratified 85/15
                                   split -> data/processed/{train,val}.csv
  download_demo_val.py            Self-reported demo-val downloaders: COCO
                                   val2017 (auto) + WildFake "DALL·E Advanced"
                                   (manual-import, indexed once you place files)
  make_demo_val.py                 Merges demo-val indexes -> demo_val.csv,
                                    runs a leakage guard against train/val
data/
  raw/                            Per-source *_index.csv + downloaded training
                                   images (gitignored)
  processed/                      train.csv / val.csv — what training actually
                                   reads (gitignored)
  demo_val/                       demo_val.csv — self-reported ONLY, see below
                                   (gitignored)
```

## Current state (as of 2026-08-29)

**Built & verified:** environment/CUDA, dataset download + indexing, the
augmentation pipeline (all 15 transform×severity combinations produce
correct `224×224` tensors, checked against both synthetic and real images),
the stratified train/val split, and the COCO half of demo-val.

**Data on disk:** CIFAKE full (120,000 images), SID_Set 4000/class subset
(8,000 images) → split into `train.csv` (108,800) / `val.csv` (19,200).
COCO val2017 (5,000 images) → `demo_val.csv`.

**Not built yet:**
- Model architecture + training loop (must respect the <2B param limit)
- Inference script emitting `{image_path, pred}` JSON (required deliverable)
- Robustness evaluation report (clean vs. each transform×severity, using
  `build_robustness_eval_transforms()`, which already exists for this)
- Error analysis note (false positive/negative examples, trade-offs)
- WildFake ingestion — both as an optional extra training source and as
  the demo-val "DALL·E Advanced" AIGC half — pending manual download (see
  README, "Demo validation set" section, for why this can't be scripted)

## Key decisions / constraints

- **Never train on `data/demo_val/`.** The brief (5.4) explicitly says not
  to; it doesn't count toward scoring. It lives in a directory
  `make_splits.py` structurally never looks at, so this can't happen by
  accident. Use it only for periodic checkpoint eval, not hyperparameter
  tuning — it's the only external benchmark available, so tuning against
  it would just be overfitting to it under another name.
- Training/iteration always uses the internal 85/15 `train.csv`/`val.csv`.
- The augmentation parameter table (JPEG q90/70/50/30, blur σ0.5/1/2, resize
  0.5x/0.25x, noise σ0.02/0.05/0.10, color jitter ±20%, center crop 80%) is
  fixed by the brief — see `transforms.py`'s module docstring.
- COCO val2017: `download_demo_val.py` prefers a Kaggle mirror
  (`xthink/coco-2017-val-images`) — the official S3 bucket was observed
  throttled to ~12kB/s (18+ hr ETA) on this network; Kaggle was ~20MB/s.
  Falls back to S3 automatically if Kaggle isn't configured.
- ModelScope (for WildFake) is unreachable at the API/SDK level from this
  network — confirmed via both `curl` and the `modelscope` Python SDK
  hanging indefinitely, even though the plain website loads. Manual
  browser download is the only path; matches the brief's own note about
  needing a translate-button step.

## Conventions

- Windows dev machine, PowerShell/Git-Bash — avoid Unicode em-dashes (and
  similar) in anything passed to `print()`; they mojibake in the console.
  Fine in docstrings/comments, just not stdout.
- CSV manifests always use columns `image_path,label,source`; label is
  `0`=real, `1`=AIGC (`aigc_detect.config.LABEL_REAL`/`LABEL_AIGC`).
