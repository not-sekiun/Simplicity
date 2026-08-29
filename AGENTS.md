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

**Read [FINDINGS.md](FINDINGS.md) before touching training data or
interpreting any metric.** It documents two label shortcuts found in
SID_Set (one fixed, one fatal), the cross-source transfer test that
exposed them, and why a high AUC on this project is evidence of a leak
rather than success.

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
  backbones.py                  Frozen VFM registry (metaclip2-h, dinov3-l,
                                 pe-core-l, dinov2-g). load_backbone(key) ->
                                 (module, pooled_dim, native_res). Asserts <2e9
                                 params. Ship the VISION TOWER only.
  embed.py                      precompute_embeddings() -> cached .npz under
                                 data/embeddings/<backbone>__<manifest>.npz
  heads.py                      LinearHead / MLPHead / build_head(kind, in_dim)
  train_head.py                 Paper recipe (AdamW lr 1e-3, bs 128, 2 epochs)
                                 on cached embeddings; per-source val AUC
scripts/
  download_data.py               CIFAKE (Kaggle) + SID_Set (HF, streamed/capped)
                                  TRAINING data downloaders + per-source indexers
  audit_data.py                   Shortcut audit + blind probe (16x16 greyscale
                                   logistic regression). Clearing ~70% means a
                                   shortcut survives. Permanent regression test.
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
augmentation pipeline, the stratified train/val split, the COCO half of
demo-val, the shortcut audit (`audit_data.py`), the aspect-preserving
resize fix, and the full frozen-backbone + probe-head pipeline
(`backbones.py` / `embed.py` / `heads.py` / `train_head.py`).

**Data on disk:** CIFAKE full (120,000 images), SID_Set 4000/class subset
(8,000 images) → split into `train.csv` (108,800) / `val.csv` (19,200).
COCO val2017 (5,000 images) → `demo_val.csv` (reals only, so it cannot
currently produce a score at all).

**Model bring-up:** only `pe-core-l` has been run end to end (val
embeddings cached, linear head trained on a 5,000-row slice). The other
three backbones are wired but deliberately untouched — see FINDINGS.md
trap 1: racing backbones on shortcut-contaminated data ranks them by
shortcut exploitation, not by detection ability.

**Blocking issue:** SID_Set is unusable as training data (composition
shortcut — 0.93 balanced accuracy from an 8x8 greyscale thumbnail, and a
SID_Set-trained head transfers to CIFAKE at chance). Replacement source
identified: `TheKernel01/Tiny-GenImage`. Full detail in FINDINGS.md.

**Not built yet:**
- Tiny-GenImage ingestion + generator-aware splits
- Inference script emitting `{image_path, pred}` JSON (required deliverable)
- Robustness evaluation report + `0.5*AUC_clean + 0.5*AUC_robust` scoring
- Calibration (temperature + threshold per degradation bucket)
- Error analysis note (false positive/negative examples, trade-offs)
- WildFake "DALL·E Advanced" demo-val half — manual download in progress

## Key decisions / constraints

- **Never train on `data/demo_val/`.** The brief (5.4) explicitly says not
  to; it doesn't count toward scoring. It lives in a directory
  `make_splits.py` structurally never looks at, so this can't happen by
  accident. Use it only for periodic checkpoint eval, not hyperparameter
  tuning — it's the only external benchmark available, so tuning against
  it would just be overfitting to it under another name.
- Training/iteration always uses the internal 85/15 `train.csv`/`val.csv`.
- **Architecture: frozen VFM + probe head**, per *Simplicity Prevails*
  (arXiv:2602.01738) — a single linear layer on the pooled output of a
  frozen backbone; AdamW, lr 1e-3, batch 128, 2 epochs. The head is
  configurable `linear | mlp` but **defaults to linear** to stay
  paper-faithful. Backbones are never fine-tuned: freezing is the
  mechanism, not a compute shortcut.
- **NC-licensed backbones are acceptable** (user decision). Competition
  rules require backbones be *public*; MIT/Apache is required only for
  *custom* architectures we release. This admits MetaCLIP-2 (cc-by-nc-4.0)
  and DINOv3 (via the ungated timm mirror).
- **Ship the vision tower only.** The full MetaCLIP2 checkpoint is 1.86B
  params; its vision tower is 630.8M. `backbones.py` asserts <2e9.
- **Per-backbone native resolution**, not a global `IMAGE_SIZE` — the four
  backbones want 224 / 256 / 336 / 518, and normalisation stats come from
  each backbone's own config, not ImageNet.
- **A high AUC on this project is evidence of a leak, not success.** Always
  run cross-source transfer before believing a number. See FINDINGS.md.
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
