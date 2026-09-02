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

**Read [docs/findings.md](docs/findings.md) before touching training data or
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
- **The project is an installed package.** `import aigc_detect` works from
  anywhere; there is no `sys.path` manipulation anywhere in the tree, and
  adding any back is a regression a test will catch.
- **Two equivalent entry points.** `uv run aigc <command>` is the console
  script; `uv run main.py <command>` is a 13-line shim kept because every
  document and every muscle memory uses that form. They dispatch to the
  same `aigc_detect.cli`.
- **Dev tooling:** `uv run ruff check .` and `uv run pytest`. Both are
  currently clean and green — keep them that way; the test suite is what
  makes restructuring safe.
- **Configuration:** copy `.env.example` to `.env`. Nothing outside
  `aigc_detect.config.settings` reads `os.environ`. `AIGC_DATA_ROOT`
  relocates the ~24 GB image tree off the system drive.

## Layout

```
main.py                        13-line shim -> aigc_detect.cli
predict.py                     Deliverable inference entry point (shim)
src/aigc_detect/
  cli/                         One module per command group; each command's
                                 handler and its argparse registration live
                                 together as register_<command>(sub). Adding a
                                 command is one file plus one line in parser.py.
                                 parser.py's docstring IS the root --help text.
  config/
    settings.py                  .env + os.environ, the ONLY reader of either.
                                   Cheap to import on purpose: no torch (device
                                   resolution is a lazy function).
    paths.py                     Every directory and manifest, rooted at
                                   $AIGC_DATA_ROOT so the tree relocates whole.
    labels.py                    0=real, 1=aigc; split seed; fallback norm stats.
    generators.py                generator -> architecture family; TRAIN_GENERATORS.
  registry/
    backbones.py                 Frozen VFM registry. load_backbone(key) ->
                                   (module, pooled_dim, native_res). Asserts
                                   <2e9 params. Ship the VISION TOWER only.
    heads.py                     LinearHead / MLPHead / build_head(kind, in_dim)
  data/
    transforms.py                The brief's exact 5.2 transform table — don't
                                   change parameter values without checking it.
    dataset.py                   ManifestImageDataset: CSV -> (tensor, label)
  cache/                       The content-addressed embedding store (tier 4).
    hashing.py                   blake2b-128 of a file's bytes = its image id,
                                   memoised as (path, mtime, size) -> id.
    store.py                     256 float16 shards per (backbone, view) + a WAL
                                   SQLite index. missing/put_batch/gather/merge,
                                   drop/compact. Bytes fsync BEFORE the index
                                   commits -- that order is load-bearing.
    identity.py                  bb_id for a backbone without loading 1.2 GB of
                                   weights, using the store's own table as memo.
    migrate.py                   Folds the legacy .npz caches in.
    verify.py                    Re-embeds a sample and compares by cosine.
    export.py                    index.csv (+ .npy) for eyeballing.
  embed/
    embeddings.py                One pooled embedding per image, cached .npz
    views.py                     Every ROBUSTNESS VIEW (18: clean + 14
                                   single-transform + 3 chained) from a single
                                   decode, written to the cache/ store. The .npz
                                   files are a manifest-ordered PROJECTION of it,
                                   rebuildable with no GPU. Read its docstring
                                   before touching seeding or cache keys.
  train/probe.py               Paper recipe on cached embeddings
  evaluation/
    grid.py                      Per-view AUC/BAcc at one fixed threshold,
                                   AUC_robust, robustness gap. Deliverable 5.5.4.
    error_analysis.py            Deliverable 5.5.5.
  inference/predict.py         {image_path, pred} JSON. Owns DECISION_THRESHOLD.
scripts/                       Ad-hoc corpus pulls and one-off drivers. Being
                                 replaced tier by tier (see below), NOT deleted
                                 ahead of their replacements.
tests/                         Cache-store invariants, CLI contract, config
                                 surface. Written against the invocation, not
                                 import paths, so they survive code moving.
docs/                          findings, experiments, data, transforms, archive/
data/                          Gitignored. $AIGC_DATA_ROOT-relocatable.
```

## Current state (as of 2026-09-02)

**The submission is tagged `hackathon-final`.** Post-hackathon work happens
on `refactor/v2` and is turning this into an extensible testbed.

**Shipping model:** `models/pe-core-l__linear__allsev_e1.pt` — a linear probe
on frozen `pe-core-l`, pinned to hub revision `e63206c8`, at decision
threshold **0.980**. That threshold is calibrated to *this* checkpoint and
currently lives as a module constant in `inference/predict.py`; re-derive it
on every head swap with `scripts/derive_threshold.py`. Superseded ablation
arms are in `models/archive/` — moved, not deleted, because
`docs/findings.md` cites their numbers.

**Backbone race: decided.** `pe-core-l` beat `metaclip2-h` and `dinov3-l` on
the OOD tier (the only tier with room left to discriminate). Results are
committed under `reports/race/`; the losing weights have been deleted from
the HF cache and re-download on demand.

**Evaluation tiers, none trained on:** `val` (in-distribution), `heldout`
(same generators), `demo_val` (the brief's benchmark), `ood` (18 generators,
ten unseen), `wildrf_test` (real social-media re-encoding), and
`dalle3_holdout` (modern diffusion, deliberately kept back).

**Built and working:** the full frozen-backbone + probe pipeline, the 18-view
robustness grid, error analysis, the deliverable inference script, and the
demo server + Chrome extension.

**Refactor progress:** Tiers 0–4 are done — safety net, docs consolidation,
a 35 GB disk scrub, the package restructure, the config package, and the
content-addressed cache.

**Tier 4, what changed and what it costs you.** An embedding is identified by
(image bytes, backbone, view spec) and nothing else — not the path, not the row
order, not which manifest asked for it. `embed-views` now writes every vector to
the store and treats `data/embeddings/*.npz` as a projection of it, so:

- a killed run resumes from the last committed batch instead of restarting;
- re-encoding one image costs one forward pass, not one whole file;
- moving the repo, or rebuilding an .npz, costs a gather and no GPU;
- two machines merge with `aigc cache merge` — `scripts/worker.py`'s
  "THE REPO PATH MUST MATCH" rule is gone, not merely documented.

The migration imported 695,321 vectors (1.4 GB) of existing GPU work.
`aigc cache verify` re-embeds a sample and confirms it, and an .npz rebuilt from
the store alone is bit-identical to the one it replaced.

**The one thing it invalidated:** stochastic views (`noise_*`, `color_jitter`,
`chain_medium`, `chain_heavy`, `trainchain_*`) were seeded on the image's
*absolute path*, so a rename silently redrew every noise realization. They are
now seeded on the content id, which makes the ~725k path-seeded vectors
unreproducible by construction — they were deliberately not migrated. Those
views recompute on their next `embed-views` run (resumable, and only the gaps);
deterministic views were never affected. `eval-grid` will report a stochastic
view as STALE until it is re-run.

The remaining tiers, in order:

- **5 — data hierarchy and manifest recipes**, then the image-level prune
  that tier 4 had to precede (the migration hashes files the prune deletes).
- **6 — a source registry and resumable fetchers**, replacing the six ad-hoc
  `download_*.py` scripts, with the blind-probe audit as a gate on every pull.
- **7 — experiment configs and a model bundle** carrying its own threshold;
  this is where backbone and head become genuinely swappable.
- **8 — demo as first class:** a `Detector` protocol and a cross-browser
  extension build.
- **9 — CI.**

**Known issue, unfixed:** `data/demo_val/demo_val.csv` references 5,000 COCO
images by absolute path inside `~/.cache/kagglehub`. A committed manifest
depends on a transient cache directory; tier 5 ingests those images.

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
  run cross-source transfer before believing a number. See docs/findings.md.
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

## Working conventions

- **uv-managed** project — always run code as `uv run main.py ...` or
  `uv run python ...`, never a bare `python`/`pip` call.
- `data/` is gitignored and mostly not present after a fresh clone; don't
  assume downloaded datasets exist without checking (`uv run main.py
  check-env` reports what's there).
- Post-hackathon research testbed (the submission itself is tagged
  `hackathon-final`). Keep commits small and scoped to what was asked;
  don't rewrite history unprompted.
