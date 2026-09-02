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
- **Dev tooling:** `uv run ruff check .` and `uv run pytest` (220 green).
  Both run in GitHub Actions on every push (`.github/workflows/ci.yml`).
  Keep them green — the suite is what makes restructuring safe. Tests that
  need image bytes or a checkpoint skip by name on a bare clone rather than
  failing; see `tests/conftest.py`.
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
    sources.yaml                 How every corpus is (re)pulled: fetcher, repo,
                                   split, cap, re-encode, quality gate. Adding a
                                   HuggingFace source is eight lines here and no
                                   Python — there is a test that says so.
    backbones.py                 Frozen VFM registry. load_backbone(key) ->
                                   (module, pooled_dim, native_res). Asserts
                                   <2e9 params. Ship the VISION TOWER only.
    heads.py                     LinearHead / MLPHead / build_head(kind, in_dim)
    corpora.yaml                 Every image corpus, DECLARED not globbed, with
                                   its role. `role: eval` is enforced, not
                                   documentary — see data/corpus.py.
  data/
    transforms.py                The brief's exact 5.2 transform table — don't
                                   change parameter values without checking it.
    dataset.py                   ManifestImageDataset: CSV -> (tensor, label)
    corpus.py                    Reads corpora.yaml. A corpus is a PROVENANCE:
                                   one pull, one image dir, one row list.
    manifest.py                  Recipes: a manifest is a declarative selection
                                   over corpora (include/filter/assign/split),
                                   not a hand-built CSV. Replaces the seven
                                   make_*.py scripts.
    prune.py                     The orphan sweep. Reports, never deletes —
                                   "unreferenced" is evidence, not a verdict.
    sources.py                   Reads sources.yaml into Source records.
    fetchers/                    One backend per kind (hf, kaggle, manual) behind
                                   one Fetcher protocol. RESUME IS THE DEFAULT:
                                   bytes land, then index.csv is appended and
                                   fsync'd, then .pull_state.json. Never the other
                                   order — read base.py's docstring before editing.
    audit/                       The blind-probe shortcut audit and the GATE it
                                   feeds. A corpus clearing 0.70 balanced accuracy
                                   cannot enter a training manifest without an
                                   override with a written reason. Enforced in
                                   corpus.assert_trainable, not a second place.
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
  train/
    features.py                  FeaturePipeline (gather / l2norm / standardize).
                                   The ONE place embeddings become head input; both
                                   trainers delegate to it. fit() takes one array,
                                   so there is no overload that leaks val stats.
    probe.py                     Paper recipe on cached embeddings
    calibrate.py                 The threshold protocol, run in-loop. The split is
                                   PINNED (RandomState(0), first half = A); it is
                                   what reproduces findings 2j's recorded table.
    experiment.py                YAML in, data/runs/<run_id>/ out. config_hash is
                                   over the RESOLVED config, so a preset that
                                   expands differently changes the hash.
  evaluation/
    grid.py                      Per-view AUC/BAcc at one fixed threshold,
                                   AUC_robust, robustness gap. Deliverable 5.5.4.
    error_analysis.py            Deliverable 5.5.5.
  inference/
    predict.py                   {image_path, pred} JSON. Loads a Bundle; owns no
                                   threshold constant.
    bundle.py                    The versioned model artifact: backbone WITH its
                                   revision, the fitted feature pipeline, the head,
                                   and the threshold plus how it was derived. A
                                   legacy checkpoint upgrades in memory so the 25
                                   archived heads docs/findings.md cites still load.
    detector.py                  The Detector protocol + FrozenProbeDetector, so a
                                   caller holds "a model" and never a registry.
apps/server/app.py             The demo's FastAPI server (`aigc-serve`). Not a
                                 graded deliverable; needs `--extra demo`.
experiments/<name>.yaml        A declared training run. `aigc experiment run` is
                                 what reproduces a checkpoint.
scripts/                       What is LEFT after tiers 5-7 replaced the rest:
                                 audit_data.py / audit_corpora.py (audit entry
                                 points), plot_run.py (charts from one run dir),
                                 run_race.py (the backbone-race driver). Each is
                                 import-tested — tests/test_scripts.py — because
                                 scripts/ is invisible to every other test and
                                 rots silently.
tests/                         Cache invariants, CLI contract, config surface,
                                 manifest recipes, fetcher resume, features,
                                 calibration, bundles, experiment configs, and
                                 predict-vs-server parity. Written against the
                                 invocation, not import paths, so they survive
                                 code moving.
.github/workflows/ci.yml       ruff + the full suite, on every push.
docs/                          findings, experiments, data, transforms, archive/
data/                          $AIGC_DATA_ROOT-relocatable. Five directories,
                                 each meaning exactly one thing:
  corpora/<id>/                  one PROVENANCE: images/ + index.csv +
                                   corpus.yaml. Images gitignored, the files
                                   describing them committed.
  manifests/<name>.yaml          the RECIPE; resolved/<name>.csv the rows it
                                   currently selects. Both committed.
  cache/                         the content-addressed embedding store
  embeddings/                    .npz projections of it (rebuildable, ignored)
  runs/<run_id>/                 one experiment run: resolved config, eval grid,
                                   threshold sweep, and the bundle it produced
  quarantine/                    a rejected corpus, reduced to evidence
```

## Current state (as of 2026-09-02)

**The submission is tagged `hackathon-final`.** Post-hackathon work happens
on `refactor/v2` and is turning this into an extensible testbed.

**Shipping model:** `models/pe-core-l__linear__allsev_e1.pt` — a linear probe
on frozen `pe-core-l`, pinned to hub revision `e63206c8`, at decision
threshold **0.980**. That threshold is calibrated to *this* checkpoint and
travels INSIDE it: `inference/bundle.py` carries `threshold` and
`threshold_source`, and `train/calibrate.py` re-derives it as the last step of
every `aigc experiment run`. There is no threshold constant to forget to
update. Superseded ablation arms are in `models/archive/` — moved, not
deleted, because `docs/findings.md` cites their numbers, and `load_bundle`
upgrades their old dict shape in memory so they still load.

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

**Refactor progress:** all nine tiers are done — safety net, docs
consolidation, a 35 GB disk scrub, the package restructure, the config
package, the content-addressed cache, the data hierarchy, resumable fetchers
and the audit gate, experiment configs and the model bundle, the demo as a
first-class app, and CI.

**Tier 5, what changed.** `data/` went from eleven top-level directories that
mixed provenance with evaluation tier to five that each mean one thing. Every
corpus is declared in `registry/corpora.yaml` and lives at
`data/corpora/<id>/`; every manifest is a recipe under `data/manifests/`,
resolved into `resolved/<name>.csv`. Seven `make_*.py` builders and four CLI
commands (`split`, `build-ood`, `build-heldout`, `build-demo-val`) are gone —
`aigc manifest resolve <name>` replaces all of them.

- **Manifests are portable.** Every path is now relative to `$AIGC_DATA_ROOT`
  with POSIX separators, so a committed manifest is not a machine-specific
  artifact. `demo_val` no longer points into `~/.cache/kagglehub`: its 5,000
  COCO images were ingested.
- **The acceptance test passed.** After moving 17.6 GB, `embed-views` on the
  OOD tier reported *"every requested view is already in the store — no forward
  passes"*, re-projected all 18 views in 34 seconds, and `eval-grid` returned
  all 18 AUCs bit-identical. Content addressing is what made the move free.
- **`never_train` is enforced,** not documented: eval-role corpora raise if a
  training recipe includes them, and the trainer refuses a flagged manifest.
- **3.3 GB deleted:** the orphaned pexels corpus, the extracted tiny_genimage
  archive, and SID_Set's 4,000 discarded FLUX fakes. All three re-fetchable.
  NOT deleted, and this is the point of the sweep reporting rather than acting:
  the 648 DALLE2/SDXL images `train_ext` holds back on purpose, and WildRF's
  1,555 train/val fakes, which no recipe has claimed yet.

**Tier 4, what changed and what it costs you.** An embedding is identified by
(image bytes, backbone, view spec) and nothing else — not the path, not the row
order, not which manifest asked for it. `embed-views` now writes every vector to
the store and treats `data/embeddings/*.npz` as a projection of it, so:

- a killed run resumes from the last committed batch instead of restarting;
- re-encoding one image costs one forward pass, not one whole file;
- moving the repo, or rebuilding an .npz, costs a gather and no GPU;
- two machines merge with `aigc cache merge` — `scripts/worker.py`'s
  "THE REPO PATH MUST MATCH" rule is gone, not merely documented (the script
  itself was retired in tier 7, once nothing was left for it to do).

The migration imported 695,321 vectors (1.4 GB) of existing GPU work.
`aigc cache verify` re-embeds a sample and confirms it, and an .npz rebuilt from
the store alone is bit-identical to the one it replaced.

**The one thing it invalidated:** stochastic views (`noise_*`, `color_jitter`,
`chain_medium`, `chain_heavy`, `trainchain_*`) were seeded on the image's
*absolute path*, so a rename silently redrew every noise realization. They are
now seeded on the content id, which makes the ~725k path-seeded vectors
unreproducible by construction — they were deliberately not migrated. Those
views were recomputed for all seven scored eval tiers; deterministic views were
never affected. The re-embed confirmed the migration: all 12 deterministic view
AUCs came back IDENTICAL, and the 6 stochastic ones moved by at most 0.0037
(mean -0.0002) — noise-scale scatter around zero, which is a different random
draw of the same test rather than a different test. The training views
(`train`, `train_ext`, ~550k rows) have NOT been recomputed and will report
STALE until someone runs `aigc embed-views --manifest train --train-chains`.

**Tiers 6-9, what changed.**

- **Pulls are declared and resumable.** Eleven sources live in
  `registry/sources.yaml`; `aigc pull run <id>` is the interface, four fetcher
  backends answer one protocol, and resume is the default rather than a flag.
  A changed `Source` config refuses to resume (`--force` is the opt-in) instead
  of interleaving two pull configurations under one corpus id. The six
  `download_*.py` scripts are gone; `download` / `download-demo` /
  `download-ood` are thin bridges so documented invocations still work.
  `download sid-set` deliberately no longer accepts `--split` or
  `--include-tampered`: the registered source is reals-only by construction.
- **The shortcut audit is a gate, not a report.** Every pull ends with the
  blind probe and writes its verdict into the corpus's own `corpus.yaml`;
  `assert_trainable` — already the one answer to "may this corpus train" —
  now fires for a cleared probe too. This is the check that would have caught
  the SD3 and depth-map-pexels incidents.
- **A run is declared, and the threshold ships inside the model.**
  `aigc experiment run <name>` reads `experiments/<name>.yaml` and writes
  `data/runs/<run_id>/` with the resolved config, eval grid, derived threshold
  and a bundle. Reproducing a head used to be a seven-flag command plus a
  separate `derive_threshold.py` run plus copying the number into a constant —
  findings 2j/2k record that step being forgotten and reconstructed twice,
  disagreeing with itself both times.
- **One implementation of the scaler.** `FeaturePipeline` replaced three
  hand-rolled copies of `(x - mean) / std` (in `train.probe`, in
  `inference.predict`, and in the demo server's own `Model` class). The
  server now holds a `Detector`; `tests/test_parity.py` holds it and
  `predict.py` to 1e-4 instead of a docstring asking the reader to trust it.
- **The extension is one source tree, two browsers.** `demo/extension/src/`
  builds to `dist/chrome/` and `dist/firefox/` from one manifest base. It no
  longer ships its own threshold default of 0.5 (18.75% FPR against 2.15%) —
  it reads the calibrated one from the server's `/health`.
- **CI runs everything on a machine that has never seen this data.** A fresh
  clone has the committed manifests and none of the images; `needs_images`
  makes that a clean skip instead of a failure that reads like a broken
  manifest.

## Key decisions / constraints

- **Never train on `demo_val`.** The brief (5.4) explicitly says not to; it
  doesn't count toward scoring. This used to rest on a structural accident
  (`make_splits.py` globbed one directory); it is now a declared rule with
  three enforcement points — the recipe is flagged `never_train`, an
  eval-role corpus raises if a training recipe includes it, and a trainer
  handed a flagged manifest raises. Use it only for periodic checkpoint eval,
  not hyperparameter tuning — it's the only external benchmark available, so
  tuning against it would just be overfitting to it under another name.
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
- COCO val2017: the `coco_val2017` source in `registry/sources.yaml` pulls a
  Kaggle mirror (`xthink/coco-2017-val-images`) — the official S3 bucket was
  observed throttled to ~12kB/s (18+ hr ETA) on this network; Kaggle was
  ~20MB/s. The S3 fallback is documented in that entry, not automated.
- ModelScope (for WildFake) is unreachable at the API/SDK level from this
  network — confirmed via both `curl` and the `modelscope` Python SDK
  hanging indefinitely, even though the plain website loads. Manual
  browser download is the only path; matches the brief's own note about
  needing a translate-button step.

## Conventions

- Windows dev machine, PowerShell/Git-Bash — avoid Unicode em-dashes (and
  similar) in anything passed to `print()`; they mojibake in the console.
  Fine in docstrings/comments, just not stdout.
- Corpus indexes and resolved manifests use columns
  `image_path,label,source,generator` (`aigc_detect.data.corpus.COLUMNS`);
  label is `0`=real, `1`=AIGC (`aigc_detect.config.LABEL_REAL`/`LABEL_AIGC`).
  Every `image_path` is relative to `$AIGC_DATA_ROOT` with POSIX separators —
  a test asserts it, because a manifest that resolves against the wrong root
  empties silently rather than raising.

## Working conventions

- **uv-managed** project — always run code as `uv run main.py ...` or
  `uv run python ...`, never a bare `python`/`pip` call.
- `data/` is gitignored and mostly not present after a fresh clone; don't
  assume downloaded datasets exist without checking (`uv run main.py
  check-env` reports what's there).
- Post-hackathon research testbed (the submission itself is tagged
  `hackathon-final`). Keep commits small and scoped to what was asked;
  don't rewrite history unprompted.
