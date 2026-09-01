# HANDOFF_2 — session of 2026-08-30 (evening)

Written for a fresh agent with none of this session's context.
`HANDOFF.md` is the **project** handoff (section A = the shipping recipe) and is
part of the submission. This file is the **session** log and supplements it.

**Where the project is:** the model is finished and frozen. A clean submission
repo exists at `C:\Users\angus\Desktop\Submission`. The user is now writing the
**demo video script** (deliverable §5.5.3).

---

## 1. OPEN THREADS — deferred and never resolved

### 1.1 Product / presentation name — BLOCKING THE VIDEO SCRIPT

The user asked for hackathon branding candidates ("we need a title for the
presentation instead of just track 5"). Candidates were proposed; **no choice was
ever made.** This is now the immediate blocker: the video script needs a name.

Proposed, with collision risk:

| name | rationale | risk |
|---|---|---|
| **Plumbline** (recommended) | a plumb line is one straight line that tells you what's true — maps onto both the 1,025-param linear probe and the refusal to falsely accuse | low |
| Halide | silver halide, real film chemistry | **high** — Halide is an image-processing language *and* a popular iOS camera app |
| Assay | a test for what something is actually made of | low |
| Grain | film grain, the fingerprint of a real sensor | medium (generic) |
| Emulsion | the photographic layer | low |
| Litmus | "one simple reliable test" | **high** — established email-testing company |

Suggested title pairings: *"Plumbline — detecting AI images without accusing
photographers"* (recommended opener), or *"Plumbline: 1,025 parameters, and
everything that mattered was data"*.

**Ask the user to pick before drafting the script.**

### 1.2 `demo/extension/*` modified by the user, never reviewed

`content.js`, `overlay.css`, `popup.html`, `popup.js` show as modified in the
working tree. **These edits are the user's, not the assistant's** — they appeared
mid-session and were flagged twice but never explained or reviewed. They were
copied into `Submission/` as-is. Confirm they are intentional and working before
the demo is filmed, since the extension is the thing on camera.

### 1.3 Nothing is committed, anywhere

- `tiktoktechjam2026`: 17 modified files + many untracked, all uncommitted.
- `Submission`: `git init` done, **zero commits** (user said "dont commit
  anything yet").

### 1.4 Model-size decision, settled but re-openable

The user disliked the ~1.27 GB backbone download and asked about bundling
backbone+head into one file. **Not possible in a public repo** — 316M params is
1.26 GB fp32 / 632 MB fp16, and GitHub hard-rejects files >100 MB. Git LFS free
tier (1 GB/month bandwidth) would break clones for judges.

Decision taken: **keep the HuggingFace download, pin the revision.** Done.

Still available if the user reopens it: `onnx-community/PE-Core-L14-336-ONNX`
has `onnx/vision_model_fp16.onnx` at **635 MB** (half). Taking it needs
`onnxruntime`, a second backbone loader, and **full re-validation** — fp16 shifts
scores, which invalidates the 0.985 threshold and every reported table. Treat as
"yes, but hours of re-running", not a quick swap.

### 1.5 Known weaknesses accepted, not investigated

Flagged, recorded in FINDINGS 2k, never chased down:
- **Twitter/Facebook FPR regressed** vs the previous head (Reddit carries the
  aggregate WildRF gain). Facebook n=160 is directional; Twitter n=341 is not.
- **Worst false-positive view is now `jpeg_q70`** — a *trained* view, and the one
  closest to real CDN recompression, i.e. the deployment condition.

### 1.6 Remaining deliverables

§5.5.1 Devpost write-up and §5.5.3 demo video are **not started**. Deliverables 4
(robustness summary) and 5 (error analysis) exist inside README/DEMO/`stats/`
but have not been extracted into standalone submission artifacts if the form
wants them separately.

---

## 2. What happened this session

### 2.1 Finished the modern-generator experiment (the headline result)

Context from the prior session: three modern AIGC corpora had been downloaded
(`nano_banana`, `midjourney_v6`, `sd3`) with DALL·E 3 to be held out.

1. **Embedded `sd3`**, trained with all three added → **regression** on the OOD
   tier, worst exactly where the data should have helped (DALLE2 degraded AUC
   0.7927 → 0.7205).
2. Re-ran at 1 epoch to rule out an epoch artifact. Still regressed.
3. **Diagnosed the cause: `gmongaras/Stable_Diffusion_3_Recaption` is not SD3
   output.** It is a *recaptioning* corpus — real photographs with SD3-authored
   captions. 1,500 real photos had been labelled AIGC.
   - The control head scored them at mean P=0.0230, indistinguishable from real
     photographs (0.0178) — confidently real, not uncertain.
   - 281 distinct resolutions in 400 images, 7.5% square, modal sizes 500x500 /
     640x480 / 800x600, 78% of the source rejected under 384px. Genuine
     generator dumps are one resolution, ~100% square.
   - `sd3_000011.jpg` is a scraped product photo with an "I wanne Buy" watermark.
   - SD3 alone contributed ~half the training loss (0.2599 → 0.1275 on removal).
4. **Retrained without SD3** → regression fully recovered.
5. **Vetted, embedded and scored the DALL·E 3 holdout** (18 views). Genuine
   output: 13 distinct resolutions, all on DALL·E 3's native 1:1 and 7:4 modes.
6. **Result: all 18 views improved** over the previous head, at a *higher*
   threshold — strict dominance. Training on Midjourney-v6 + nano-banana
   improved a generator from neither source. First time in the project that
   generator-side data moved the numbers.

### 2.2 Final head selected and shipped

`models/pe-core-l__linear__aigcmodern_nosd3_e1.pt` at threshold **0.985**.

Trained 1 epoch, not 2 — epoch 2 raises clean AUC and lowers robustness
(val AUC_robust 0.9740 → 0.9670), and `train_head.py:318-333` has **no
best-epoch selection**, so it silently saves the worse epoch.

Threshold re-derived with `predict.py`'s exact documented protocol (WildRF
pooled over `clean + jpeg_q70/q90/resize_0.5x/chain_light`, split by image,
0.005 sweep, F1 on half A reported on half B). The protocol reproduces the
previous head's recorded 0.940 / .0283 / .9686 as 0.940 / .0280 / .9663,
which validates the reconstruction.

| head | threshold | held-out FPR | held-out TPR |
|---|---|---|---|
| trainext (previous) | 0.940 | .0280 | .9663 |
| **nosd3_e1 (SHIPPED)** | **0.985** | **.0246** | **.9773** |

Lower FPR *and* higher recall simultaneously — not a trade.

Final numbers (clean AUC / 17-view transformed mean): demo_val 0.9999/0.9962,
ood 0.9979/0.9689, wildrf 0.9969/0.9855, dalle3 0.9984/0.9906.

### 2.3 Quarantined SD3 so it cannot come back

- Images, index, manifest and 11 embeddings moved to `data/quarantine/`.
- `data/quarantine/README.md` holds the full evidence table.
- Removed from `SOURCES` in `scripts/download_aigc_modern.py`, from
  `config.py`, and from the `main.py` manifest registry. Verified: both
  `--manifest sd3` and `--source sd3` now hard-fail at the CLI.
- Tombstone comments explain the `-recap`/`_Recaption` naming trap — it is **not**
  a provenance signal in either direction (`Photoroom/midjourney-v6-recap`
  carries it and IS genuine output). Check pixel resolutions instead.

### 2.4 Fixed the audit hole that let it through

`scripts/audit_data.py:59` globbed `data/raw/` only, so every corpus added after
it — all of `aigc_ext/` and `real_ext/` — was **never probed**. The audit
reported nothing wrong by not looking. Now spans `AUDIT_DIRS = (RAW_DIR,
AIGC_EXT_DIR, REAL_EXT_DIR)`. Post-fix pooled blind probe **0.5829 PASS**; report
saved to `reports/audit_data_2026-08-30.txt`.

### 2.5 Stats + charts pipeline (new)

Three scripts, all regenerable in ~4 minutes:
- `scripts/train_instrumented.py` — replica of the shipping run that logs
  per-step loss and val AUC every 50 steps. Imports the real trainer's own
  `load_view_cache` and `_grid_auc`, so it is a replica not a reimplementation.
  Reproduces the shipping run exactly (epoch 1 loss 0.1275, AUC_clean 0.9987,
  AUC_robust 0.9740).
- `scripts/export_eval_stats.py` — all evaluation numbers as tidy CSVs.
- `scripts/plot_stats.py` — renders `stats/charts/01..07.png`.

`matplotlib` added as an optional `viz` extra (matching the existing `demo`
extra convention). Output documented in `stats/README.md`.

**One methodological correction worth preserving:** the ablation chart first
compared arms at each head's own threshold, which showed the SD3-*poisoned* arm
with the highest DALL·E 3 recall. Arms are now compared at a **matched 2.5%
WildRF FPR**. Correct picture: modern data buys DALL·E 3 recall (0.874 → 0.944
over 18 views) and costs legacy OOD recall (0.919 → 0.883).

### 2.6 Documentation

- **FINDINGS.md**: added §2k — full arc (regression → SD3 diagnosis → recovery →
  DALL·E 3 verdict), the real-FPR breakdown, and the ship-1-epoch finding.
- **HANDOFF.md**: section A amendment supersedes the old "Ship this".
- **DEMO.md**: rewritten against the new head; SD3 added as data fault **(c)**;
  §6 now leads with the compact clean-vs-transformed summary (deliverable
  §5.5.4) with the 18-row table demoted beneath it.
- **README.md** (source repo): substantially rewritten. It had been badly stale —
  named `augchain.pt`, quoted OOD 0.9532, said a retrain was "in progress", and
  still listed the WhichFaceIsReal false-positive cluster as an open finding when
  that was the project's own label bug, long since fixed.

### 2.7 Bug found and fixed: the graded script used a stale head

**Root `predict.py` still defaulted to `models/pe-core-l__linear__augchain.pt`**,
several heads out of date. `main.py`, `demo/server.py` and
`src/aigc_detect/predict.py` had all been updated; this one was missed. The
required §5.5.2 deliverable would have scored with the wrong model.

Fixed, with a sync comment. Verified end-to-end.

### 2.8 Documentation error corrected

`README.md` claimed training used a *stochastic* augmentation mix. It does not.
`RobustnessAugment` / `build_train_transform` exist but are **not in the training
path** — the only caller is `main.py preview-augment`. Everything is fixed and
deterministic because training runs on cached embeddings. Corrected in the source
README; the Submission README (a separate concise rewrite) never had the claim.

### 2.9 Backbone revision pinned

`src/aigc_detect/backbones.py` now pins `pe-core-l` to
`e63206c8e3a0e9b699e40f31080eebd78fd2258e` via timm's `hf-hub:repo@sha` form.
The threshold is calibrated against those exact weights; a silent upstream
re-upload would shift every score with no error raised. Verified predictions
unchanged (max delta 4.77e-07) and `HF_HUB_OFFLINE=1` gives identical output.

---

## 3. The Submission repo

`C:\Users\angus\Desktop\Submission` — 98 files, ~2.5 MB excluding `.venv`,
`git init` done, **no commits**.

Contains: `src/aigc_detect/`, all 20 `scripts/`, `main.py`, `predict.py`, uv
files, `demo/` (server + extension), **one** checkpoint, `stats/` (7 CSVs +
7 charts), minimal `reports/`, stubbed `data/` tree, and the docs.

Judgment calls made: docs included (README is a required deliverable and links to
DEMO/FINDINGS/NARRATIVE/HANDOFF and `reports/race/`); `AGENTS.md`/`CLAUDE.md`
excluded as agent instructions; `stats/instrumented_head.pt` excluded (only the
final checkpoint ships); `data/` stubbed with `.gitkeep` rather than copying
7.8 MB of manifests full of machine-specific absolute paths;
`data/quarantine/README.md` kept as evidence.

**`.gitignore` gotcha fixed:** `data/**` plus `!data/**/.gitkeep` silently
dropped every stub — git will not descend into an excluded directory, so a
negation for a file inside one never fires. Needs `!data/**/` first.

### Verified acceptance test

Clean `uv sync`, then `predict.py` on a mixed directory (5 DALL·E 3, 2
nano-banana, 7 Unsplash reals, 1 nested subdir, 1 deliberately corrupt file):

```
14/14 correct — 7/7 AI flagged, 7/7 real cleared
valid JSON array; keys exactly {image_path, pred}; floats in [0,1];
relative POSIX paths; deterministic ordering; corrupt file skipped with a
warning rather than crashing
```

**The Submission README diverges from the source README on purpose** — it was
rewritten to ~237 lines targeting exactly the five things judges ask for
(overview / setup / reproduce / limitations / contributions). The user then
edited it further. Do not "sync" the two.

---

## 4. Code changes, for a fresh agent

**Bugs fixed**
- `predict.py` — root deliverable script defaulted to a stale head
  (`augchain.pt` → `aigcmodern_nosd3_e1.pt`). Highest-impact fix of the session.
- `scripts/audit_data.py` — blind probe only covered `data/raw/`, so every corpus
  added later was never audited. Now spans raw + aigc_ext + real_ext.
- `README.md` — removed a false claim that training uses stochastic augmentation,
  and a stale "unresolved WhichFaceIsReal false positives" finding that was
  actually a fixed label bug.
- `Submission/.gitignore` — directory-negation bug that dropped all `data/` stubs.

**Features added**
- `scripts/train_instrumented.py`, `scripts/export_eval_stats.py`,
  `scripts/plot_stats.py` → `stats/` (7 CSVs, 7 charts, `run_meta.json`).
- `viz` optional dependency extra (matplotlib) in `pyproject.toml`.
- Backbone hub-revision pinning in `src/aigc_detect/backbones.py`
  (`_load_timm_backbone(checkpoint, revision=None)`, registry `"revision"` key).
- `data/quarantine/` + README as a permanent record of a rejected corpus.

**Changed**
- Shipping head + threshold updated in `src/aigc_detect/predict.py`
  (`DECISION_THRESHOLD = 0.985`), `demo/server.py` (`DEFAULT_HEAD`),
  `main.py`, and root `predict.py`.
- SD3 removed from `scripts/download_aigc_modern.py` `SOURCES`,
  `src/aigc_detect/config.py`, and the `main.py` manifest registry.
- README / DEMO / FINDINGS / HANDOFF substantially updated (see §2.6).

**Untouched by the assistant:** `demo/extension/*` (user's own edits — see §1.2).

---

## 5. Reproduce the shipping head

```bash
uv run main.py train-head-views --backbone pe-core-l --with-chains \
  --val-sample-rows 2000 --train-manifest train-ext \
  --extra-train-manifest sid-real unsplash-real nano-banana midjourney-v6 \
  --balance --epochs 1 --out models/pe-core-l__linear__aigcmodern_nosd3_e1.pt
```

`--epochs 1` is load-bearing. `--balance` is load-bearing (the extra corpora are
class-skewed; without `pos_weight` any FPR change is a prior shift, FINDINGS 2h).
**Never** add `sd3` — it is quarantined and removed from the CLI.

---

## 6. Material for the demo video script

The strongest beats available, in rough order of impact:

1. **The false-positive framing.** Every other team will report accuracy on AI
   images; this is the only one whose headline is about not harming real users.
   2.46% FPR at 97.73% recall on real Reddit/X/Facebook photos.
2. **The held-out DALL·E 3 result.** Pulled three modern generators, trained on
   two, kept one back. All 18 views improved. That is a falsifiable
   generalisation claim, not a benchmark score.
3. **The era inversion.** Weakest generators are the *oldest* (ADM 0.439,
   DALLE2 0.466, both 2021-22); the newest, never trained on, is at 0.990.
   Directly answers "will this work on tomorrow's generators".
4. **Three mislabelled corpora caught by looking at pixels, not metrics** —
   depth maps as photographs, GAN faces as real, real photos as SD3. The
   "I wanne Buy" watermark screenshot is a strong visual.
5. **1,025 trainable parameters** on one RTX 3080.
6. **`stats/charts/02_validation_auc.png`** — robustness peaks mid-epoch-1 then
   declines while clean AUC stays pinned. One picture that justifies `--epochs 1`
   and shows the team validates on the right quantity.

`DEMO.md` is already written as a narration script with "Show:" cues per section.
