# Handoff — AIGC Detection (TikTok TechJam 2026, Track 5)

Session of 2026-08-29 (second session). Previous session's handoff is
`HANDOFF.md` — **it is now stale** (it says "no model/training code was written
yet", which is no longer true). This file supersedes it.

**Read [FINDINGS.md](FINDINGS.md) first.** It is the durable technical record —
every measurement, the reproduction commands, rejected data sources with
reasons, and 7 traps. This handoff is the narrative around it.

---

## ⚠️ Deferred / unresolved — needs attention

1. **Kaggle API token was pasted in plain chat (carried over, still
   unconfirmed).** A live `KGAT_`-prefixed token was given in a previous
   session's chat to set up CIFAKE/COCO downloads. It was never written to a
   file, but it is in that conversation's history. The user was advised to
   revoke/regenerate at kaggle.com/settings and **has still never confirmed
   doing so**. Worth raising once more; it is the user's call, not a code task.
2. **The competition's own scoring formula is not implemented.** The user
   specified `Final score = 0.5*AUC_clean + 0.5*AUC_robust`. Nothing in the repo
   computes it. Half the score is currently unmeasured. This is the single
   biggest gap — see "Next steps".
3. **Old `HANDOFF.md` is stale and still untracked.** Its only live content is
   item 1 above. Recommend deleting it or folding that item in; it was left
   alone because it belongs to a previous session.
4. **SID_Set is deliberately still on disk (8,000 images) despite being unusable
   for training.** This is intentional, not an oversight — the shortcut analysis
   is strong *Innovation & Problem Insight* material, and it is the only FLUX
   data available. It is excluded by flag (`--exclude-source sid_set`), never by
   deletion.
5. **`--holdout-generators` does not by itself isolate unseen-generator
   performance.** It moves held-out generators into val, but the stratified
   split has already put seen-generator fakes there too, so a single val AUC
   mixes both. The eval step must filter val to held-out generators + reals.
   Not yet built.

---

## What happened

The session began with the user supplying a prior-art artifact ("The Robustness
Gap") and notes, and asking for an architecture recommendation. It ended with a
working, trained, externally-validated model. The through-line was that **almost
every problem turned out to be in the data, not the model.**

### Architecture decisions (settled)

- **Frozen VFM + probe head**, per *Simplicity Prevails* (arXiv:2602.01738,
  fetched and read — it is past the assistant's training cutoff, so it was
  retrieved rather than recalled). Single linear layer on pooled frozen
  features; AdamW, lr 1e-3, batch 128, 2 epochs.
- Backbones are **never fine-tuned**. The organizers' day-1 advice ("fine-tune a
  pretrained backbone") is effectively CNNSpot (Wang, CVPR 2020) — the canonical
  fragile baseline. Ojha (CVPR 2023) found freezing strictly better for
  cross-generator generalisation. Recommendation on record: **build the
  fine-tuned version anyway as the baseline row of the robustness table** — the
  contrast between it and the frozen model is the submission's story.
- **User decision: NC-licensed backbones are acceptable** (rules require
  backbones be *public*; MIT/Apache applies to *custom* architectures we
  release). This admits MetaCLIP-2 (cc-by-nc-4.0) and DINOv3. If one ships,
  disclose the license in the README.
- **User decision: head is configurable `linear | mlp`, linear by default.**

### The data findings (the substance of the session)

Full detail in FINDINGS.md §1, §2b, §2c, §2d. In brief:

- **SID_Set had two independent shortcuts.** An aspect-ratio one (AIGC 100%
  square vs real 4.5%) — fixed by replacing the non-aspect-preserving resize.
  And a **composition** one: its real half (OpenImages photos) and AI half
  (FLUX) are separable at **0.93 balanced accuracy from an 8x8 greyscale
  thumbnail**. Unfixable by preprocessing. Confirmed by aspect-matched control.
- **A frozen-backbone linear probe scored AUC 1.0000 within SID_Set, then
  0.5047 balanced accuracy — chance — transferred to CIFAKE.** A perfect score
  is evidence of a leak, not skill.
- **Replaced with `TheKernel01/Tiny-GenImage`** (content-matched: ImageNet reals
  vs ImageNet-class-prompted fakes). Audited clean — all probes at or below
  chance.
- **CIFAKE was then dropped too.** 22–27% FPR alone; mixing it in doubled FPR
  while barely moving AUC.
- **Result on the external benchmark (demo-val): AUC 0.9529 → 0.9949, FPR
  0.149 → 0.019.** Roughly 8x fewer false positives on real photos, entirely
  from fixing data. Size-matched ablation confirmed it was data *quality*, not
  volume (5,000-row runs: 0.9529 → 0.9910).

### Corrections made during the session (recorded so they are not re-litigated)

- Predicted demo-val would be "inflated the same way" as SID_Set. **Wrong** —
  its composition probe is 0.63, not 0.93. It is a usable benchmark with a
  ~0.65 shortcut floor.
- Argued from `sid_set → cifake` (bacc 0.5047) that calibration was
  load-bearing. **Overstated** — on demo-val the optimal threshold moved 0.5 →
  0.5587 for +0.0034. Threshold collapse was specific to that pathological
  transfer.
- Described `data/heldout/` as a cross-generator test set. **It is not** — it
  holds the same 7 generators as train. Corrected in `config.py`.
- Initially recommended dropping CIFAKE on resolution grounds, then discovered
  it was the only content-clean data on disk, then finally dropped it anyway on
  FPR evidence. The final answer is right for a different reason than the first.
- A subagent attributed the surviving blind-probe signal to a "crop signature".
  Disproved by restricting to square-only images. **Do not take subagent
  explanations on faith — the numbers were honest, the interpretation was not.**

---

## Code changes

Commits this session (all on `main`; local-only repo, no remote):

```
f5cbcbe  Record Tiny-GenImage audit, training results, and the stale-cache trap
2396b1b  Invalidate embedding caches by manifest fingerprint, not filename
df0a6e6  Ingest Tiny-GenImage; add heldout tier and generator-aware split flags
2edefd4  Document data forensics and model bring-up results in FINDINGS.md
ef48c0f  Complete demo-val with the WildFake DALL-E half; allow embedding it
de0c28d  Add frozen vision-foundation-model backbones and probe-head training
62fef3a  Fix aspect-ratio label shortcut in the resize tail; add data shortcut audit
```

### New files

- `src/aigc_detect/backbones.py` — frozen VFM registry. `metaclip2-h` (1280d,
  224px, 630.8M), `dinov3-l` (1024d, 256px), `pe-core-l` (1024d, 336px,
  316.1M), `dinov2-g` (1536d, 518px). All verified ungated on HF. **Loads the
  vision tower only** and asserts <2e9 params.
- `src/aigc_detect/embed.py` — cached pooled embeddings, with manifest
  fingerprinting.
- `src/aigc_detect/heads.py` — `LinearHead` / `MLPHead` / `build_head`.
- `src/aigc_detect/train_head.py` — paper recipe; per-source val AUC.
- `scripts/audit_data.py` — shortcut audit + blind probe. **Permanent
  regression test.**
- `scripts/download_tiny_genimage.py`, `scripts/make_heldout.py`.

### Bugs fixed

1. **Aspect-ratio label shortcut** (`transforms.py`). `v2.Resize((S,S))` did not
   preserve aspect, stretching only real images. Survived every transform in the
   scored grid. Replaced with `build_backbone_transform()`.
2. **Stale embedding cache** (`embed.py`, `train_head.py`) — see FINDINGS.md
   trap 7. Caches were keyed by manifest *filename*, but `main.py split`
   rewrites `train.csv`/`val.csv` in place. Silent; would have produced the
   exact inverse conclusion about whether clean data helped. Now fingerprinted;
   `train_head` hard-exits on mismatch. **This fired for real** when a killed
   embed job left a 47,600-row cache under the name the new 23,800-row split
   expected.
3. **Mojibake in stdout** — em-dashes/middle-dots printed as `DALL?E` on the
   Windows console. ASCII-only in `print()` is a documented convention.

### Current CLI

```
check-env | download {cifake,sid-set,tiny-genimage} | split | audit-data
preview-augment | download-demo {...} | build-demo-val | build-heldout
list-backbones | embed --backbone K --manifest {train,val,heldout,demo-val}
train-head --backbone K [--head linear|mlp]
```

`split` flags: `--exclude-source` (repeatable), `--max-per-source N`,
`--holdout-generators G [G ...]`.

### Data on disk (all gitignored)

- `data/raw/`: CIFAKE 120,000 (unused), SID_Set 8,000 (unused, kept
  deliberately), Tiny-GenImage 28,000 (**the training source**)
- `data/heldout/`: 7,000, untouched in-distribution
- `data/demo_val/`: 13,843 (5,000 COCO real + 8,843 DALL-E) — external benchmark
- `data/embeddings/`, `models/pe-core-l__linear.pt`

### Reproduce the current model

```bash
uv run main.py split --exclude-source sid_set --exclude-source cifake
uv run main.py embed --backbone pe-core-l --manifest train
uv run main.py embed --backbone pe-core-l --manifest val
uv run main.py train-head --backbone pe-core-l
```

---

## Next steps, in priority order

1. **Robustness grid + `0.5*AUC_clean + 0.5*AUC_robust`.** The 15 pipelines
   already exist in `build_robustness_eval_transforms()`; they need an eval loop
   and a per-view embedding cache (**which must carry a manifest fingerprint**).
   This also settles FINDINGS.md §2c's open hypothesis — that part of the signal
   is "generated at 256x256" and will degrade under resize 0.5x/0.25x.
2. **`predict.py --input_dir <dir> --output preds.json`** emitting
   `[{"image_path": ..., "pred": <float 0-1>}, ...]`. Required deliverable, not
   started. Write it once and do not touch it again.
3. **Augmentation ablation** — clean-only vs clean + K precomputed degraded
   views. Only became answerable now the data is clean. See FINDINGS.md trap 4:
   with a frozen backbone only ~1,025 params train, so the usual
   overfitting justification does not apply; the real value is teaching the head
   which directions in embedding space to ignore.
4. **Backbone race** across all four. Deliberately deferred — clean-only val is
   already saturated at 0.9996, so only the robustness grid can discriminate.
5. **Cross-generator eval** (see deferred item 5).
6. **Error analysis note** (5.5.5). demo-val FPR 0.019 / TPR 0.948 is the
   starting material, plus the FPR journey 0.149 → 0.044 → 0.019.
