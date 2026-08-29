# FINDINGS — data forensics and model bring-up

Session of 2026-08-29. Everything here was **measured on this machine**, not
assumed or quoted. Commands to reproduce are given so a later agent can
re-verify rather than trust this file.

Read this before touching training data or interpreting any metric.

---

## TL;DR

1. **SID_Set is unusable as training data.** Its real half (OpenImages V7
   photos) and AI half (FLUX renders) are separable at **0.93 balanced
   accuracy from an 8x8 greyscale thumbnail**. The giveaway is *composition*,
   not generation artifacts.
2. **A linear probe on frozen PE-Core features scores AUC 1.0000 on SID_Set**
   and then **0.5047 balanced accuracy (chance) when transferred to CIFAKE**.
   The high number is the shortcut, not skill.
3. **CIFAKE is the honest dataset** despite being 32x32 — its strongest signal
   is texture (real generation artifacts), and it *does* transfer.
4. A separate **aspect-ratio shortcut has been found and fixed** in
   `transforms.py`.
5. Next data source: **`TheKernel01/Tiny-GenImage`** — content-matched, 8
   generators, has a `generator` column.

---

## 1. Two independent shortcuts in SID_Set

### 1a. Aspect-ratio shortcut — FIXED

Measured over 600 sampled images per class from `data/raw/sid_set_index.csv`:

```
label 1 (AIGC):  600/600 square (100.0%)   all 1024x1024
label 0 (real):   27/600 square (  4.5%)   aspect 0.562 - 3.436
```

`width == height` alone gives ~97.75% balanced accuracy without reading a pixel.

**Why it was severe:** `transforms.py` used `v2.Resize((224, 224))`, a
*non*-aspect-preserving resize, so real images were anisotropically squashed and
AIGC images were not. Every transform in the brief's robustness grid (JPEG,
blur, noise, colour jitter, centre-crop-80%, resize-roundtrip) **preserves
aspect ratio**, so this shortcut survives the entire scored evaluation. A model
riding it would produce a flat, excellent-looking robustness table having
learned nothing.

**Fix:** `build_backbone_transform()` in `src/aigc_detect/transforms.py` —
aspect-preserving `Resize(shortest_side)` + square crop (`RandomCrop` at train,
`CenterCrop` at eval). All three pipeline builders now use it. The 5.2
parameter table is unchanged.

### 1b. Content shortcut — NOT fixable by preprocessing

After the fix the blind probe stayed high. A subagent attributed this to a
residual "crop signature" (how much of the frame survives the crop). **That
explanation is wrong.** Verified by restricting to **square-only images in both
classes**, where the crop is a no-op and no framing difference can exist:

```
SQUARE-ONLY (aspect matched, zero crop)   n=342   bacc=0.9307  auc=0.9844
NON-SQUARE reals vs square fakes (ctrl)   n=800   bacc=0.9754  auc=0.9973
```

Geometry eliminated, still 0.93. The signal is **content**.

### What exactly the model reads — feature decomposition

Same probes on both datasets, geometry controlled (all images centre-cropped
square first), n=700 each:

| Feature the probe can see | SID_Set | CIFAKE |
|---|---|---|
| Colour/tone only (6 numbers) | 0.657 | 0.613 |
| **Coarse layout — 8x8 grey (64 numbers)** | **0.935** | 0.667 |
| **Texture — high-frequency residual (4 numbers)** | 0.678 | **0.776** |

Compare **across datasets within a row**, not down a column — the feature sets
differ in size, so 64 numbers beating 4 within a column proves nothing.

**Interpretation:**

- SID_Set's giveaway is **composition** — where light and dark regions sit in
  the frame. FLUX output is composed like stock imagery (centred subject, even
  lighting, clean background); OpenImages photos are cluttered snapshots. That
  is visible in an 8x8 thumbnail.
- CIFAKE's strongest signal is **texture** (0.776 > SID_Set's 0.678) *despite
  being 32x32*. High-frequency residual statistics are genuine generation
  artifacts. CIFAKE's real/fake halves both depict the same ten CIFAR-10
  classes, so no composition shortcut can form.

### Dataset composition (root cause)

| | Real half | AI half | Matched? |
|---|---|---|---|
| **CIFAKE** | CIFAR-10, 32x32, ten classes | Stable Diffusion 1.4 rendering **the same ten classes** | **Yes, by construction** |
| **SID_Set** | OpenImages V7, ~1024px, unrestricted subjects | **FLUX**, 1024x1024, generated from its own prompts | **No — two unrelated piles** |

SID_Set generator confirmed via the SIDA paper (CVPR 2025): they trialled FLUX,
Kandinsky 3.0, SDXL and AbsoluteReality, and generated the full-synthetic set
with FLUX.

**Format shortcut is NOT present** — JPEG for both classes in both datasets.

---

## 2. Model bring-up results

### Test train (PE-Core-L/14, frozen, linear head)

Paper recipe: AdamW, lr 1e-3, batch 128, 2 epochs. Trained on 5,000 of 108,800
train rows; validated on all 19,200.

```
epoch 1/2   val_auc=0.9893   val_balanced_acc=0.9493
epoch 2/2   val_auc=0.9936   val_balanced_acc=0.9608
by source:  cifake  AUC=0.9937 (n=18000)    sid_set  AUC=0.9906 (n=1200)
```

**Do not read this as success.** 18,000 of 19,200 val rows are CIFAKE, so the
headline is mostly CIFAKE's score.

### Cross-source transfer — the decisive test

Fresh logistic heads on the cached val embeddings:

```
WITHIN  cifake  -> cifake     AUC=0.9977   bacc=0.9777
WITHIN  sid_set -> sid_set    AUC=1.0000   bacc=1.0000
CROSS   cifake  -> sid_set    AUC=0.9555   bacc=0.8750
CROSS   sid_set -> cifake     AUC=0.7435   bacc=0.5047
```

- **SID_Set scores a perfect 1.0000 on itself.** Nothing here is perfectly
  solvable; a perfect score means the model found something other than the
  actual question.
- **SID_Set-trained -> CIFAKE collapses to chance accuracy (0.5047).**
  Everything it learned was dataset-specific.
- **CIFAKE-trained -> SID_Set holds at 0.8750.** The content-clean dataset
  teaches something transferable.
- **AUC 0.7435 with bacc 0.5047** in that same row: real ranking signal, but the
  decision threshold lands in completely the wrong place on new data. Direct
  evidence that **calibration is required work, not polish**.

Caveat: this diagnostic used validation embeddings for both halves and the
within-sid_set numbers come from 600 train / 600 test. Directions are solid;
treat exact decimals as indicative.

---

## 2b. Demo-val is a usable external benchmark (2026-08-29, after manual WildFake fetch)

`data/demo_val/demo_val.csv` is now complete: **13,843 images** = 5,000 COCO
val2017 reals + 8,843 WildFake "DALL-E Advanced" AIGC (exactly the count the
brief cites). Leakage guard passed: no filename overlap with train/val.

**Audited for the same shortcuts** (same probes as section 1, geometry controlled):

```
geometry:  COCO real   3.2% square (aspect 0.561-3.787)
           DALL-E     84.8% square (aspect 0.571-1.750)

probes (n=800):  colour/tone 0.5737 | coarse layout 0.6345 | texture 0.5949
```

**An earlier prediction in this project was that demo-val would be inflated the
same way SID_Set is. That was too pessimistic and is corrected here.** The
aspect shortcut is real (3.2% vs 84.8% square, worth ~0.91 balanced accuracy if
read raw) but `build_backbone_transform`'s square crop neutralises it. The
*composition* shortcut — the fatal one — is only **0.63 here vs SID_Set's 0.93**.
COCO's cluttered snapshots against DALL-E's varied output are simply not as
separable as OpenImages against FLUX.

**Treat ~0.65 as demo-val's shortcut floor.** A score in the 0.65-0.70 range is
no better than cheating; above that is genuine signal.

### First external evaluation

`pe-core-l` + linear head (trained on 5,000 CIFAKE+SID_Set rows), applied to
demo-val — unseen data, and DALL-E is a generator absent from all training data:

```
AUC               = 0.9529
bacc @0.5         = 0.8908
bacc @best thresh = 0.8942   (threshold 0.5587 -- barely moved from 0.5)
mean p(real)=0.245   mean p(aigc)=0.801
flagged as AIGC:  real 14.9%   aigc 93.0%
```

Reading:

- **0.9529 against a ~0.65 shortcut floor is real signal**, on an unseen dataset
  and an unseen generator. The strongest evidence so far that the approach works.
- **The threshold barely needed moving** (0.5 -> 0.5587, +0.0034 bacc). Note this
  *contradicts* the calibration alarm from the `sid_set -> cifake` transfer
  (bacc 0.5047 at default threshold). Threshold collapse is not universal — it
  was specific to that pathological transfer. Calibration still matters for the
  scored robustness grid, but the case for it is weaker than section 2 implied.
- **14.9% false-positive rate on real photos** is the number for the error
  analysis deliverable (5.5.5). On a platform where most images are real, that
  is not deployable — good material for the trade-off discussion.

Caveats: head trained on only 5,000 of 108,800 rows, and on data known to
contain the SID_Set shortcut; demo-val is class-imbalanced 5,000/8,843 (use
balanced accuracy and AUC, never raw accuracy); and this is **clean** demo-val,
with no robustness transforms applied yet.

---

## 2c. Tiny-GenImage ingested and audited (2026-08-29) -- the content shortcut is gone

Ingested `TheKernel01/Tiny-GenImage`. HF `train` -> `data/raw/tiny_genimage/`
(28,000 images, 14,000 real / 14,000 AIGC); HF `validation` -> `data/heldout/`
(7,000, 3,500/3,500). Every image re-encoded to JPEG q95 on write, which closes
the format shortcut by construction (verified decisively: a source PNG in the
parquet lands on disk as JPEG).

**Only 7 fake generators are actually present, not 8.** The ClassLabel declares
`['Real','ADM','BigGAN','GLIDE','Midjourney','SD14','SD15','VQDM','Wukong']` but
**SD14 has zero rows in either split**. Present: ADM, BigGAN, GLIDE, Midjourney,
SD15, VQDM, Wukong -- 2,000 each in train, 500 each in heldout.

**`data/heldout/` is an in-distribution test set, not a cross-generator one.**
It contains the same 7 generators as train. For unseen-generator evaluation use
`main.py split --holdout-generators`.

### Shortcut audit -- compare across datasets within a row

| Probe (geometry controlled) | SID_Set | CIFAKE | **Tiny-GenImage** |
|---|---|---|---|
| colour/tone only (6 nums) | 0.657 | 0.613 | **0.469** |
| coarse layout 8x8 (64 nums) | **0.935** | 0.667 | **0.477** |
| texture high-freq (4 nums) | 0.678 | 0.776 | **0.495** |

All three at or below chance. **The content shortcut that made SID_Set unusable
does not exist here** -- GenImage's fakes are generated from ImageNet class
prompts against ImageNet reals, so both halves depict the same subjects.

### The geometry skew is as bad as SID_Set's, but it is handled

```
label 0 (real): 4.3% square,  aspect p50 1.333, short side p50 375
label 1 (AIGC): 100% square,  aspect p50 1.000, short side p50 256
```

Same structure as SID_Set's aspect shortcut. `build_backbone_transform`'s square
crop removes it from the tensor -- and the probes above were run *with* that
crop applied, which is why they sit at chance.

### The residual frequency signal is real, not a resampling artifact

Reals get **downscaled** to the backbone's input (375 -> 336) while fakes get
**upscaled** (256 -> 336), so differing resample histories were a live concern.

```
A  current pipeline (crop -> 336, paths differ)      bacc=0.6055
B  resampling-normalised (both 200 -> 336)           bacc=0.7227
```

**Normalising the resampling path made the signal stronger, not weaker.** A
resampling shortcut would have collapsed toward chance under B; instead the
differing paths were partly *masking* a genuine generator fingerprint. So the
~0.6-0.72 is real frequency-domain signal of the kind frequency-based detectors
are supposed to use, not a leak.

**Open hypothesis, to settle in the robustness grid:** the fakes are natively
square at fixed resolutions (256/512/1024), so part of this signal may be "was
this generated at 256x256". If so it will degrade under the grid's resize
0.5x/0.25x transforms. Measure it; do not assume either way.

---

## 2d. Training on clean data (2026-08-29) -- what the data swap actually bought

Split: `main.py split --exclude-source sid_set --max-per-source 28000`
(47,600 train / 8,400 val, CIFAKE and Tiny-GenImage equally weighted).
Backbone `pe-core-l`, linear head, paper recipe.

```
val     (in-distribution)          n= 8400  AUC=0.9978  bacc=0.9786  FPR=0.021
heldout (untouched, in-dist)       n= 7000  AUC=0.9985  bacc=0.9827  FPR=0.016
demo-val (EXTERNAL, unseen gen)    n=13843  AUC=0.9947  bacc=0.9665  FPR=0.044
```

demo-val, like-for-like against the contaminated model on identical data:

```
contaminated (CIFAKE+SID_Set, 5k rows)  AUC=0.9529  bacc=0.8908  FPR=0.149
clean        (CIFAKE+TinyGenImage)      AUC=0.9947  bacc=0.9665  FPR=0.044
```

False positives on real photos fell 14.9% -> 4.4%.

### It was data quality, not data volume

Two variables changed at once (data *and* 5,000 -> 47,600 rows), so this was
size-matched at 5,000 rows. demo-val, sklearn logistic heads throughout so the
rows are internally comparable:

| Training data | rows | AUC | bacc | FPR |
|---|---|---|---|---|
| CIFAKE + SID_Set *(old)* | 5,000 | 0.9529 | 0.8908 | 0.149 |
| CIFAKE + Tiny-GenImage | 5,000 | 0.9863 | 0.9395 | 0.079 |
| **Tiny-GenImage only** | 5,000 | **0.9910** | 0.9534 | 0.029 |
| CIFAKE only | 5,000 | 0.9410 | 0.8454 | 0.268 |
| CIFAKE + Tiny-GenImage | 47,600 | 0.9919 | 0.9605 | 0.032 |
| Tiny-GenImage only | 23,800 | 0.9907 | 0.9348 | 0.013 |

At matched size the data swap alone moves 0.9529 -> 0.9910. Going 5,000 ->
47,600 rows moved AUC only 0.9863 -> 0.9919. **Replacing SID_Set was the win;
volume was nearly irrelevant.**

### CIFAKE is actively harmful and was dropped

Alone it reaches 0.93-0.94 AUC with a **22-27% false-positive rate**. Mixed into
Tiny-GenImage it barely moves AUC (0.9907 -> 0.9919) while more than doubling
FPR (0.013 -> 0.032). The 32x32 -> 336 upsampling domain gap damages calibration
even where it does not damage ranking. Final training data is Tiny-GenImage
only: `main.py split --exclude-source sid_set --exclude-source cifake`.

### FINAL model: Tiny-GenImage only (23,800 train / 4,200 val)

`main.py split --exclude-source sid_set --exclude-source cifake`, backbone
`pe-core-l`, linear head, paper recipe. Saved to `models/pe-core-l__linear.pt`.

```
val      (in-distribution)        n= 4200  AUC=0.9996  bacc=0.9902  FPR=0.006  TPR=0.986
heldout  (untouched, in-dist)     n= 7000  AUC=0.9997  bacc=0.9913  FPR=0.006  TPR=0.989
demo-val (EXTERNAL, unseen gen)   n=13843  AUC=0.9949  bacc=0.9647  FPR=0.019  TPR=0.948
```

Dropping CIFAKE confirmed the ablation's prediction: demo-val AUC is unchanged
(0.9947 -> 0.9949) but **FPR more than halves, 0.044 -> 0.019**. CIFAKE was
damaging calibration without contributing ranking power.

False-positive rate on real photos across the whole session:

```
contaminated (CIFAKE+SID_Set)   FPR=0.149
clean (CIFAKE+Tiny-GenImage)    FPR=0.044
FINAL (Tiny-GenImage only)      FPR=0.019     ~8x reduction
```

**Every number in this section is clean-image only.** The robustness grid --
half the competition score -- is not yet measured, and the open hypothesis from
section 2c (that part of the signal is "generated at 256x256", which would
degrade under resize 0.5x/0.25x) is still unresolved. Treat val/heldout AUC
~0.999 as in-distribution and near-meaningless on its own; demo-val 0.9949
against a ~0.65 shortcut floor is the number with content.

---

## 3. What was built

### Wave 1
- `scripts/audit_data.py` (new) — shortcut audit: per (source, label) format /
  resolution / aspect distributions, plus a **blind probe** (logistic regression
  on 16x16 greyscale). `--transform` runs it on tensors from the real eval
  pipeline. **Clearing ~70% means a shortcut survives.** Keep as a permanent
  regression test.
- `src/aigc_detect/transforms.py` — added `build_backbone_transform()`,
  repointed all three builders. Parameter table untouched.
- `main.py` — `audit-data` subcommand.

### Wave 2
- `src/aigc_detect/backbones.py` — frozen-backbone registry. Asserts vision-tower
  params < 2e9 and prints the count.
- `src/aigc_detect/embed.py` — `precompute_embeddings(...)`, caches to
  `data/embeddings/<backbone>__<manifest>.npz` (arrays: `embeddings`, `labels`,
  `sources` + metadata). Supports `--limit` and `--force`.
- `src/aigc_detect/heads.py` — `LinearHead`, `MLPHead`, `build_head(kind, in_dim)`.
- `src/aigc_detect/train_head.py` — paper recipe, standardises on **train**
  statistics only, reports val AUC **broken down by source**, saves to
  `models/<backbone>__<head>.pt`.
- `src/aigc_detect/config.py` — added `EMBEDDINGS_DIR`.
- `main.py` — `list-backbones`, `embed`, `train-head`.
- `pyproject.toml` — added `timm`, `transformers`.

### Backbone registry (all verified ungated on HF)

| key | checkpoint | loader | dim | native res | vision-tower params |
|---|---|---|---|---|---|
| `metaclip2-h` | `facebook/metaclip-2-worldwide-huge-quickgelu` | transformers | 1280 | 224 | 630,766,080 |
| `dinov3-l` | `timm/vit_large_patch16_dinov3.lvd1689m` | timm | 1024 | 256 | ~303M |
| `pe-core-l` | `timm/vit_pe_core_large_patch14_336.fb` | timm | 1024 | 336 | 316,102,656 |
| `dinov2-g` | `facebook/dinov2-giant` | transformers | 1536 | 518 | ~1.14B |

Notes:
- **Ship the vision tower only.** The full MetaCLIP2 checkpoint is 1.86B params;
  the vision tower is 630.8M. `transformers==5.16.1` has `MetaClip2VisionModel`
  and loads it directly — the timm fallback was implemented but not needed.
- The four **native resolutions differ**, so `IMAGE_SIZE=224` is not a global
  constant for embedding; the registry carries per-backbone resolution.
- Normalisation comes from each backbone's own config (PE-Core:
  mean/std = 0.5 via `timm.data.resolve_model_data_config`; MetaCLIP2: OpenAI-CLIP
  stats), **not** ImageNet stats.
- Embedding speed ~82 img/s for `pe-core-l`; full `train.csv` (108,800 rows)
  is ~22 minutes.

**Only `pe-core-l` has been run.** The other three are wired but untouched —
deliberately, see below.

---

## 4. Reference: the paper being implemented

*Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual
Foundation Models* — Zhou, He, Lin, Fan, Ding, Li. arXiv:2602.01738.

- **Head:** a *single linear layer* on the **pooled output** of a frozen backbone.
- **Optimiser:** AdamW, lr 1e-3, batch 128, **2 epochs**.
- **Training data:** GenImage's **Stable Diffusion v1.4 subset only** (one generator).
- **Preprocessing:** "resized and center-cropped to the native resolution of each
  model **without any additional data augmentation**" — stated as deliberate.
- **Robustness:** MetaCLIP2 ~93% under JPEG q65-95; DINOv3 89-91% under blur;
  **PE-CLIP degrades 96% -> 78% under blur**.
- **Limitations:** blind to VAE reconstruction (~3-5%), localized editing 50-60%,
  degrades under recapture/transmission.

Backbones it evaluates: MetaCLIP-H/14, MetaCLIP-2 Worldwide Giant, SigLIP-L/16,
SigLIP-2 Giant/16, PE-Core-L/14, DINOv2-giant, DINOv3-ViT-7B/16.

---

## 5. Decisions taken

- **NC-licensed backbones are acceptable.** Competition rules require backbones
  be *public*; MIT/Apache is required only for *custom* architectures we release.
  (User decision.)
- **Head is configurable `linear | mlp`, linear by default** to stay
  paper-faithful. (User decision.)
- **SID_Set is not training data**, but stays in the repo: the shortcut analysis
  is strong submission material for *Innovation & Problem Insight*, and it is our
  only FLUX data (usable as one clearly-caveated cross-generator row).
- **CIFAKE stays** — content-clean, genuine texture signal, transfers.

---

## 6. Next data source

**`TheKernel01/Tiny-GenImage`** — ungated, parquet, streams like SID_Set did.

```
train      28,000 images
validation  7,000
download   ~8.4 GB
license    cc-by-nc-sa-4.0
columns    image | label (real/fake) | generator
generators Real, ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM, Wukong
```

Why: **content-matched** (ImageNet reals vs ImageNet-class-prompted fakes, so no
composition shortcut can form), **8 generators with a `generator` column** so a
genuine held-out-generator split becomes possible for the first time, and it is
the **same dataset family the paper trains on**.

### Rejected alternatives (do not retry)

- **`bitmind/GenImage_*`** — contains only an `image` column: **no labels, no
  real images**, just generator dumps. MidJourney alone is 200GB. Unusable.
- **`OwensLab/CommunityForensics-Small` / `-Eval`** — ungated, cc-by-nc-sa-4.0,
  viable but deprioritised in favour of Tiny-GenImage (smaller, already labelled,
  already generator-tagged).
- **ModelScope (WildFake)** — API and Python SDK both hang indefinitely from this
  network. Manual browser download only.

---

## 7. Open items

- [x] **WildFake DALL-E Advanced** — DONE (manual fetch, 2026-08-29). 8,843
      images in nested subdirs; `index_wildfake_dalle_advanced()` already used
      `rglob` so no code change was needed. `demo_val.csv` now has 13,843 rows.
      `main.py embed --manifest demo-val` added (EVALUATION ONLY — `train_head`
      hardcodes TRAIN/VAL manifests and cannot see it).
- [x] Ingest Tiny-GenImage — DONE. 28,000 train + 7,000 heldout, `generator`
      column, generator-aware split flags (`--exclude-source`,
      `--max-per-source`, `--holdout-generators`). See section 2c.
- [x] Audit the new data before training on it — DONE, section 2c. Content
      shortcut absent (all probes at or below chance).
- [x] Train on clean data — DONE, section 2d. Final model
      `models/pe-core-l__linear.pt`, demo-val AUC 0.9949, FPR 0.019.
- [x] Add `data/embeddings/`, `models/`, `data/heldout/` to `.gitignore` — DONE.
- [ ] **Robustness grid + `0.5*AUC_clean + 0.5*AUC_robust` scoring — the single
      biggest gap.** Half the competition score is unmeasured. All 15 pipelines
      already exist in `build_robustness_eval_transforms()`; they need an eval
      loop and a per-view embedding cache (which must carry a manifest
      fingerprint — see trap 7). Settles section 2c's open hypothesis about the
      "generated at 256x256" signal degrading under resize 0.5x/0.25x.
- [ ] `predict.py --input_dir <dir> --output preds.json` emitting
      `[{"image_path": ..., "pred": <float 0-1>}, ...]` — required deliverable,
      not started.
- [ ] Backbone race across all four (see trap 1). Deferred until the robustness
      grid exists — clean-only AUC cannot separate them (val is already 0.9996,
      saturated), and the grid is what actually discriminates.
- [ ] Augmentation ablation: clean-only vs clean + K precomputed degraded views,
      scored on the grid. See trap 4 — this only became answerable now that the
      data is clean.
- [ ] Cross-generator evaluation using `--holdout-generators`. NOTE: val will
      mix seen and unseen generators, so a single val AUC does not isolate
      unseen-generator performance. The eval step must filter val to the
      held-out generators plus reals. Not yet built.
- [ ] Calibration (temperature + threshold per degradation bucket). Note the
      case for this is weaker than section 2 implied — see section 2b.
- [ ] Error analysis note (5.5.5). demo-val FPR 0.019 / TPR 0.948 is the
      starting material.

---

## 8. Traps for future agents

1. **Do not race backbones on shortcut-contaminated data.** It ranks them by how
   well they exploit the shortcut, not by detection ability. Fix data first.
2. **Do not trust a high number.** On this task, AUC near 1.0 is evidence of a
   leak. Always run cross-source transfer before believing anything.
3. **Augmentation cannot fix a content shortcut.** JPEG, blur, noise, jitter and
   resize do not change where the subject sits in the frame; an 8x8 composition
   signature survives all of them. Content mismatch is a *data* problem.
4. **Augmentation's usual justification does not apply here.** With a frozen
   backbone only ~1,025 parameters are trained, so there is nearly nothing to
   overfit. Its real value in this setup is different: showing the head paired
   clean/degraded embeddings of the same image teaches it which directions in
   embedding space to ignore. Get this without losing the embedding cache by
   precomputing **K fixed degraded views** (from
   `build_robustness_eval_transforms()`) rather than random per-epoch
   augmentation. Whether it helps is an **ablation**, and it must be run
   *after* the data is fixed.
5. **Never train on `data/demo_val/`** (brief 5.4). It lives in a directory
   `make_splits.py` structurally never globs.
6. **ASCII only in anything passed to `print()`** — Windows console mojibakes
   non-ASCII. Unicode is fine in docstrings and Markdown.
7. **The embedding cache is keyed by manifest FILENAME, not contents — always
   confirm it says it matched.** `main.py split` rewrites
   `data/processed/train.csv` and `val.csv` **in place**: same filename,
   completely different images. The cache file is
   `data/embeddings/<backbone>__<manifest stem>.npz`, so a re-split silently
   collides with the previous run's cache.

   Originally `embed` skipped whenever a file of that name existed, with no
   check that the images matched. That would have handed back embeddings
   computed from the *old* (SID_Set-contaminated) data while we believed we
   were measuring the new clean data — printing "already exists -- skipping",
   which reads like success. The resulting AUC would have looked entirely
   plausible, and the conclusion would have been the exact inverse of the truth:
   "replacing the dirty data made no difference."

   Fixed in `embed.py`: `manifest_fingerprint()` hashes the manifest's
   `image_path` column in order and stores it in the `.npz`; a mismatch (or a
   missing fingerprint, i.e. any cache written before this existed) forces a
   recompute. **Expected healthy output is one of:**
   ```
   [embed] ... already exists and matches the manifest -- skipping
   [embed] ... is STALE (manifest changed) -- recomputing
   ```
   If you ever see a skip without the words "matches the manifest", the check
   has been removed or bypassed — stop and investigate before trusting any
   number downstream.

   `train_head` also verifies both fingerprints before training and exits with
   a `STALE EMBEDDINGS` error rather than proceeding -- that is where wrong data
   would actually be consumed, and `embed`'s check alone does not cover it (a
   run killed part-way leaves the previous `.npz` in place, which is exactly
   what happened once on 2026-08-29). Healthy output is:
   ```
   [train-head] train embeddings match train.csv (fingerprint OK)
   [train-head] val embeddings match val.csv (fingerprint OK)
   ```

   The general lesson, which applies beyond this one file: **a cache keyed on a
   name rather than on contents produces confident wrong answers instead of
   visible failures.** Anything else that caches per-manifest (future robustness
   grids, per-view embedding caches) needs the same fingerprint.
