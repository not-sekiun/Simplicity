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

## 2e. First valid robustness grid (2026-08-29) — half the score, measured

`models/pe-core-l__linear.pt` over all 18 views of a seeded, label-balanced
2,000-row val subsample (`reports/grid__pe-core-l__val-s2000__*.csv`). This is
the **first grid run that is valid at all** — traps 8, 9, 10, 12, 13, 15 all
had to be fixed first.

```
AUC_clean                    0.9997
AUC_robust  pooled           0.8779   ->  0.5*clean+0.5*robust = 0.9388
            mean             0.9526   ->                         0.9762
            worst            0.8454   ->                         0.9226   (chain_heavy)
robustness gap  AUC          0.0471
                BAcc@t       0.2093
```

**The choice of AUC_robust definition moves the headline score by 5.4 points**
(0.9226 to 0.9762) on identical predictions. Report all three; primary is
pooled. The pooled-vs-mean spread (0.8779 vs 0.9526) is itself the diagnostic:
each view is internally well-ranked, but the score *scales* drift hard between
views, which only pooled sees.

### The failure is calibration, not signal — and it points two ways

| view | AUC | BAcc@t | TPR | FPR |
|---|---|---|---|---|
| clean | 0.9997 | 0.9915 | 0.990 | 0.007 |
| jpeg_q50 | 0.9970 | 0.8650 | **0.730** | 0.000 |
| jpeg_q30 | 0.9934 | 0.8690 | **0.740** | 0.002 |
| blur_sigma1.0 | 0.9368 | **0.5230** | 1.000 | **0.954** |
| blur_sigma2.0 | 0.8503 | **0.5070** | 0.999 | **0.985** |
| resize_0.5x | 0.9637 | **0.5310** | 1.000 | **0.938** |
| resize_0.25x | 0.8982 | **0.5170** | 0.999 | **0.965** |
| noise_sigma0.1 | 0.8840 | 0.6165 | 0.995 | 0.762 |
| chain_heavy | 0.8454 | 0.7740 | 0.821 | 0.273 |

Blur and resize collapse balanced accuracy to **chance** (0.507-0.531) while
their AUC stays 0.85-0.96. Ranking survives almost intact; the decision
boundary does not. Concretely, **FPR goes to 0.94-0.99: the model calls
essentially every blurred or downscaled REAL photo AIGC.** The plain reading
is that it learned "smooth / low-frequency -> generated" — reasonable on
256px Tiny-GenImage, where the AI half really is smoother — so removing high
frequencies from a real photo makes it look generated.

JPEG fails in the **opposite direction**: TPR drops to 0.73 with FPR pinned at
0.000, i.e. re-compressed AIGC images get called real. Its AUC barely moves
(0.9934 at q30).

**No single threshold can fix both.** This is the measured case for
per-degradation calibration, and it is invisible to AUC alone — trap 15
covers why it is also invisible if the threshold rule is chosen carelessly.

### A resolution asymmetry exists in the training data, but the head is not riding it

Measured over 1,600 sampled `train.csv` rows:

```
resolution alone (max-dim): AUC=0.5814   best bacc=0.7554 at max-dim <= 256
  label 0 (real): median 500  p10 315  p90  550   6.9% are <=256px
  label 1 (AIGC): median 256  p10 128  p90 1024  58.0% are <=256px
```

**0.7554 balanced accuracy from image size alone.** Weaker than SID_Set's
aspect-ratio shortcut (0.9775) but real, and mechanistically linked to the
blur/resize failure above: every pipeline resizes to 336px, so the asymmetry
reaches the model as *sharpness* — a 256px image upscaled to 336 is smooth, a
500px image downscaled to 336 is not.

Two pieces of evidence say the head is nevertheless **not** simply reading
resolution:

1. Correlating clean-view P(AIGC) against source resolution *within* each true
   class: `reals rho=-0.053 (p=0.09, n.s.)`, `AIGC rho=-0.233 (p=9e-14)`. If
   the model rode resolution, low-res *reals* would score high. They do not.
2. On demo-val the class/resolution relationship is **inverted** — real median
   640px vs AIGC median 1024px, against train's 500 vs 256 — and clean AUC is
   still 0.9935. A resolution-riding model collapses there; this one does not.

So the honest statement is narrower than "it learned smooth means generated":
blur and resize move real embeddings **along the head's AIGC direction**, and
the training asymmetry is the plausible reason that direction has a sharpness
component, but the failure does not reduce to a resolution readout.

Do not "fix" this by resolution-matching the training data before testing
whether augmented-view training removes it — that would be an expensive
intervention against a shortcut the model is measurably not using.

### Chains: worst view overall, but they partially self-correct

```
mean AUC, single-transform views  0.9592
mean AUC, chained views           0.9216   (delta -0.0376)
```

`chain_heavy` (blur 1.0 -> resize 0.25x -> noise 0.05 -> JPEG q30) is the
**single worst view in the grid at 0.8454 AUC**, below every individual
transform including blur_sigma2.0. Composition costs more than any one axis,
which is the whole reason trap 11 exists.

But its BAcc@t is 0.7740 — far *better* than blur_sigma2.0's 0.5070. The
chain contains both failure directions: blur pushes scores toward AIGC, JPEG
q30 pushes them toward real, and at the threshold they partially cancel. So
composition **hurts ranking most and hurts calibration least**. A grid
reporting only accuracy would have concluded chains are mild; one reporting
only AUC would have missed that the single-transform rows are the ones at
chance. Both columns are load-bearing.

Decay across chain depth is graded rather than cliff-like for this model
(0.9812 -> 0.9383 -> 0.8454), which is what the frozen-backbone architecture
is supposed to buy.

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
- [x] **Robustness grid + `0.5*AUC_clean + 0.5*AUC_robust` scoring** — DONE
      2026-08-29. `main.py embed-views` caches all 18 views (clean + the 14
      single-transform rows of 5.2 + 3 chained rows, trap 11);
      `main.py eval-grid` scores a trained head across them and writes
      `reports/grid__<backbone>__<stem>__<head>.csv`.

      Five prerequisite defects were found and fixed on the way — traps 8, 9,
      10, 12, 13. The grid was unusable as first written: its noise views
      destroyed 84.5% of every image, it normalized with the wrong constants,
      two views were nondeterministic, its seeding broke under subsampling,
      and its staleness key was too coarse. **No grid number produced before
      2026-08-29 is valid.**

      `AUC_robust` is now reported **three ways every run** rather than
      settled by argument: `pooled` (one AUC over all degraded views'
      scores concatenated — additionally penalizes score-scale drift *between*
      views), `mean` (average of per-view AUCs — blind to that drift),
      `worst` (min per-view AUC — what an adversary who picks the transform
      gets). Primary remains **pooled**; the gap between pooled and mean is
      itself a diagnostic.

      The grid also reports **one fixed threshold, chosen on clean and applied
      to every view**. Re-tuning per view is the standard way to make a
      fragile detector look robust — section 2 already recorded AUC 0.7435
      alongside balanced accuracy 0.5047 on this project's own data, i.e.
      intact ranking with the boundary in the wrong place. A deployed detector
      has one threshold.
- [ ] `predict.py --input_dir <dir> --output preds.json` emitting
      `[{"image_path": ..., "pred": <float 0-1>}, ...]` — required deliverable,
      not started.
- [ ] Backbone race across all four (see trap 1). Unblocked now that the grid
      exists — clean-only AUC cannot separate them (val is already 0.9996,
      saturated), and the grid is what actually discriminates.

      **Race on `--sample-rows 2000`, not the full manifest.** The cost is not
      4x PE-Core: dinov2-g runs at 518px (2.4x the pixels of PE-Core's 336)
      with 1.14B params vs 316M, so all four at full size is 10+ hours. At
      2,000 balanced rows an AUC's standard error is ~+/-0.005-0.01, far
      tighter than the between-backbone gaps the race is trying to resolve,
      and the seeded subsample guarantees every backbone faces the identical
      images. Spend the full grid only on the winner and runner-up.

      **Subsample images, never views.** The entire premise of the race is
      that backbones fail on *different* transforms; dropping views removes
      the signal being measured.
- [x] **Augmentation ablation** — DONE 2026-08-29, trap 4's hypothesis
      confirmed and it is the largest single win so far. Training the head on
      clean + 6 degraded views of the same 6,000 images
      (`main.py train-head-views`) moves val `0.5*clean+0.5*robust` from
      **0.9472 to 0.9878** and demo-val from **0.9631 to 0.9978**, at no cost
      to clean AUC. Every held-out view improved, including the three chains
      the head never saw, so it is not memorization. Blur/resize FPR falls
      from 0.90-0.98 to 0.05-0.21 — Run 4's calibration collapse is repaired.

      Controlled for compute: the augmented arm sees 7x more rows per epoch,
      so the clean-only control was re-run at 14 epochs. It got **worse** on
      robustness (0.8993 -> 0.8767) while clean rose to 0.9996. **Clean-only
      training trades robustness away for clean accuracy**; the gain is
      augmentation, not budget.

      Full detail and per-view tables in NARRATIVE.md Run 5.

- [ ] **Chained views in TRAINING** — the one gap Run 5 left open. Single-axis
      augmentation transfers to unseen *severities* (blur sigma2.0 gains +0.36
      balanced accuracy having only seen sigma1.0) but only weakly to
      *compositions*: the single-vs-chain AUC delta was -0.0488 before and
      -0.0481 after, i.e. unchanged. `chain_heavy` is still the worst view in
      the grid (0.8627) and binds the worst-case score. Add the chains to
      `TRAIN_VIEWS_DEFAULT` and re-run — but hold at least one chain out, or
      the result is uninterpretable for the same reason the severity holdout
      exists.
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

8. **The robustness eval grid was destroying its own noise views. FIXED
   2026-08-29.** `build_robustness_eval_transforms()` composed the noise views
   as `[resize, ToImage, ToDtype, Normalize, noise]` -- noise *after*
   normalization. `GaussianNoiseLevels` ends in `.clamp(0.0, 1.0)`, a
   valid-pixel guard that is correct in the [0, 1] domain and catastrophic
   outside it.

   After `Normalize`, 0 no longer means "black" -- it means "exactly the
   channel mean". Solving `0 <= (x - mean)/std <= 1` for the original pixel
   value gives the only band the clamp preserves:

   ```
   ch0: clamp window [0,1] == pixels [0.485, 0.714]
   ch1: clamp window [0,1] == pixels [0.456, 0.680]
   ch2: clamp window [0,1] == pixels [0.406, 0.631]
   ```

   Everything darker than the channel mean was floored to a single flat value;
   everything brighter than mean+std was saturated to another. Measured on a
   real val image at sigma=0.02, the *gentlest* level in the brief's table:

   ```
   floored to 0 (darker than channel mean)  24.5%
   saturated to 1 (brighter than mean+std)  60.0%
   total destroyed by the clamp             84.5%
   mean|noised - clean|   OLD 0.6925  ->  NEW 0.0705
   ```

   The damage is independent of sigma -- it happens at sigma -> 0 too, so the
   noise views were not measuring noise at all. A second, smaller error was
   stacked on top: the brief's sigmas (0.02/0.05/0.10) are fractions of the
   pixel range, but applied post-Normalize they were interpreted in normalized
   units (span ~4.4), making the intended perturbation ~4x too weak.

   Nothing throws. The tensor keeps its shape, dtype, and a plausible-looking
   range. This would have produced three catastrophic `noise_sigma*` rows,
   supported the conclusion "PE-Core has a severe sensor-noise weakness", and
   sent a whole session engineering against a defect living entirely in the
   eval harness. **The train pipeline had it right all along**
   (`build_train_transform` applies noise before `Normalize`); only the eval
   grid was inverted, which is the more dangerous half to get wrong.

   Fixed by splitting the tail into `to_tensor` / `normalize` and composing the
   noise views as `[resize, *to_tensor, noise, *normalize]`. Verified: the
   fixed views now scale linearly with sigma and match theory exactly (mean
   absolute deviation of a Gaussian is `sigma*sqrt(2/pi)`; at sigma=0.02 under
   0.5/0.5 normalization that predicts 0.0319, measured 0.0318).

   ```
   noise_sigma0.02  mean|d|=0.0318
   noise_sigma0.05  mean|d|=0.0780
   noise_sigma0.1   mean|d|=0.1496
   ```

9. **Normalization stats must come from the backbone, not `config.py`. FIXED
   2026-08-29.** The grid builders hardcoded `config.NORM_MEAN/NORM_STD`
   (ImageNet), while `embed.py` normalizes with `module.norm_mean` /
   `module.norm_std` -- each backbone's own (PE-Core is 0.5/0.5, MetaCLIP2 uses
   OpenAI-CLIP stats). Feeding grid tensors to a backbone would have evaluated
   every view under normalization the model was never trained with, and the
   grid's `clean` view would have silently disagreed with the already-cached
   clean embeddings it is supposed to reproduce.

   All three builders now take `norm_mean`/`norm_std`, defaulting to config so
   existing callers (`audit_data.py`) are unaffected. **Any new caller feeding
   a frozen VFM must pass the backbone's stats and its native resolution** --
   the `image_size` default is 224, but PE-Core wants 336.

10. **Neither `color_jitter` nor the noise views are deterministic by
    default.** `v2.ColorJitter` samples fresh brightness/contrast/saturation
    factors on every call, and `GaussianNoiseLevels` draws from the global
    torch RNG. For a *cached* eval view that is a correctness problem, not a
    cosmetic one: re-running the grid would score a different set of images,
    so two backbones raced against each other would not face the same test.
    `embed_views.py` seeds every stochastic view to make it byte-reproducible
    across runs, workers, and backbones. **The seed key was changed once --
    see trap 12.**

11. **A single-transform grid cannot see the failure it exists to measure.
    FIXED 2026-08-29 (3 chained views added).** The brief's 5.2 table degrades
    one axis at a time, and a grid built only from it reports fourteen numbers
    that all look survivable. Nothing on the internet arrives having survived
    exactly one transform: a screenshot that was filtered, re-uploaded and
    thumbnailed has been through four. The consistent finding in the
    literature is that baselines degrade *gracefully* per-axis and then drop
    off a **cliff** once transforms compose -- so a per-axis grid measures the
    regime detectors do not fail in, and stays silent about the one they do.

    `CHAIN_SPECS` in `transforms.py` adds `chain_light` (2 ops),
    `chain_medium` (4), `chain_heavy` (4), and `eval-grid` reports the
    single-view mean against the chained mean so the delta is explicit. Ops
    run in **physical** order -- JPEG always last (the final upload always
    re-encodes), noise before its JPEG (sensor noise exists at capture, and
    compressing noisy content is precisely the interaction that breaks
    frequency-domain detectors). Getting that order wrong makes the chain
    milder than reality while still looking like a chain.

    That ordering is why chains need `PILGaussianNoise`: the single-transform
    noise rows must keep noise in the tensor domain (trap 8), which forces it
    to the *end* of their pipeline. Chains apply the identical
    `GaussianNoiseLevels` math to the PIL image instead, via a uint8
    round-trip. The noise math is deliberately shared rather than
    reimplemented -- two versions of "add sigma noise" would drift and the
    chain rows would stop being comparable to the single rows.

12. **Seeding a stochastic eval view by ROW INDEX breaks silently the moment
    you subsample. FIXED 2026-08-29.** The obvious key for trap 10's seeding
    is `(row index, view name)`, and it is wrong for the workflow this cache
    is actually used in. A label-balanced 2,000-row subsample gives every
    image a different row index than the full manifest does, so the same photo
    receives a *different* noise realization and *different* jitter factors in
    the subsample than in the full run. The subsample's numbers then differ
    from the full run's -- by a small, plausible, entirely spurious amount
    that reads exactly like a real effect, and that grows as you compare more
    subsets.

    Keyed on the image path (`SEED_SCHEME = "path-v1"`), an image's
    degradations are identical wherever it appears: a subsample's grid is
    directly comparable to the full grid, and adding rows to a manifest does
    not rewrite every other row's noise.

13. **The cached grid's staleness check is now per-view, and it invalidated
    the pre-existing caches.** The first version hashed the whole 5.2
    parameter table into one `transform_fingerprint` shared by all views, so
    editing any severity -- or adding the chained views -- invalidated all of
    them. It is now `view_fingerprint`, a hash of that view's own canonical
    spec string (`"blur(sigma=1.0)"`, `"chain[...]"`), plus the seed scheme
    for stochastic views only.

    The spec string is built on the same line as the pipeline it describes, in
    `transforms._build_grid`, and only there. A spec table maintained
    separately from the pipelines is worse than no fingerprint at all: it
    certifies the wrong thing while looking like a check.

    **Consequence:** the 15 `pe-core-l__val__*.npz` written before this change
    carry the old key and are correctly reported STALE. Re-running the full
    val grid is ~13 min; nothing is silently wrong in the meantime.

14. **Row selection is part of the cache's identity, not just its contents.**
    `--limit N` and `--sample-rows N` both change *which* images a cache
    holds while the filename `<backbone>__<manifest stem>__<view>.npz` stays
    identical. The fingerprint catches it, so no wrong number results -- but
    alternating between a subsample and the full run would have each recompute
    destroy the other. `cache_stem()` tags the stem (`val-s2000`, `val-l500`)
    so both live on disk at once, and `eval-grid` takes the same flags to
    address the right one.

    Also: **prefer `--sample-rows` to `--limit`.** A manifest prefix is
    ordered by however the split was written and can be arbitrarily skewed in
    label and source; `stratified_sample` draws n/2 per label and allocates
    each label's quota across sources proportionally, so the subsample is a
    miniature of the manifest. Verified on val: the 2,000-row sample's
    generator mix tracks the full 4,200-row manifest's to within ~1pp.

15. **`thresholds[argmax(balanced_accuracy)]` fabricates a robustness cliff on
    near-separable data. FIXED 2026-08-29.** The grid freezes one threshold
    chosen on the clean view. The obvious way to choose it is wrong here.

    `sklearn.roc_curve` only emits thresholds at *observed score values*. When
    clean is perfectly separable — and it nearly is on this data, val AUC
    0.9996 — every threshold in the open interval (highest real score, lowest
    AIGC score] is equally optimal, but that whole interval is represented by
    a **single index**. `argmax` returns its top end, which puts the decision
    boundary flush against the lowest-scoring AIGC image, with the entire
    margin sitting unused on the real side.

    Every degraded view is then measured at an operating point where any
    downward score drift at all is an immediate false negative. Measured on
    the same 64-row cache, before vs after taking the midpoint of the margin
    instead:

    ```
    threshold   0.8364 (edge)      0.4183 (margin midpoint)
    jpeg_q30    BAcc 0.7031        BAcc 0.8750
    jpeg_q70    BAcc 0.7812        BAcc 0.9062
    blur_sig2.0 BAcc 0.5156        BAcc 0.5000   (FPR 0.91 -> 1.00)
    ```

    Both columns are "balanced-accuracy-optimal on clean" and both report
    BAcc 1.0000 there. The JPEG rows moved by up to 17 points on the choice of
    tie-break alone. Nothing warns you: the clean row looks perfect either way.

    **What the corrected numbers then show is a real asymmetry, and it is the
    interesting result:** JPEG pushes scores *down* (false negatives, TPR
    falls, FPR stays 0), while blur and resize push scores *up* — FPR goes to
    1.00, i.e. the head calls **every blurred or downscaled real photo AIGC**.
    The two failure modes point in opposite directions, so no single threshold
    can fix both. That is the concrete case for per-degradation calibration,
    and it is only visible once the threshold rule stops manufacturing
    failures of its own.
