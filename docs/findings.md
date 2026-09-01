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

**Fix:** `build_backbone_transform()` in `src/aigc_detect/data/transforms.py` —
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

## 2f. Run 6 full grid — chain-augmented head, all 18 scored views

`models/pe-core-l__linear__augchain.pt`. Trained on 6,000 images x 11 views
(`TRAIN_VIEWS_DEFAULT` + the four `trainchain_*` compositions). The three
scored chains and eight of the fourteen single-transform severities were
**never seen in training**, and are marked below.

Each head gets its own operating threshold, chosen on its own clean view and
then frozen across all 18 views (trap 15). `BAcc@t` uses that threshold;
`BAcc@0.5` is shown alongside for reference. Thresholds: val **0.4117**,
demo-val **0.2389**.

Reproduce: `uv run main.py eval-grid --backbone pe-core-l --manifest {val,demo-val}
--sample-rows 2000 --head models/pe-core-l__linear__augchain.pt`

### val-s2000 — Tiny-GenImage, IN-DISTRIBUTION

1,000 real / 1,000 AIGC across 7 generators (GLIDE 167, BigGAN 164, ADM 146, Wukong 140, Midjourney 137, SD15 125, VQDM 121). Same generators and same real source as train.

| view | in training? | AUC | BAcc@t | BAcc@0.5 | TPR | FPR |
|---|---|---|---|---|---|---|
| `clean` | n/a (baseline) | 0.9981 | 0.9770 | 0.9760 | 0.9710 | 0.0170 |
| `jpeg_q90` | **held out** | 0.9987 | 0.9755 | 0.9730 | 0.9600 | 0.0090 |
| `jpeg_q70` | trained | 0.9987 | 0.9850 | 0.9835 | 0.9860 | 0.0160 |
| `jpeg_q50` | **held out** | 0.9966 | 0.9600 | 0.9565 | 0.9330 | 0.0130 |
| `jpeg_q30` | **held out** | 0.9918 | 0.9095 | 0.9010 | 0.8280 | 0.0090 |
| `blur_sigma0.5` | **held out** | 0.9971 | 0.9725 | 0.9740 | 0.9760 | 0.0310 |
| `blur_sigma1.0` | trained | 0.9929 | 0.9630 | 0.9645 | 0.9760 | 0.0500 |
| `blur_sigma2.0` | **held out** | 0.9646 | 0.8675 | 0.8865 | 0.9650 | 0.2300 |
| `resize_0.5x` | trained | 0.9920 | 0.9535 | 0.9585 | 0.9790 | 0.0720 |
| `resize_0.25x` | **held out** | 0.9546 | 0.8795 | 0.8850 | 0.9200 | 0.1610 |
| `noise_sigma0.02` | **held out** | 0.9791 | 0.9105 | 0.9190 | 0.9610 | 0.1400 |
| `noise_sigma0.05` | trained | 0.9597 | 0.8955 | 0.9000 | 0.9210 | 0.1300 |
| `noise_sigma0.1` | **held out** | 0.9179 | 0.8445 | 0.8455 | 0.9060 | 0.2170 |
| `color_jitter` | trained | 0.9969 | 0.9755 | 0.9705 | 0.9700 | 0.0190 |
| `center_crop_80` | trained | 0.9970 | 0.9725 | 0.9680 | 0.9600 | 0.0150 |
| `chain_light` *(chain)* | **held out** | 0.9952 | 0.9665 | 0.9675 | 0.9790 | 0.0460 |
| `chain_medium` *(chain)* | **held out** | 0.9591 | 0.8775 | 0.8655 | 0.8000 | 0.0450 |
| `chain_heavy` *(chain)* | **held out** | 0.9182 | 0.8455 | 0.8440 | 0.8760 | 0.1850 |

Summary: AUC_clean 0.9981 | AUC_robust pooled **0.9803** / mean 0.9771 / worst 0.9179
| score 0.5*clean+0.5*pooled = **0.9892**, worst-case = **0.9580**
| robustness gap AUC 0.0210, BAcc@t 0.0503.

### demo_val-s2000 — COCO val2017 + WildFake DALL-E Advanced, EXTERNAL

1,000 COCO reals / 1,000 WildFake DALL-E AIGC. Never trained on (brief 5.4). Doubles as an unseen-generator AND unseen-real-source test: DALL-E is not among the 7 training generators and COCO is not the training real source.

| view | in training? | AUC | BAcc@t | BAcc@0.5 | TPR | FPR |
|---|---|---|---|---|---|---|
| `clean` | n/a (baseline) | 0.9998 | 0.9920 | 0.9945 | 0.9980 | 0.0140 |
| `jpeg_q90` | **held out** | 1.0000 | 0.9980 | 0.9985 | 0.9980 | 0.0020 |
| `jpeg_q70` | trained | 0.9998 | 0.9910 | 0.9950 | 0.9970 | 0.0150 |
| `jpeg_q50` | **held out** | 0.9999 | 0.9950 | 0.9975 | 0.9990 | 0.0090 |
| `jpeg_q30` | **held out** | 0.9999 | 0.9980 | 0.9950 | 0.9980 | 0.0020 |
| `blur_sigma0.5` | **held out** | 0.9997 | 0.9880 | 0.9940 | 0.9980 | 0.0220 |
| `blur_sigma1.0` | trained | 0.9990 | 0.9695 | 0.9855 | 0.9980 | 0.0590 |
| `blur_sigma2.0` | **held out** | 0.9951 | 0.9170 | 0.9555 | 0.9920 | 0.1580 |
| `resize_0.5x` | trained | 0.9989 | 0.9740 | 0.9870 | 0.9960 | 0.0480 |
| `resize_0.25x` | **held out** | 0.9958 | 0.9425 | 0.9675 | 0.9900 | 0.1050 |
| `noise_sigma0.02` | **held out** | 0.9943 | 0.9465 | 0.9630 | 0.9840 | 0.0910 |
| `noise_sigma0.05` | trained | 0.9789 | 0.9145 | 0.9265 | 0.9550 | 0.1260 |
| `noise_sigma0.1` | **held out** | 0.9470 | 0.8605 | 0.8800 | 0.9280 | 0.2070 |
| `color_jitter` | trained | 0.9997 | 0.9890 | 0.9940 | 0.9980 | 0.0200 |
| `center_crop_80` | trained | 0.9998 | 0.9935 | 0.9935 | 0.9970 | 0.0100 |
| `chain_light` *(chain)* | **held out** | 0.9996 | 0.9800 | 0.9900 | 0.9980 | 0.0380 |
| `chain_medium` *(chain)* | **held out** | 0.9993 | 0.9815 | 0.9870 | 0.9940 | 0.0310 |
| `chain_heavy` *(chain)* | **held out** | 0.9946 | 0.9175 | 0.9490 | 0.9880 | 0.1530 |

Summary: AUC_clean 0.9998 | AUC_robust pooled **0.9970** / mean 0.9942 / worst 0.9470
| score 0.5*clean+0.5*pooled = **0.9984**, worst-case = **0.9734**
| robustness gap AUC 0.0056, BAcc@t 0.0299.

### Per-view AUC progression on val-s2000

All three heads trained on the **same 6,000 images**, so the only variable is
which views they saw. This is the controlled comparison; Run 4's head used
23,800 clean rows and is not a valid control for it.

| view | in training? | Run 5 control<br>clean-only | Run 5<br>augmented | Run 6<br>+chains | Run 6 vs control |
|---|---|---|---|---|---|
| `clean` | n/a (baseline) | 0.9987 | 0.9985 | 0.9981 | -0.0006 |
| `jpeg_q90` | **held out** | 0.9974 | 0.9984 | 0.9987 | +0.0013 |
| `jpeg_q70` | trained | 0.9965 | 0.9982 | 0.9987 | +0.0023 |
| `jpeg_q50` | **held out** | 0.9936 | 0.9958 | 0.9966 | +0.0030 |
| `jpeg_q30` | **held out** | 0.9886 | 0.9906 | 0.9918 | +0.0032 |
| `blur_sigma0.5` | **held out** | 0.9973 | 0.9977 | 0.9971 | -0.0002 |
| `blur_sigma1.0` | trained | 0.9498 | 0.9922 | 0.9929 | +0.0432 |
| `blur_sigma2.0` | **held out** | 0.8668 | 0.9584 | 0.9646 | +0.0978 |
| `resize_0.5x` | trained | 0.9661 | 0.9907 | 0.9920 | +0.0260 |
| `resize_0.25x` | **held out** | 0.8984 | 0.9486 | 0.9546 | +0.0562 |
| `noise_sigma0.02` | **held out** | 0.9630 | 0.9776 | 0.9791 | +0.0161 |
| `noise_sigma0.05` | trained | 0.9334 | 0.9575 | 0.9597 | +0.0263 |
| `noise_sigma0.1` | **held out** | 0.8803 | 0.9119 | 0.9179 | +0.0377 |
| `color_jitter` | trained | 0.9972 | 0.9975 | 0.9969 | -0.0003 |
| `center_crop_80` | trained | 0.9976 | 0.9972 | 0.9970 | -0.0006 |
| `chain_light` *(chain)* | **held out** | 0.9775 | 0.9880 | 0.9952 | +0.0177 |
| `chain_medium` *(chain)* | **held out** | 0.9268 | 0.9435 | 0.9591 | +0.0323 |
| `chain_heavy` *(chain)* | **held out** | 0.8265 | 0.8627 | 0.9182 | +0.0917 |

Reading it:

- **Clean is untouched** (0.9987 -> 0.9981). Robustness was not bought with
  clean accuracy.
- **The worst rows improve most.** `blur_sigma2.0` +0.0978 and `chain_heavy`
  +0.0917 — both held out.
- **Held-out severities track their trained siblings.** `blur_sigma1.0`
  (trained) +0.0432 and `blur_sigma2.0` (held out) +0.0978; `resize_0.5x`
  (trained) +0.0260 and `resize_0.25x` (held out) +0.0562. The unseen, harsher
  severity gains *more* than the trained one, which is the opposite of what
  memorization would produce.
- **Chains only move once chains are trained on.** Run 5 gave `chain_heavy`
  +0.0362; Run 6 added a further +0.0555. Composition is a separate axis from
  severity and had to be trained separately.
- **The rows that barely move were already saturated** (`color_jitter`,
  `center_crop_80`, `blur_sigma0.5`, all >0.997 from the start).

### Composition penalty — chain AUC minus its own weakest component's AUC

The primary composition diagnostic. The chained-mean-minus-single-mean delta
conflates depth with severity choice, since the single-view mean averages in
`jpeg_q90` and `blur_sigma0.5` at ~0.998. Comparing a chain to its own parts
holds severity fixed by construction, so what remains is the cost of composing.

| chain | components | Run 5 control | Run 5 aug | Run 6 +chains |
|---|---|---|---|---|
| `chain_light` | `resize_0.5x`, `jpeg_q70` | +0.0114 (vs `resize_0.5x`) | -0.0027 (vs `resize_0.5x`) | +0.0031 (vs `resize_0.5x`) |
| `chain_medium` | `center_crop_80`, `color_jitter`, `resize_0.5x`, `jpeg_q50` | -0.0393 (vs `resize_0.5x`) | -0.0473 (vs `resize_0.5x`) | -0.0329 (vs `resize_0.5x`) |
| `chain_heavy` | `blur_sigma1.0`, `resize_0.25x`, `noise_sigma0.05`, `jpeg_q30` | -0.0719 (vs `resize_0.25x`) | -0.0860 (vs `resize_0.25x`) | -0.0364 (vs `resize_0.25x`) |

`chain_heavy`'s penalty more than halves (-0.0860 -> -0.0364) and
`chain_light`'s goes positive. Note the control's `chain_light` penalty was
*also* positive (+0.0114) — at that point `resize_0.5x` was itself so broken
(0.9661, FPR 0.90) that adding JPEG on top made the input easier, not harder.
A positive penalty is only good news when the components are healthy.

**Binding constraint is now `noise_sigma0.1` (0.9179), a held-out severity,**
not a chain. Heavy sensor noise is the next axis to attack.

---

## 2g. What the paper actually says about backbones (fetched 2026-08-29)

Pulled from the full text of arXiv:2602.01738, not from memory. **Our registry
does not match the paper's variants**, which changes how much its ranking can
be trusted here.

### The paper's numbers

| Backbone | GenImage avg acc | In-the-wild avg | AIGIHolmes | AIGI-Now |
|---|---|---|---|---|
| DINOv3-Linear | **0.964** | **0.940** | 0.972 | 0.864 |
| SigLIP2-Linear | 0.945 | 0.822 | — | — |
| PE-CLIP-Linear | 0.938 | 0.899 | **0.978** | 0.891 |
| MetaCLIP2-Linear | 0.892 | 0.842 | 0.942 | **0.907** |
| DINOv2-Linear | 0.852 | 0.636 | — | — |
| SigLIP-Linear | 0.851 | 0.610 | — | — |
| MetaCLIP-Linear | 0.766 | 0.654 | — | — |

### Robustness (Table 7, Chameleon) — the part that matters to us

| Backbone | Base | JPEG-65 | **Blur sigma=2.0** |
|---|---|---|---|
| MetaCLIP2-Linear | 0.930 | 0.898 | **0.932** (improves) |
| DINOv3-Linear | 0.914 | 0.891 | **0.891** |
| PE-CLIP-Linear | **0.959** | 0.921 | **0.778** (collapses) |
| SigLIP2-Linear | 0.858 | 0.828 | 0.671 |

Real-world transmission (Table 8, RRDataset): MetaCLIP2 leads on recapture
(0.719 vs DINOv3 0.647 vs PE-CLIP 0.548); DINOv3 and MetaCLIP2 tie on social
transfer (~0.712).

**The paper independently confirms our own measurement.** It reports PE-CLIP
with the highest clean baseline of any backbone *and* the worst blur
collapse (0.959 -> 0.778). Our clean-trained head showed exactly that shape:
clean 0.9987, `blur_sigma2.0` 0.8668 with balanced accuracy at chance. Our
augmented training then lifted `blur_sigma2.0` to 0.9646 -- i.e. **the
augmentation compensated for a documented, backbone-specific weakness.**

### Three caveats before racing on these numbers

1. **We cannot use the paper's winner.** Its DINOv3 is **ViT-7B/16** (1664-dim).
   Seven billion parameters violates the competition's hard <2B rule. Our
   `dinov3-l` is ViT-Large/16 (1024-dim) -- a different model that happens to
   share a family name.
2. **Only PE-Core-L matches exactly.** Paper MetaCLIP2 is *Worldwide Giant*
   (1664-dim); our `metaclip2-h` is *Worldwide Huge* (1280-dim). Paper DINOv2
   runs at 224px; ours at 518px. So for three of four registry entries the
   paper's ranking is evidence about a **family**, not about the checkpoint we
   would actually ship.
3. **The paper reports no Gaussian-noise robustness at all** -- only JPEG and
   blur. Our binding weakness is noise (`noise_sigma0.1` 0.9179, and the
   ceiling probe showed it does not respond to augmentation). So on the one
   axis we most need to fix, the paper offers no guidance and the race is a
   genuine experiment rather than a confirmation.

### How much room is actually left?

Decomposing the remaining distance to a perfect 1.0, on the current shipping
head (Run 6):

```
val       score 0.9892   headroom 0.0108
  of which AUC_clean  0.9981 -> 0.5 * 0.0019 = 0.0010   ( 9% of headroom)
           AUC_robust 0.9803 -> 0.5 * 0.0197 = 0.0099   (91% of headroom)

demo-val  score 0.9984   headroom 0.0016   -- effectively saturated
```

**91% of what is left sits in AUC_robust, and it is concentrated in six
cells:** `noise_sigma0.1` 0.9179, `chain_heavy` 0.9182, `resize_0.25x` 0.9546,
`chain_medium` 0.9591, `noise_sigma0.05` 0.9597, `blur_sigma2.0` 0.9646.

A backbone that fixed the three noise views outright (3 of 17 pooled views)
would move pooled to roughly 0.986-0.988 and the score to ~0.992-0.993:
**a gain of +0.003 to +0.004.** A uniformly better backbone might reach
+0.006. On demo-val, ~+0.001.

**So the race is competing for about half a point on our own benchmarks, and
they can no longer discriminate.** The paper's in-the-wild spread between
these same backbones is *ten points* (0.940 / 0.899 / 0.842); ours is half a
point, because our val is in-distribution and both our sets are saturated.

That is an argument for racing *carefully*, not for skipping it: judge the
race on the **weak cells and the worst-case score**, which still have real
range (0.9179 to 1.0), and treat the pooled headline as uninformative. And
weight the paper's in-the-wild evidence over our own val when the two
disagree -- our val cannot see the difference it is being asked about.

### Consequence for the race

Race **`metaclip2-h` and `dinov3-l`** and skip `dinov2-g`: the paper puts
DINOv2 second-from-bottom on both benchmarks (0.852 / 0.636), and at 518px it
is by far the most expensive to embed (~2.4x PE-Core's pixels, 3.6x its
parameters -- an estimated 3 hours against ~30 minutes each for the other
two). Spending three hours on the candidate the paper ranks near-last is the
worst available trade.

Worth checking separately: whether MetaCLIP2 *Worldwide Giant* (the variant
the paper actually benchmarked) fits under 2B params. If it does, it is a
better race entrant than our current `huge`.

---

## 2h. Unseen REAL domains are called AI (2026-08-30)
## >>> PARTIALLY RETRACTED 2026-08-30 -- read 2h-CORRECTION first <<<

> **The headline of this section was a label bug in our own pipeline, not a model
> failure.** The "100% FPR on human portraits" rows below are 250 StyleGAN faces
> that we relabelled as real. The live r/itookapicture observation that prompted
> the investigation was genuine and is reproduced properly in 2h-CORRECTION; the
> portrait mechanism, the "threshold cannot fix it" analysis, and every action
> item derived from them are withdrawn. Section kept intact below because the
> reasoning trail is why the bug took so long to find.

Found from a live Chrome extension running the shipping head over r/itookapicture,
where many real photographs scored > 0.50. That is not a threshold problem and it
is not noise. Broken down by real-image SOURCE on ood-s4000, clean view:

```
real source          n     FPR@0.5   FPR@0.99   mean P(AIGC)
Real (ImageNet)   1890       0.037      0.001         0.065
WhichFaceIsReal    110       1.000      1.000         1.000   <== every single one
val (ImageNet)    1000       0.013      0.000         0.026
demo-val (COCO)   1000       0.006      0.000         0.016
```

**Every one of the 110 human-portrait photographs is classified AI-generated with
probability 1.000.** Not borderline -- saturated. That single subpopulation is
5.5% of ood's reals and accounts for essentially the whole 8.9% aggregate FPR;
the ImageNet-like reals sit at a healthy 3.7%.

### Why threshold tuning cannot fix it

Pooled over clean + CDN-like degradations (jpeg q70, resize 0.5x, chain_light):

```
threshold  0.50   0.70   0.90   0.95   0.99
FPR       0.153  0.114  0.077  0.065  0.058
TPR       0.955  0.933  0.885  0.849  0.746
```

FPR floors out near 6% no matter how high the threshold goes, because the
failures are at probability 1.0. Reaching 5% FPR needs threshold 0.9998, which
drops TPR to 0.362.

### Mechanism

The training pool's REAL half is ImageNet photos, and nothing else. Any real
image from a domain absent there is mapped confidently into AI territory. Human
portraits are the demonstrated case; the r/itookapicture report suggests
artistic/enthusiast photography is another.

Two secondary, weaker effects were also measured among true reals (n=700,
ood clean): P(AIGC) rises as images get **smoother** (edge-energy rho=-0.215,
p=9e-9) and **more saturated** (rho=+0.130, p=6e-4). Both describe processed
photography -- shallow depth of field, denoised raws, colour grading -- and
Reddit's CDN resize/re-encode pushes further the same way. Resolution
(rho=-0.049) and background texture (rho=-0.016) were NOT significant, so the
SID_Set-style composition shortcut is not what is happening here.

Effect sizes for those two are modest. The dominant term is plain domain
coverage of the REAL class, not any single image statistic.

### What this invalidates about our own reporting

**Aggregate AUC completely masked a 100% failure on a real subpopulation.**
ood AUC is 0.9532 and the per-generator table looked healthy, because that table
breaks down the AIGC half by generator and treats reals as one undifferentiated
pool. Any future eval must report **FPR per real source**, not just per
generator.

### Actions

1. **Expand the REAL corpus by domain, not by count.** Portraits/faces (from a
   source that is NOT WhichFaceIsReal, which must stay held out as the detector
   for this bug), artistic/enthusiast photography, phone snapshots, screenshots.
   This outranks every remaining modelling change.
2. **Do not ship a binary verdict.** Emit the score with an explicit uncertain
   band; a confident-wrong label is worse than an abstention.
3. `data/train_ext/` helps only slightly here -- it adds AIGC-bench "Real",
   which is still ImageNet-like. It does not close the portrait gap.

---

## 2h-CORRECTION. The portrait failure was our label bug (2026-08-30)

`WhichFaceIsReal` in `data/ood/` is **not** a real-image source. whichfaceisreal.com
shows an FFHQ photograph beside a StyleGAN fake; the HF port ships only the fakes
under a class name that reads as real. Three independent signals agree:

- upstream's own `label` column is `1` (`label names: ['real','fake']`) for every
  row sampled -- 117/117;
- the pixels show GAN artefacts on inspection (incoherent backgrounds, melted
  hair, blob artefacts where jewellery should be);
- the model scored all 250 above 0.997, which is what a working detector does.

Only the upstream dataset card says "real", and it loses 2-to-1.

**How the wrong label got in.** `config.GENERATOR_FAMILY` mapped
`WhichFaceIsReal -> "real"`. The OOD index-rebuild step (`reindex_from_disk`,
used to recover after an interrupted download) derived each label from its
directory name via that map, rewriting 250 rows from `label=1` to `label=0`.
Nothing downstream could catch it: the images sat in the right folder, and
`manifest_fingerprint` hashes `image_path` only, so no cache went stale.

**What it cost.** 250 correctly-detected fakes were counted as false positives,
which manufactured the "100% portrait FPR", which motivated a portrait-coverage
theory, a search for real-face corpora, and a face-crop mining plan. All of that
is withdrawn. The "portrait direction" used in that analysis was computed as
`mean(WhichFaceIsReal) - mean(ood Real)` -- i.e. a **StyleGAN-face direction**.
That is why real portraits never reached it (best available: 4.87 against a
"region" starting at 10.37) and why tighter cropping moved nothing (4.84 -> 5.09
from full frame to centre-35%): the measurement was distance to StyleGAN faces.

### Corrected OOD numbers

Measured with `eval-grid` after re-embedding `ood-s4000` (the stratified sample
keys on label, so correcting 250 labels changes which rows it selects; the fresh
sample is 2,000/2,000 and contains 124 WhichFaceIsReal rows, all `label=1`):

| head | | mean AUC (18 views) | clean AUC | clean FPR | clean TPR |
|---|---|---|---|---|---|
| `augchain` | as reported | 0.9153 | 0.9532 | — | — |
| `augchain` | corrected | 0.9597 | 0.9940 | 0.0205 | 0.9490 |
| `photoreal` | as reported | 0.9265 | 0.9670 | — | — |
| **`photoreal`** | **corrected** | **0.9620** | **0.9961** | **0.0220** | **0.9660** |

Worst single view for the shipping head is `noise_sigma0.1` at AUC 0.8616.

### Fix

`WhichFaceIsReal -> "gan"` in `GENERATOR_FAMILY`; the 250 index rows restored to
`label=1`; `ood.csv` rebuilt to `{real: 4000, aigc: 4200}`; the 54 cached
`*__ood-s4000__*.npz` label arrays patched in place (safe -- the fingerprint
covers `image_path` only). `reindex_from_disk` no longer derives labels at all:
it recovers them from the existing index and **hard-fails** on any image it
cannot account for, because a generator class can be multi-label upstream.

### What survives from 2h

- **The live observation was real.** Enthusiast photography does false-positive.
  Measured properly on WildRF (real Reddit/X/Facebook photographs, a tier with
  no label ambiguity), the then-shipping head flagged **33%** of real social-media
  photographs at threshold 0.5.
- **Expanding the REAL corpus by domain works.** Adding SID_Set reals and 4,000
  Unsplash photographs took WildRF FPR@0.5 from **.330 to .183** at unchanged TPR
  (.995 -> .993), and improved OOD rather than costing it.
- **Report FPR per real source.** Still right, with a caveat this episode earned:
  a per-source breakdown makes a mislabelled source look exactly like a model
  failure. Inspect the images of any source that reports an extreme number
  BEFORE theorising about it. Four images would have saved days.

### Withdrawn

The smoothness/saturation correlations among "true reals" (rho=-0.215 / +0.130)
were computed over an ood real pool that included the 250 mislabelled StyleGAN
faces. They have **not** been recomputed and should not be cited until they are.

---

## 2i. Head depth ablation: the linear probe wins where it matters (2026-08-30)

Free to test -- the embeddings already exist, so each arm is ~90s of CPU. Same
data, same 11 training views, same `--balance`, same seed; only the head differs.

| head | val clean | val robust | OOD mean AUC | **WildRF clean AUC** | **WildRF FPR@TPR.98** |
|---|---|---|---|---|---|
| **linear** | 0.9987 | 0.9754 | 0.9620 | **0.9935** | **0.0512** |
| mlp (1x512) | 0.9996 | 0.9820 | 0.9637 | 0.9867 | 0.1087 |
| mlp2 (1024,512) | 0.9997 | 0.9817 | 0.9597 | 0.9861 | 0.0759 |

**The ranking inverts between the tier you fit and the tier you ship to.** Both
MLPs beat linear on val -- by enough to look like a real upgrade -- and both are
decisively worse on WildRF, where the MLP more than DOUBLES false positives at
matched recall (.1087 vs .0512 at TPR 0.98). On the unseen-generator tier it is
a wash: mlp edges mean AUC (0.9637 vs 0.9620) while losing at the high-recall
operating point anyone would actually deploy (FPR@TPR.99 .1225 vs .0875).

This is UniversalFakeDetect (Ojha et al., CVPR 2023) reproduced on our own data:
on frozen features, the linear probe transfers to unseen generators better than
a trained deep classifier, and the deep classifier's in-distribution advantage
inverts off-distribution. "Simplicity Prevails" (arXiv:2602.01738) reaches the
same conclusion and uses a single linear layer.

Mechanism: the backbone already did the representation learning. Head capacity
past a linear boundary is spent fitting the seven TRAINING generators, and the
deployment requirement is generalization to generators nobody has seen.

**Do not revisit this by looking at val.** Any future head change must be judged
on WildRF and the unseen-generator tier, at matched TPR. `mlp2` is kept in
`heads.py` so the negative result stays reproducible.

---

## 2j. train_ext: generator diversity helped the family we had already solved (2026-08-30)

`train_ext` = `train` + 5,178 extra REALS + 1,941 new AIGC rows across 6
generators absent from our pool. Trained with the identical recipe and the same
`--extra-train-manifest sid-real unsplash-real`, so the base pool is the only
variable.

**Predicted before running it, from the composition:** of those 1,941 new AIGC
rows, **1,617 are GAN** and only 324 are diffusion (SD14, already at 0.967
*unseen*). So its generator diversity points at the family already solved, and
its likely value is the extra reals. That is exactly what happened.

| tier | metric | photoreal | trainext |
|---|---|---|---|
| WildRF | clean AUC | 0.9935 | **0.9945** |
| WildRF | **FPR @ TPR .98** | .0512 | **.0344** |
| ood | clean AUC | 0.9961 | **0.9978** |
| ood | **FPR @ TPR .98** | .0420 | **.0240** |
| demo_val | clean AUC | 0.9990 | **0.9995** |

Better at every matched operating point, and on worst-view too (WildRF
`noise_sigma0.1` 0.8373 -> 0.8495). But the family split shows where it came
from:

| head | diffusion TPR@0.95 | GAN TPR@0.95 |
|---|---|---|
| photoreal | 0.859 | 0.966 |
| trainext | **0.859** | **0.992** |

**Diffusion did not move.** DALLE2 .550 -> .542, Midjourney .752 -> .736,
ADM .626 -> .659 — all noise. +2.6 points of GAN recall on a family already at
0.966, plus an FPR improvement attributable to the 5,178 reals.

**Third time real-side data beat generator-side data in this project.** The
pattern is consistent enough to plan around: adding REAL domains moves the
numbers, adding generators mostly does not — unless the generators are ones we
are actually failing on, which these were not.

**Caveat on the ood gain.** `train_ext` was drawn from AIGC-bench positions
8,400+ while `ood` occupies 1-8,400. The images are disjoint by construction but
the SOURCE RENDITION is shared, so part of that +0.0054 mean AUC is
source-matching rather than generalization. WildRF is a completely different
source and improved independently, which is why the arm still ships. `val` stayed
flat (0.9987 -> 0.9986), ruling out the reals-overlap inflation HANDOFF warned
about.

### Threshold is a property of the head, re-derive it on every swap

Swept in 0.005 steps on WildRF pooled over clean + CDN-like views, split by
image, F1-optimal on one half, reported on the other:

| head | chosen | HELD-OUT FPR | HELD-OUT TPR |
|---|---|---|---|
| photoreal | 0.920 | .0408 | .9721 |
| **trainext** | **0.940** | **.0283** | **.9686** |

At matched FPR (~2.8%) the new head buys a full point of recall over the old one
(.9686 vs .9584). Note the finer grid also moved photoreal's own optimum from the
0.95 an earlier coarse sweep reported to 0.920 — the coarse grid was not wrong
about the direction, just about the resolution.

---

## 2k. Modern generators: one mislabelled corpus masked a real gain (2026-08-30)

Pulled three 2024-25 generators from three publishers (1,500 each) to attack the
diffusion gap 2j left open, DALLE3 held out as the eval tier. Trained with the
identical recipe, `--extra-train-manifest sid-real unsplash-real` plus the new
corpora, so the added data is the only variable.

**First run was a clear regression**, worst exactly where the data was supposed
to help:

| ood metric | control (trainext) | +3 modern corpora |
|---|---|---|
| clean AUC | 0.9978 | 0.9950 |
| 18-view mean | 0.9674 | 0.9566 |
| DALLE2 clean | 0.9789 | **0.9428** |
| DALLE2 degraded | 0.7927 | **0.7205** |
| real FPR@t | 0.011 | 0.019 |

Re-ran at 1 epoch in case the saved-final-epoch rule (`train/probe.py`, no
best-epoch selection) had caught an overshoot. It had not: 0.9949 / 0.9579. The
regression was real.

### Cause: `gmongaras/Stable_Diffusion_3_Recaption` is not SD3 output

It is a RECAPTIONING corpus -- real photographs with SD3-authored captions. We
labelled 1,500 real photos `label=1`. Diagnosis, in the order that forced it:

1. **The architecture was never the constraint.** The control head, which never
   saw any of this data, scores midjourney_v6 at mean P=0.9737 and nano_banana
   at 0.8738. A capacity ceiling on modern diffusion would have sunk those too.
2. **SD3 scored 0.0230** -- indistinguishable from genuine photographs (0.0178),
   not the ~0.5 an over-capacity head produces on the unresolvable. Confidently
   real, not uncertain.
3. **Because they are photographs.** 281 distinct resolutions in 400 images vs 1
   for both genuine dumps; 7.5% square vs 100%; modal sizes 500x500, 640x480,
   800x600; 78% of the source rejected under 384px, impossible for a 1024x1024
   renderer. `sd3_000011.jpg` is a scraped product photo with a burned-in
   "I wanne Buy" watermark.

`SD3 alone contributed roughly half the training loss` (0.2599 -> 0.1275 on
removal) -- the signature of unfittable label noise. Quarantined to
`data/quarantine/` with the evidence; removed from `SOURCES`, `config.py` and
the `main.py` manifest registry so it cannot be re-pulled.

**The `-recap` / `_Recaption` suffix is not a provenance signal in either
direction.** `Photoroom/midjourney-v6-recap` carries it and IS genuine output.
Check the pixels: a generator dump has one or two resolutions, a scraped corpus
has hundreds. That check costs ten seconds and would have caught this
pre-download.

### With SD3 removed, generator-side data finally moved the needle

`nano-banana` + `midjourney-v6` only, same recipe. Threshold re-derived by 2j's
protocol (0.005 sweep on WildRF over clean + CDN-like views, split by image,
F1-optimal on one half, reported on the other); the protocol reproduces
trainext's recorded 0.940 / .0283 / .9686 as 0.940 / .0272 / .9688.

| tier | metric | trainext | **aigcmodern_nosd3** |
|---|---|---|---|
| WildRF | clean AUC | 0.9945 | **0.9960** |
| WildRF | 18-view mean | 0.9764 | **0.9828** |
| WildRF | **FPR @ TPR .98** | .0424 | **.0336** |
| ood | clean AUC | 0.9978 | **0.9980** |
| ood | 18-view mean | **0.9674** | 0.9664 |
| ood | **FPR @ TPR .98** | .0265 | **.0165** |
| ood | DALLE2 degraded | 0.7927 | **0.8203** |
| demo_val | clean AUC | 0.9995 | **0.9997** |
| val | clean AUC | 0.9986 | **0.9989** |
| — | threshold | 0.940 | **0.990** |

**FPR at matched recall falls 21% relative on WildRF and 38% on ood.** AUC is
flat-to-better everywhere; the ood 18-view mean is -0.0010, noise. Do not read
the raw own-threshold FPR (.0272 -> .0315) as a regression -- the new head sits
at higher recall (.9688 -> .9773); the matched-operating-point rows are the
comparable ones.

**This breaks 2j's pattern.** Three times running, real-side data moved the
numbers and generator-side data did not. Here generator-side data did -- because
for the first time the generators were ones we were actually failing on. The
2j rule survives with its own stated caveat intact.

### Real-image FPR: flat in aggregate, but the platform mix shifted

At each head's own operating point (trainext 0.940 / TPR .9688, nosd3 0.990 /
TPR .9773), FPR on REAL images only:

| real source | n | trainext clean | nosd3 clean | trainext 18-view | nosd3 18-view |
|---|---|---|---|---|---|
| WildRF reddit | 750 | .0267 | **.0213** | .0245 | **.0167** |
| WildRF twitter | 341 | **.0352** | .0499 | **.0375** | .0415 |
| WildRF facebook | 160 | **.0437** | .0563 | **.0368** | .0441 |
| **WildRF all** | 1251 | .0312 | .0336 | .0296 | **.0270** |
| ood reals | 2000 | .0010 | **.0005** | .0064 | **.0013** |
| demo_val reals | 1000 | .0010 | **.0000** | .0018 | **.0006** |
| val reals | 1000 | .0020 | **.0000** | .0129 | **.0027** |

Aggregate WildRF FPR is flat (.0312 -> .0336 clean) but bought a full point of
recall, which is why the matched-recall row above is the honest comparison. On
every curated tier the new head is at or near zero.

**Two things worth watching.** (1) The gain is not uniform across platforms:
Reddit improves clearly, Twitter and Facebook regress. Facebook is n=160, so
treat it as directional, but Twitter at n=341 is harder to dismiss. (2) The
new head's worst FPR view on WildRF is `jpeg_q70` at every platform, displacing
`noise_sigma0.1` — a TRAINED view, and the one closest to real CDN
recompression, which is the actual deployment condition. Both deserve a look
before this ships as the default.

### VERDICT: DALLE3 holdout confirms real generalization

The whole point of holding DALLE3 out. Never trained on, fourth independent
publisher, vetted genuine before use (13 distinct resolutions across 1,500
images, every ratio on DALL-E 3's native 1:1 / 7:4 modes -- contrast SD3's 281).
Single-class, so AUC is computed by pairing its fakes against WildRF reals.

| DALLE3 metric | trainext (0.940) | nosd3 e2 (0.995) | **nosd3_e1 (0.985)** |
|---|---|---|---|
| clean TPR@t | 0.9693 | 0.9873 | **0.9900** |
| 18-view mean TPR@t | 0.8900 | 0.9357 | **0.9457** |
| worst view TPR@t (`noise_sigma0.1`) | 0.5140 | 0.6567 | **0.6973** |
| 18-view TPR@0.5 | 0.9658 | 0.9890 | **0.9905** |
| clean AUC vs WildRF reals | 0.9949 | 0.9977 | **0.9984** |
| 18-view AUC vs WildRF reals | 0.9785 | 0.9889 | **0.9910** |

**All 18 views improve, on every metric, at a HIGHER threshold** -- strict
dominance, not a recall trade. Training on Midjourney-v6 + nano-banana improved
detection of a generator from NEITHER source, which is the generalization claim
2j's caveat said we could not make for train_ext. The worst-case view gains
+18.3 points of recall (0.514 -> 0.697), which is the number that matters for a
product: it is the failure mode, not the average.

### Ship 1 epoch, not 2 -- and that is a general finding here

`train_head_views` saves the FINAL epoch with no best-epoch selection. Epoch 2
overfits robustness in every arm measured: val AUC_robust 0.9740 -> 0.9670 on
the nosd3 arm, 0.9657 -> 0.9581 on the SD3-poisoned one. Threshold re-derived
with predict.py's exact protocol (WildRF, clean + jpeg_q70/q90/resize_0.5x/
chain_light, split by image, 0.005 sweep, F1 on half A reported on half B;
reproduces trainext's recorded 0.940/.0283/.9686 as 0.940/.0280/.9663):

| head | threshold | HELD-OUT FPR | HELD-OUT TPR |
|---|---|---|---|
| trainext | 0.940 | .0280 | .9663 |
| nosd3 (2 epoch) | 0.995 | **.0224** | .9643 |
| **nosd3_e1** | **0.985** | .0246 | **.9773** |

nosd3_e1 beats trainext on BOTH axes at once -- 12% lower FPR and +1.1 points of
recall. The 2-epoch arm buys a little more FPR for 1.3 points less recall and
loses on DALLE3, demo_val and WildRF AUC, so it is not the pick.

**SHIPPED: `pe-core-l__linear__aigcmodern_nosd3_e1.pt` at threshold 0.985**,
now the default in `predict.py`, `demo/server.py` and `main.py predict`.

```
uv run main.py train-head-views --backbone pe-core-l --with-chains   --val-sample-rows 2000 --train-manifest train-ext   --extra-train-manifest sid-real unsplash-real nano-banana midjourney-v6   --balance --epochs 1 --out models/pe-core-l__linear__aigcmodern_nosd3_e1.pt
```

### The audit could not have caught this

`audit_data.load_source_frames` globbed `data/raw/` only, so every corpus added
after it -- all of `aigc_ext/` and `real_ext/` -- was never probed. The audit
reported nothing wrong by not looking. Fixed: `AUDIT_DIRS` now spans raw,
aigc_ext and real_ext, and a new corpus dir must be added there. Post-fix pooled
blind probe 0.5829 PASS; report in `reports/audit_data_2026-08-30.txt`. Note
single-class corpora skip the per-source probe (needs both labels) and are
covered only by the pooled one.

Third fault of this family, after the SID_Set aspect-ratio shortcut and the
depth-map fault: **a label correlating with something other than the generator.**

---

## 3. What was built

### Wave 1
- `scripts/audit_data.py` (new) — shortcut audit: per (source, label) format /
  resolution / aspect distributions, plus a **blind probe** (logistic regression
  on 16x16 greyscale). `--transform` runs it on tensors from the real eval
  pipeline. **Clearing ~70% means a shortcut survives.** Keep as a permanent
  regression test.
- `src/aigc_detect/data/transforms.py` — added `build_backbone_transform()`,
  repointed all three builders. Parameter table untouched.
- `main.py` — `audit-data` subcommand.

### Wave 2
- `src/aigc_detect/registry/backbones.py` — frozen-backbone registry. Asserts vision-tower
  params < 2e9 and prints the count.
- `src/aigc_detect/embed/embeddings.py` — `precompute_embeddings(...)`, caches to
  `data/embeddings/<backbone>__<manifest>.npz` (arrays: `embeddings`, `labels`,
  `sources` + metadata). Supports `--limit` and `--force`.
- `src/aigc_detect/registry/heads.py` — `LinearHead`, `MLPHead`, `build_head(kind, in_dim)`.
- `src/aigc_detect/train/probe.py` — paper recipe, standardises on **train**
  statistics only, reports val AUC **broken down by source**, saves to
  `models/<backbone>__<head>.pt`.
- `src/aigc_detect/config/paths.py` — added `EMBEDDINGS_DIR`.
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

      Full detail and per-view tables in experiments.md Run 5.

- [x] **Chained views in TRAINING** — DONE 2026-08-29 (Run 6).
      `TRAIN_CHAIN_SPECS` adds four training-only chains, disjoint from the
      three scored ones and built only from severities already in
      `TRAIN_VIEWS_DEFAULT`, so composition is the single new variable.
      `--with-chains` on `train-head-views`, `--train-chains` on
      `embed-views`.

      **Composition training transfers to unseen compositions.** All three
      scored chains were held out and all three improved: `chain_heavy`
      0.8627 -> **0.9182**, and its composition penalty (chain AUC minus its
      own weakest component's AUC) more than halved, -0.0860 -> -0.0364.
      Pooled score moves little (0.9878 -> 0.9892) but **worst-case moves a
      lot: 0.9306 -> 0.9580**, and the binding view is no longer a chain.

      Next weakest axis is now `noise_sigma0.1` at 0.9179 — a held-out
      severity. Attack heavy sensor noise, not composition.

- [ ] **A chain may not repeat a transform family** — enforced at import by
      `transforms._validate_chain_specs()`, and worth understanding before
      editing either spec table. A repeated family compounds *past the 5.2
      severity envelope the grid claims to test*: blur 2.0 twice is
      sigma_eff = 2.83 against a table maximum of 2.0; resize 0.5x twice is
      0.25x reported under another name; jitter twice is +-44%. The chain
      still runs and still produces a plausible number, so nothing flags that
      the reported robustness envelope quietly widened. Repeated *re-encoding*
      is genuinely realistic and worth studying — but as its own named
      severity axis, not smuggled in through a chain.
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

   Fixed in `embed/embeddings.py`: `manifest_fingerprint()` hashes the manifest's
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
   (ImageNet), while `embed/embeddings.py` normalizes with `module.norm_mean` /
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
    `embed/views.py` seeds every stochastic view to make it byte-reproducible
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
