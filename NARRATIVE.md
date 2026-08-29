# Training narrative — AIGC detection (TikTok TechJam 2026, Track 5)

Every model run, in order, with what it changed and what it taught. This is
the "how we got here" document; `FINDINGS.md` holds the forensics and the
traps, `AGENTS.md` holds the current-state map.

## Comparability epochs — READ BEFORE COMPARING ANY TWO NUMBERS

Numbers in this file are only comparable **within** an epoch. An epoch ends
when something changes what a cached embedding *means*. Cross-epoch
comparison produces differences that look like results and are artifacts.

| Epoch | Boundary | Caches | Status |
|---|---|---|---|
| **E0** | anything before the 2026-08-29 trap fixes | — | **INVALID, discard.** Noise views destroyed 84.5% of every image (trap 8), wrong normalization constants (trap 9), nondeterministic views (trap 10). No number survives. |
| **E1** | commit `1c6fa35` | `pe-core-l__val__*.npz` (15 views, full 4,200-row val) | **Superseded.** Index-based seeding (trap 12) and one global `transform_fingerprint` (trap 13). No published number came from it. Automatically reported STALE by the per-view fingerprint, so it cannot be silently mixed in. |
| **E2** | commit `abc28a7` onward — **current** | `*__val-s2000__*`, `*__demo_val-s2000__*`, `*__train-s6000__*` | **Valid and mutually comparable.** Path-based seeding, per-view fingerprints, 18 scored views + 4 training chains. **Runs 4, 5, 6 and the ceiling probe all live here.** |

Within E2 there is one further caveat: **Run 4's head trained on 23,800 clean
rows; Runs 5, 6 and the ceiling probe used 6,000.** Their *evaluations* are
directly comparable (identical eval caches), but Run 4 is **not** a controlled
control for the augmentation ablation. That role belongs to Run 5's
clean-only arm, which uses the same 6,000 images. This is why the ablation
tables in section 2f compare cleanonly / aug / augchain and exclude Run 4.

**What would open E3** (and force a full recompute + a new comparability
block here): changing any severity in the 5.2 table, changing the noise
domain (see the open question in FINDINGS 2f), re-running `main.py split`,
or changing `--sample-rows` / `--sample-seed`. Adding a *new* named view does
not — per-view fingerprints mean only the changed view invalidates. **Prefer
adding a new view name over redefining an existing one**, precisely so the
history in this file stays readable.

---

**Ground rules for anything added here:**

- One row per *run*, never per hope. A number goes in only after it was
  produced by a command that can be re-run.
- Record the runs that were **wrong**, and why. Three of the entries below
  exist to stop someone re-deriving a conclusion we already disproved.
- Always cite the artifact (`reports/*.csv`, `models/*.pt`) so a claim can be
  checked rather than trusted.

---

## The architectural bet, stated once

The backbone is **frozen and never fine-tuned**. This is the mechanism, not a
compute shortcut:

- Ojha 2023: a linear probe on frozen CLIP beats CNNs trained from scratch by
  **+15.07 mAP on unseen generators**; full fine-tuning destroys the general
  representation and overfits to the training generator.
- Cozzolino: **+13%** from a frozen backbone specifically on impaired and
  laundered data.
- The organizers' day-1 advice ("fine-tune a pretrained backbone") describes
  CNNSpot, the canonical fragile baseline — F1 **82.66 -> 33.98** under mild
  JPEG.

Only a ~1,025-parameter head trains. Two consequences run through this whole
document:

1. Embeddings can be **precomputed and cached**, so every experiment below
   costs seconds of head training rather than hours of GPU time.
2. Training the head on *degraded* embeddings does **not** invalidate the
   cache — it is the reason the cache exists. Only a change of backbone
   weights, resolution, or preprocessing does.

---

## Runs

### Run 1 — CIFAKE + SID_Set, clean-only. **INVALID (shortcut).**

`pe-core-l` frozen + linear head, 5,000 of 108,800 train rows.

```
val_auc 0.9936   val_balanced_acc 0.9608
by source: cifake AUC 0.9937 (n=18000)   sid_set AUC 0.9906 (n=1200)
```

Looks like a win. Is not one. Cross-source transfer settled it:

```
WITHIN  sid_set -> sid_set   AUC 1.0000   bacc 1.0000
CROSS   sid_set -> cifake    AUC 0.7435   bacc 0.5047   <- chance
CROSS   cifake  -> sid_set   AUC 0.9555   bacc 0.8750
```

SID_Set scores a **perfect 1.0000 on itself** and transfers at chance.
Nothing here is perfectly solvable, so a perfect score means the model found
something other than the question. Two independent shortcuts, both measured:
aspect ratio (AIGC 100% square vs real 4.5%) and composition (0.93 balanced
accuracy from an **8x8 greyscale thumbnail**).

**Lesson kept:** a high number on this task is evidence of a leak. Cross-source
transfer before believing anything. (FINDINGS 1, 2.)

### Run 2 — Tiny-GenImage only, clean-only. The baseline everything else is measured against.

SID_Set dropped (composition shortcut, unfixable by preprocessing). CIFAKE
dropped — measured as **actively harmful**, not merely useless. Train pool
became Tiny-GenImage alone: 23,800 train / 4,200 val, content-matched, 7
generators, generator-tagged.

`models/pe-core-l__linear.pt`:

```
val AUC        0.9996  (saturated)
demo-val AUC   0.9949   FPR 0.019   TPR 0.948
```

**Lesson kept:** it was data *quality*, not data volume — the pool shrank 4.5x
and every honest metric improved. (FINDINGS 2c, 2d.)

### Runs 3a-3c — three robustness grids that were all invalid. **DISCARDED.**

Before a single valid robustness number existed, the grid had to be repaired
six times. Each defect was silent — nothing threw, every output looked
plausible:

| # | Defect | What it would have "proved" |
|---|---|---|
| 8 | Noise applied *after* `Normalize`, where its `[0,1]` clamp destroys 84.5% of every image | "PE-Core has a severe sensor-noise weakness" |
| 9 | ImageNet normalization stats instead of the backbone's own | every view scored under normalization the model never saw |
| 10 | `color_jitter` and noise views nondeterministic | two backbones raced against each other never faced the same test |
| 12 | Stochastic views seeded by **row index** | a subsample's numbers differ from the full run's, spuriously |
| 13 | One shared staleness key over the whole 5.2 table | any severity edit invalidates all views, so nobody re-runs |
| 15 | `thresholds[argmax(bacc)]` on near-separable data | a robustness cliff manufactured by the threshold rule (worth up to **17 points** of BAcc) |

**No grid number produced before 2026-08-29 is valid.** Full detail in
FINDINGS traps 8-15.

### Run 4 — first valid grid, clean-trained head. **CURRENT BASELINE.**

`models/pe-core-l__linear.pt` (Run 2's head, trained on clean embeddings only)
scored across all 18 views. Seeded, label-balanced 2,000-row subsamples.

`reports/grid__pe-core-l__val-s2000__pe-core-l__linear.csv`
`reports/grid__pe-core-l__demo_val-s2000__pe-core-l__linear.csv`

| | val-s2000 | demo_val-s2000 |
|---|---|---|
| AUC_clean | 0.9997 | 0.9935 |
| AUC_robust (pooled) | **0.8779** | **0.9032** |
| AUC_robust (mean) | 0.9526 | 0.9545 |
| AUC_robust (worst) | 0.8454 `chain_heavy` | 0.7859 `blur_sigma1.0` |
| **score, 0.5 clean + 0.5 pooled** | **0.9388** | **0.9484** |
| robustness gap (AUC) | 0.0471 | 0.0389 |
| robustness gap (BAcc@t) | 0.2093 | 0.1479 |
| mean AUC, single views | 0.9592 | 0.9474 |
| mean AUC, chained views | 0.9216 (**-0.0376**) | 0.9879 (**+0.0405**) |

Three things this run establishes:

**(a) The definition of AUC_robust moves the headline score by 5.4 points**
(0.9226 worst-case to 0.9762 mean) on *identical predictions*. All three are
reported every run; pooled is primary. The pooled-vs-mean spread is itself the
diagnostic — each view is internally well-ranked, but score *scales* drift
between views, which only pooled penalizes.

**(b) The failure is calibration, and it points two opposite ways.**

| view | AUC | BAcc@t | TPR | FPR |
|---|---|---|---|---|
| clean | 0.9997 | 0.9915 | 0.990 | 0.007 |
| jpeg_q50 | 0.9970 | 0.8650 | **0.730** | 0.000 |
| blur_sigma1.0 | 0.9368 | **0.5230** | 1.000 | **0.954** |
| resize_0.5x | 0.9637 | **0.5310** | 1.000 | **0.938** |

Blur and resize drop balanced accuracy to **chance** while AUC stays
0.85-0.96 — ranking survives, the boundary does not. FPR 0.94-0.99 means the
model calls **essentially every blurred or downscaled real photo AIGC**.

The obvious explanation is a resolution asymmetry in the training data, and
one does exist — **balanced accuracy 0.7554 from image resolution alone**
(58.0% of train AIGC images are <=256px vs 6.9% of reals; AUC 0.5814, lower
because the AIGC half is bimodal with p90 = 1024px). Since every pipeline
resizes to 336px, that asymmetry reaches the model as *sharpness*: a 256px
image upscaled to 336 is smooth, a 500px image downscaled to 336 is not.

But the head is **not** simply reading resolution off the image. Correlating
its clean-view P(AIGC) against source resolution *within* each true class:

```
true reals  rho = -0.053  p = 0.09   (not significant)
true AIGC   rho = -0.233  p = 9e-14
```

If the model were riding resolution, low-res reals would score high — and
they essentially do not. So the honest statement is narrower than "it learned
smooth means generated": blur and resize move real embeddings **along the
head's AIGC direction**, and the training asymmetry is the plausible reason
that direction has a sharpness component, but the failure is not reducible to
a resolution readout. Distinguishing the two properly is what Run 5 tests.

JPEG fails in the **opposite** direction — TPR 0.73 with FPR 0.000, i.e.

re-compressed AIGC gets called real — while its AUC barely moves.

No single threshold fixes both. This is the measured case for per-degradation
calibration, and it is the specific weakness the next run tries to fix.

**(c) Chains behave oppositely on the two sets, and the difference is the
finding.** On val, `chain_heavy` is the **worst view in the grid** (0.8454,
below every single transform) — composition costs more than any one axis,
which is why chained views exist. On demo-val the chained mean is *higher*
than the single-transform mean (+0.0405), and every JPEG view outscores clean
(0.9999 vs 0.9935).

That is not robustness. The resolution relationship between the classes is
**inverted** between the two sets:

| | real median max-dim | AIGC median max-dim |
|---|---|---|
| train (Tiny-GenImage) | 500 px | **256 px** |
| demo-val (COCO + WildFake) | 640 px | **1024 px** (7.5% PNG) |

In training, AIGC images are the *smaller* ones; in demo-val they are the
*larger*. Mild degradation moves demo-val inputs toward the training domain,
so it helps — every JPEG view beats clean there. The clean view is the
outlier, not the chains.

That the head still scores 0.9935 clean AUC on demo-val **despite** the
inversion is the strongest evidence so far that it is reading generation
artifacts rather than resolution. A resolution-riding model would have
collapsed on demo-val, not scored 0.99. `eval-grid` now prints this warning whenever the delta is
positive, because the naive reading ("our model is robust to chains!") is
exactly backwards.

### Run 5 — augmented-view head training. **Largest single win so far.**

Hypothesis, from FINDINGS trap 4: with a frozen backbone only ~1,025
parameters train, so there is almost nothing to overfit and augmentation's
usual justification does not apply. Its real value here is different —
showing the head **paired clean/degraded embeddings of the same image**
teaches it which directions in embedding space to ignore. That should attack
Run 4's finding (b) directly, since the failure is a boundary that moves under
degradation.

Design, so the comparison is honest:

- Baseline and augmented head train on the **same 6,000 images**, so the
  variable is augmentation and not sample size. (Run 2's head used 23,800
  clean rows and is therefore *not* a valid control.)
- Train on a **subset** of degradation types, evaluate on all 18 views. If
  augmentation only helps on the corruptions it saw, that is memorization,
  not robustness — and held-out views are the only way to tell.
- Both heads scored on the identical `val-s2000` and `demo_val-s2000` grids.
- The scaler is computed from the **clean** train view in both arms, so the
  single variable is which rows the head saw, not which statistics
  standardized them.

Trained: `clean, jpeg_q70, blur_sigma1.0, resize_0.5x, noise_sigma0.05,
color_jitter, center_crop_80` (7 views x 6,000 images = 42,000 rows).
Held out: the other 11 views, **including all three chains**.

#### Headline

| | clean-only 2ep | clean-only 14ep | **augmented** |
|---|---|---|---|
| **val** AUC_clean | 0.9987 | 0.9996 | 0.9985 |
| **val** AUC_robust (pooled) | 0.8958 | 0.8767 | **0.9770** |
| **val score** | 0.9472 | 0.9382 | **0.9878** |
| **val** robustness gap (BAcc@t) | 0.2122 | — | **0.0635** |
| **demo-val** AUC_clean | 0.9947 | — | **0.9997** |
| **demo-val** AUC_robust (pooled) | 0.9314 | — | **0.9960** |
| **demo-val score** | 0.9631 | — | **0.9978** |

`models/pe-core-l__linear__aug.pt`, `reports/grid__*__aug.csv`.

**+0.0406 on val, +0.0347 on demo-val**, at zero cost to clean AUC — and
demo-val clean actually *improved* (0.9947 -> 0.9997), because learning to
ignore degradation directions also absorbs part of that set's domain shift.

**It is not a training-budget artifact.** The augmented arm sees 7x more rows
per epoch, so the clean-only control was re-run at 14 epochs to match. It went
the *other way*: AUC_robust 0.8993 -> 0.8767 as clean rose 0.9969 -> 0.9996.
More clean-only training actively **trades robustness away for clean
accuracy**. The gain is augmentation, not compute.

The pooled-vs-mean relationship also inverts, which is the mechanism showing
through: clean-only has pooled 0.8958 **below** mean 0.9504 (heavy score-scale
drift between views); augmented has pooled 0.9770 **above** mean 0.9710. The
drift is largely gone — which is exactly what "teach the head which embedding
directions to ignore" predicts.

#### Does it generalize, or did it memorize? (val, BAcc@t in brackets)

| view | trained? | clean-only | augmented |
|---|---|---|---|
| `blur_sigma1.0` | **trained** | 0.9498 [0.5355] | 0.9922 [0.9625] |
| `blur_sigma2.0` | held out | 0.8668 [0.5115] | 0.9584 [0.8700] |
| `resize_0.5x` | **trained** | 0.9661 [0.5480] | 0.9907 [0.9420] |
| `resize_0.25x` | held out | 0.8984 [0.5215] | 0.9486 [0.8605] |
| `noise_sigma0.1` | held out | 0.8803 [0.5680] | 0.9119 [0.8240] |
| `jpeg_q30` | held out | 0.9886 [0.8815] | 0.9906 [0.9245] |
| `chain_light` | held out | 0.9775 [0.9085] | 0.9880 [0.9400] |
| `chain_medium` | held out | 0.9268 [0.8430] | 0.9435 [0.8765] |
| `chain_heavy` | held out | 0.8265 [0.7220] | 0.8627 [0.7755] |

**Every held-out view improved**, so this is not memorization of the specific
corruptions. Unseen *severities* transfer nearly as well as trained ones —
`blur_sigma2.0` gains +0.36 balanced accuracy having only ever seen σ1.0.

Run 4's calibration collapse is repaired at its root: FPR on
`blur_sigma1.0` falls **0.927 -> 0.052**, `resize_0.5x` **0.903 -> 0.091**,
`blur_sigma2.0` **0.977 -> 0.214**. The "calls every blurred real photo AIGC"
failure is gone.

#### The one thing it did not fix

Chains improve, but far less than single transforms, so the composition gap
**did not close**:

```
mean AUC, single views   0.9590 -> 0.9795   (+0.0205)
mean AUC, chained views  0.9102 -> 0.9314   (+0.0212)
single-vs-chain delta   -0.0488 -> -0.0481  (unchanged)
```

`chain_heavy` remains the **worst view in the grid** (0.8627) and is the
binding constraint on the worst-case score (0.9306). Robustness learned from
single-axis degradations transfers to unseen *severities* of those axes, and
only weakly to *compositions* of them.

That is a direct, measured argument for putting chained views into
**training**, not just evaluation — the next experiment, and the reason
`CHAIN_SPECS` is a table rather than three hardcoded pipelines.

### Run 6 — chained views in training. **Closes Run 5's gap.**

Run 5 left one thing unfixed: composition. Added `TRAIN_CHAIN_SPECS` — four
training-only chains, **disjoint from the three scored ones** and built
*exclusively* from severities already in `TRAIN_VIEWS_DEFAULT`, so the single
new variable is composition and the severity holdout is untouched.

```
trainchain_a  blur1.0 -> jpeg70                              (2 ops)
trainchain_b  noise0.05 -> jpeg70                            (2 ops)
trainchain_c  crop80 -> jitter -> blur1.0                    (3 ops, no JPEG at all)
trainchain_d  jitter -> resize0.5 -> noise0.05 -> jpeg70     (4 ops)
```

`trainchain_c` deliberately does not end in a re-encode, so the head cannot
key on "a chain is the thing that ends in JPEG". `trainchain_*` views are
excluded from `eval_view_names()`, so they can never enter AUC_robust.

| | Run 5 (aug) | **Run 6 (augchain)** |
|---|---|---|
| val AUC_clean | 0.9985 | 0.9981 |
| val AUC_robust (pooled) | 0.9770 | **0.9803** |
| **val score (pooled)** | 0.9878 | **0.9892** |
| val worst view | `chain_heavy` 0.8627 | `noise_sigma0.1` 0.9179 |
| **val score (worst-case)** | 0.9306 | **0.9580** |
| demo-val score (pooled) | 0.9978 | **0.9984** |

The pooled headline barely moves (+0.0014). **The chains move a lot** — and
all three were held out:

| held-out chain | Run 5 | Run 6 | |
|---|---|---|---|
| `chain_light` | 0.9880 | 0.9952 | +0.0072 |
| `chain_medium` | 0.9435 | 0.9591 | +0.0156 |
| `chain_heavy` | 0.8627 | **0.9182** | **+0.0555** |

Composition penalty (each chain vs its own weakest component) more than
halves at depth:

```
             Run 5      Run 6
chain_light  -0.0027   +0.0031
chain_medium -0.0473   -0.0329
chain_heavy  -0.0860   -0.0364
```

**Composition training transfers to unseen compositions**, the same way
severity training transferred to unseen severities in Run 5. `chain_heavy` is
no longer the worst cell in the grid, and the worst-case score — the
adversarial number, what someone who gets to pick the transform achieves —
improves **0.9306 -> 0.9580**.

#### CORRECTION (2026-08-29): these numbers predate seeding, and one claim was wrong

`train_head_on_views` was **unseeded** until commit `53426ac` — head init and
DataLoader shuffle varied run to run. Three repeat runs of this exact
configuration gave AUC_robust(pooled) **0.9803 / 0.9797 / 0.9796**, i.e.
roughly ±0.0005. Every Run 4/5/6 number above is one such draw.

Canonical **seeded** figures for this run (`--seed 42`, now reproducible
exactly — verified by two identical runs):

| | reported above (unseeded) | **canonical (seeded)** |
|---|---|---|
| val AUC_robust (pooled) | 0.9803 | **0.9791** |
| val score | 0.9892 | **0.9886** |
| val worst-case score | 0.9580 | **0.9551** |
| demo-val score | 0.9984 | **0.9983** |

The ±0.0005 does not touch any conclusion here — augmented-vs-clean training
moved the score by +0.04, eighty times larger. **But one claim was wrong and
is retracted:**

> ~~"The binding constraint moved to `noise_sigma0.1` (0.9179), a held-out
> severity. Heavy sensor noise is now the weakest axis, not composition."~~

That rested on `noise_sigma0.1` 0.9179 vs `chain_heavy` 0.9182 — **a gap of
0.0003, against a per-view standard error of ~0.0065.** The seeded rerun puts
`chain_heavy` (0.9121) back as the worst view. The two are **statistically
tied**; which one prints as "worst" flips on the training seed.

The correct statement: **`chain_heavy` and `noise_sigma0.1` are jointly the
binding constraint at ~0.91–0.92, and this benchmark cannot separate them.**
Deep composition and heavy sensor noise are both open; neither has been shown
to dominate.

This is a second, sharper reason for the OOD tier. A benchmark that cannot
rank its own two worst cells certainly cannot rank two backbones that differ
by less.

**Full per-view tables for both evaluation sets — all 18 scored views, AUC /
BAcc@t / BAcc@0.5 / TPR / FPR, with the trained-vs-held-out marking and the
per-view progression across all three heads — are in FINDINGS.md section 2f.**
Machine-readable per-view CSVs are in `reports/`.

Two things the full table shows that the summary above hides:

- **Held-out severities gain *more* than their trained siblings.**
  `blur_sigma1.0` (trained) +0.0432 vs `blur_sigma2.0` (held out) **+0.0978**;
  `resize_0.5x` (trained) +0.0260 vs `resize_0.25x` (held out) **+0.0562**,
  measured against the clean-only control. Memorization would produce the
  opposite ordering.
- **A positive composition penalty is not automatically good.** The clean-only
  control also shows `chain_light` at +0.0114 — but only because its
  `resize_0.5x` component was itself broken (AUC 0.9661, FPR 0.90), so adding
  JPEG on top made the input *easier*. Read a positive penalty only after
  checking the components are healthy.

---

## Scoreboard

Competition metric is `0.5*AUC_clean + 0.5*AUC_robust`, pooled.

| Run | Model | val-s2000 | demo_val-s2000 |
|---|---|---|---|
| 2 | clean-trained, clean-only eval | AUC 0.9996 (no robust measure) | AUC 0.9949 |
| 4 | clean-trained, full grid | 0.9388 | 0.9484 |
| 5 control | clean-only, 6k images | 0.9472 | 0.9631 |
| 5 control | clean-only, 6k images, 14 epochs | 0.9382 | — |
| 5 | augmented-view trained | 0.9878 | 0.9978 |
| 6 | + chained views in training (unseeded draw) | 0.9892 | 0.9984 |
| **6** | **+ chains, SEEDED canonical (`--seed 42`)** | **0.9886** | **0.9983** |

All rows above except the last predate seeding and carry ~±0.0005 (three
repeat runs measured 0.9803 / 0.9797 / 0.9796 pooled on the same config).
The shipping checkpoint `models/pe-core-l__linear__augchain.pt` is the seeded
one and reproduces exactly.

Worst-case (`0.5*clean + 0.5*min per-view AUC`), the adversarial reading:

| Run | val | binding view |
|---|---|---|
| 4 | 0.9126 | `chain_heavy` |
| 5 | 0.9306 | `chain_heavy` |
| **6 (seeded)** | **0.9551** | `chain_heavy` 0.9121 ~ `noise_sigma0.1` (tied) |

Next: `chain_heavy` and `noise_sigma0.1` are jointly binding at ~0.91-0.92 and
this benchmark cannot separate them (see the Run 6 correction). Both deep
composition and heavy sensor noise remain open. Resolving which matters needs
an evaluation set with more range -- hence the OOD tier -- not another run
against val.
