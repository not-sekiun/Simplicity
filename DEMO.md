# DEMO.md — AIGC Image Detection (TikTok TechJam 2026, Track 5)

Narration script and reference for the demo video. Sections map to the §5.5
deliverables and the §5.6 judging criteria; the right-hand notes say what to show
on screen.

**One-line thesis:** a frozen vision foundation model plus a single linear layer
detects AI-generated images well enough to ship, and everything that actually
moved the numbers was data work, not modelling.

---

## 1. The problem, framed sharply

AI image detectors are usually reported as one number on one benchmark. Two
things break that number in the real world:

1. **Unknown generators.** A detector trained on today's models meets tomorrow's.
   Our seven training generators are GenImage-era; the competition's test
   generators are unstated.
2. **Unknown *real* domains.** This is the failure nobody reports. A detector
   whose "real" class is one dataset learns *that dataset*, not "photographs".
   Real images from an unseen domain get called AI with high confidence.

The second one is the product-killing failure. A false positive tells a person
their own photograph is fake. We built the evaluation around it.

> **Show:** the Chrome extension scoring a Reddit feed, then a Google Images grid.

---

## 2. Architecture

```
image ──> aspect-preserving resize + centre crop to 336px
      ──> PE-Core-L (frozen, 316M params, no gradients ever)
      ──> 1024-d pooled embedding
      ──> standardise with the TRAIN-set scaler stored in the checkpoint
      ──> Linear(1024 -> 1) ──> sigmoid ──> P(AIGC)
      ──> threshold 0.980 ──> verdict
```

**Backbone:** `timm/vit_pe_core_large_patch14_336.fb` (Meta Perception Encoder),
316M parameters — comfortably under the 2B limit. Frozen throughout; we never
backprop into it.

**Head:** one linear layer. 1,025 trainable parameters.

Because the backbone is frozen, every image is embedded **once** and cached as
an `.npz`. Training a head is then seconds of CPU on cached vectors, not a GPU
run. That is the single decision that made everything else affordable on one
consumer RTX 3080 — it turned "train a model" into "fit a boundary", so we could
run dozens of data ablations instead of one or two training runs.

> **Show:** `heads.py` (it fits on one screen), then a training run finishing in
> ~90 seconds.

---

## 3. Why a linear layer — measured, not assumed

We tested deeper heads. Same data, same views, same seed, only the head differs:

| head | val clean | OOD mean AUC | **WildRF clean AUC** | **WildRF FPR @ TPR .98** |
|---|---|---|---|---|
| **linear** | 0.9987 | 0.9620 | **0.9935** | **0.051** |
| MLP (1×512) | **0.9996** | 0.9637 | 0.9867 | 0.109 |
| MLP (1024, 512) | 0.9997 | 0.9597 | 0.9861 | 0.076 |

*(Run on the previous training pool, before `train_ext` was added — all three
arms share that pool, so the comparison is internally valid. The shipping head's
absolute numbers are higher; see §6.)*

**The ranking inverts between the tier you fit and the tier you ship to.** Both
MLPs beat linear on validation by a margin that looks like a real upgrade, and
both more than double real-world false positives at matched recall.

This reproduces UniversalFakeDetect (Ojha et al., CVPR 2023): on frozen features
a linear probe transfers to unseen generators better than a trained deep
classifier, whose in-distribution advantage inverts off-distribution. The
backbone already did the representation learning; extra head capacity is spent
memorising the seven training generators, and generalisation is the entire job.

> **Talking point:** this is why we report on four tiers instead of one. A single
> validation number would have chosen the wrong architecture.

---

## 4. The pipeline

| stage | command | notes |
|---|---|---|
| Download | `main.py download …` | streamed + capped, re-encoded JPEG q95 |
| Split | `main.py split` | globs `data/raw/` only — eval tiers physically cannot leak |
| Audit | `scripts/audit_corpora.py` | corpus health gate (see §7) |
| Embed | `main.py embed-views` | one decode → all views, cached per `(backbone, manifest, view)` |
| Train | `main.py train-head-views` | seconds, on cached embeddings |
| Evaluate | `main.py eval-grid` | 18 views per tier |
| Error analysis | `main.py error-analysis` | §5.5.5 |
| Predict | `main.py predict --input_dir … --output preds.json` | §5.5.2 |

Every cache is keyed by a **fingerprint of the manifest's `image_path` column**,
so rebuilding a split invalidates exactly the caches it should and no others.

**Robustness by construction.** The head trains on 11 views — clean plus one
severity per degradation family plus four composed chains. The 3 *scored* chains
and the harder severities are **never trained on**, so the robustness numbers in
§6 are held-out, not fitted.

> **Show:** `data/embeddings/` filling up; a `train-head-views` run.

---

## 5. Data — what we used and why

**Training** (41,919 images):

| corpus | n | role |
|---|---|---|
| `train_ext` | 30,919 | Tiny-GenImage's 7 generators + 6 more + ImageNet reals |
| SID_Set reals | 4,000 | OpenImages-register real photographs |
| Unsplash | 4,000 | curated photography — a real domain ImageNet does not cover |
| Midjourney v6 | 1,500 | **modern generator (2024)** — art / illustration register |
| nano-banana | 1,500 | **modern generator (2025)**, Gemini 2.5 Flash Image |

The two modern corpora come from **different publishers on purpose**. One dataset
is one provenance — one prompt distribution, one resolution, one encoder — and a
detector will learn "this source" as readily as "this generator". Mixing
generators inside one source does not fix that; the confound is source-level.

**Evaluation** (never trained on, enforced by directory layout):

| tier | n | what it answers |
|---|---|---|
| `val` | 4,200 | in-distribution sanity |
| `heldout` | 7,000 | same generators, unseen images |
| `ood` | 8,200 | **10 generators absent from training** |
| `demo_val` | 13,843 | the brief's §5.4 benchmark — 5,000 COCO val2017 + 8,843 DALL·E Advanced |
| `wildrf_test` | 2,503 | **real Reddit/X/Facebook images**, platform re-encoded |
| `dalle3_holdout` | 1,500 | **DALL·E 3 — a modern generator held out of training entirely** |

`dalle3_holdout` is the tier that makes the modern-generator claim falsifiable.
We pulled three current generators and deliberately trained on only two, so the
third answers the question the other tiers cannot: does this generalise to a
modern model it has never seen, from a publisher it has never seen?

WildRF (arXiv:2406.09398) is the tier we care most about: real photographs and
real AI images as they actually circulate, already carrying platform compression.

**The result that mattered.** Adding SID_Set reals and 4,000 Unsplash photographs
— no modelling change at all — took WildRF false positives from **33.0% to
18.3%** at unchanged recall, and *improved* the unseen-generator tier at the same
time. Domain coverage of the REAL class was worth more than any architecture
change we tried.

**And it held on the third try.** `train_ext` added 6 new generators plus 5,178
more real images. The generators moved the family we had already solved — GAN
recall 0.966 → 0.992 — and moved diffusion **not at all** (0.859 → 0.859). The
reals moved the thing we care about: WildRF FPR at matched recall .051 → .034.
We predicted this from the corpus composition before running it, because 1,617 of
its 1,941 new AI images are GAN.

**Then, on the fourth try, generator-side data finally won — once we fixed it.**
Adding Midjourney-v6 + nano-banana improved every tier *and* lifted DALL·E 3, a
generator from neither source, on all 18 views. The rule from the three earlier
rounds still holds with its own stated caveat: generator data helps when the
generators are ones you are actually failing on. The first three rounds added
generators we had already solved. This one did not.

> **Talking point:** three times, real-side data beat generator-side data — and
> the fourth time it did not, for a reason we can now state precisely. That
> progression, not any single number, is the project's actual finding.

> **Show:** the corpus ledger page; the FPR-per-real-source table.

---

## 6. Robustness evaluation (§5.5.4)

**Compact summary — clean vs transformed, all four tiers.** One fixed threshold
(0.980) applied to every view; re-tuning per view is the standard way to make a
fragile detector look robust, so we do not do it.

| tier | n | clean AUC | transformed (17-view mean) | worst transform | clean BAcc | transformed BAcc |
|---|---|---|---|---|---|---|
| **demo_val** (the brief's §5.4 benchmark) | 13,843 | **0.9999** | **0.9960** | 0.9641 | 0.9890 | 0.9628 |
| **OOD** (10 unseen generators) | 8,200 | **0.9982** | **0.9724** | 0.8878 | 0.9490 | 0.8542 |
| **WildRF** (real social media) | 2,503 | **0.9969** | **0.9875** | 0.9273 | 0.9796 | 0.9518 |
| **DALL·E 3** (held out, unseen) | 1,500 | **0.9988** | **0.9917** | 0.9411 | 0.9840 | 0.9641 |

The worst transform is `noise_sigma0.1` on three tiers and `chain_heavy` on OOD.
The head now trains on every noise severity, so the remaining failure has moved
to *composition* — the only thing in the grid it has never seen.

**Single vs composed degradation**, the axis a single-transform grid cannot see:

| tier | single-transform mean (14) | chained mean (3) | composition penalty |
|---|---|---|---|
| OOD | 0.9786 | 0.9435 | **-0.0351** |
| WildRF | 0.9883 | 0.9836 | -0.0047 |
| demo_val | 0.9958 | 0.9970 | +0.0012 |
| DALL·E 3 | 0.9914 | 0.9932 | +0.0018 |

Negative means composition costs more than any single axis. It is worst on OOD
and *positive* on demo_val and DALL·E 3 — there, degradation moves images toward
the training domain, so the clean view is the outlier rather than the chains.

> **Chart:** `stats/charts/07_robustness_summary.png`. Underlying data:
> `stats/robustness_summary.csv` and `stats/per_view_auc.csv`.
> Note: the 18-view grid runs on a 2,000-row sample of demo_val and a 4,000-row
> sample of OOD; clean AUC on full demo_val is unchanged at 0.9999.

---

### Full per-view detail

Per-view AUC for the shipping head, on all four evaluation tiers. Views marked ✗
were **never trained on**. The shipping head trains on every *severity*, so only
3 of 18 are held out — the three composed chains. That makes the ✗ rows the only
unfitted numbers here, and they are the ones to read.

| view | trained | OOD AUC | WildRF AUC | demo_val AUC |
|---|---|---|---|---|
| clean | ✓ | 0.9982 | 0.9969 | 0.9999 |
| jpeg q90 | ✓ | 0.9978 | 0.9973 | 1.0000 |
| jpeg q70 | ✓ | 0.9971 | 0.9962 | 0.9999 |
| jpeg q50 | ✓ | 0.9956 | 0.9960 | 0.9999 |
| jpeg q30 | ✓ | 0.9898 | 0.9957 | 0.9999 |
| blur σ0.5 | ✓ | 0.9976 | 0.9971 | 0.9999 |
| blur σ1.0 | ✓ | 0.9909 | 0.9971 | 0.9997 |
| blur σ2.0 | ✓ | 0.9737 | 0.9935 | 0.9986 |
| resize 0.5× | ✓ | 0.9871 | 0.9969 | 0.9996 |
| resize 0.25× | ✓ | 0.9660 | 0.9927 | 0.9980 |
| noise σ0.02 | ✓ | 0.9725 | 0.9896 | 0.9970 |
| noise σ0.05 | ✓ | 0.9490 | 0.9644 | 0.9849 |
| **noise σ0.1** | ✓ | **0.8895** | **0.9273** | **0.9641** |
| colour jitter | ✓ | 0.9972 | 0.9966 | 0.9999 |
| centre crop 80% | ✓ | 0.9971 | 0.9967 | 0.9999 |
| chain light | ✗ | 0.9871 | 0.9947 | 0.9998 |
| chain medium | ✗ | 0.9555 | 0.9833 | 0.9979 |
| **chain heavy** | ✗ | **0.8878** | **0.9728** | 0.9932 |
| **clean** | | **0.9982** | **0.9969** | **0.9999** |
| **18-view mean** | | **0.9724** | **0.9875** | **0.9960** |

**Held-out modern generator — the number we are proudest of.** DALL·E 3, never
trained on, scored at the shipping threshold:

| DALLE3 | previous head | **shipping head** |
|---|---|---|
| clean recall | 0.9887 | **0.9920** |
| 18-view mean recall | 0.9437 | **0.9580** |
| worst-view recall | 0.6873 | **0.7227** |

Both heads are held at the **same 2.5% WildRF false-positive rate** here, so
recall is the only free variable. Reading each head at its own threshold instead
would not be a comparison: a head that simply flags more looks better on recall
while quietly spending false positives.

**Weakest point, stated plainly:** heavy additive noise (σ0.1) at 0.89 on OOD,
and `chain_heavy` at 0.89. The chain is the more honest number of the two — it is
a composition the head has never seen in any form, while every noise severity is
now trained on. `chain_heavy` sits below every transform it is built from, which
a single-transform grid cannot see.

---

## 7. Two data faults we caught — and how

This is the part we would most like judges to ask about. Both were found by
**looking at the data**, not by a metric moving.

**(a) A "Pexels photography" corpus that was depth maps.** The mirror
`pexels-110k-768p-min-jpg-depth-anything-large-hf` ships Depth Anything *outputs*
named after photographs it does not contain. Every file is single-channel;
`img.convert("RGB")` widens that to three identical channels without error, and
they cleared our 384px floor at 768p. 4,000 trained as REAL.

Caught by *scoring the corpus*: mean P(AIGC) 0.999 with 100% over threshold —
more AI-looking than the actual AI training set at 0.980.

**(b) A benchmark class labelled backwards.** `WhichFaceIsReal` in the OOD tier
holds StyleGAN faces, not photographs. Our index-rebuild step derived labels from
directory names, flipping 250 rows to "real". Those 250 correctly-detected fakes
were scored as false positives — which manufactured a "100% false-positive rate
on human portraits" and, with it, an entire theory about portraits being missing
from our training data. Days of work, all downstream of one wrong dictionary
entry.

Correcting it moved OOD clean AUC from 0.9670 to **0.9961**.

**(c) A "Stable Diffusion 3" corpus that was real photographs.**
`gmongaras/Stable_Diffusion_3_Recaption` is a *recaptioning* corpus — real
images paired with SD3-authored captions — not SD3 output. We pulled 1,500 and
labelled them AIGC. Training on them forced the boundary to enclose a region of
feature space occupied by genuine photographs, dragging real images across with
it: DALLE2 recall fell 0.79 → 0.72, real FPR rose 1.1% → 1.9%, and SD3 alone
contributed roughly **half the total training loss** — the signature of
unfittable label noise.

Caught the same way as (a): *scoring the corpus.* The previous head, which had
never seen it, rated it 0.023 — statistically identical to real photographs
(0.018) and nowhere near the AI class (0.980). It was not failing to detect
them; it was correctly seeing photographs. Confirmed on provenance: 281 distinct
resolutions across 400 images and 7.5% square, against exactly one resolution
and 100% square for both genuine generator dumps.

> The trap: `-recap` / `_Recaption` in a dataset name usually means
> *recaptioned real images*. But `Photoroom/midjourney-v6-recap` carries the
> same suffix and IS genuine output. The name is not a provenance signal in
> either direction — the pixels are. A generator dump has one or two
> resolutions; a scraped corpus has hundreds. That check takes ten seconds.

**What we changed so none recur:**
- `scripts/audit_corpora.py` fingerprints every corpus by saturation and
  bytes/pixel and exits non-zero on an outlier. Real photography sits at
  0.30–0.36 saturation; the depth maps sat at 0.000.
- The puller checks channel mode *before* RGB conversion, aborts a
  majority-greyscale pull, and refuses to overwrite a populated directory.
- The index rebuild no longer derives labels at all — it recovers them from the
  existing index and hard-fails on any unaccounted image.
- `scripts/audit_data.py` ran only over `data/raw/`, so every corpus added after
  it was **never probed** — the audit reported nothing wrong by not looking.
  It now spans every corpus directory, and a new one must be registered there.
- Rejected corpora are quarantined with their evidence and removed from the CLI,
  so a future contributor cannot re-pull them by accident.

> **Talking point:** metrics did not catch either fault. Both were invisible to
> AUC and to the fingerprint checks. Four images would have saved days — so the
> audit now looks at pixels, and it runs before training.

---

## 8. Error analysis (§5.5.5)

**False positives — what still trips it.** Real social media photography,
concentrated in enthusiast/edited photography rather than casual snapshots.
Measured on WildRF's clean view, by platform:

| | Facebook | Reddit | Twitter | **overall FPR** | **TPR** |
|---|---|---|---|---|---|
| at 0.50 | 36.3% | 14.3% | 22.6% | 19.3% | 99.5% |
| **at 0.980 (shipping)** | 5.6% | 0.8% | 4.4% | **2.4%** | **98.3%** |

**Why:** the training pool's real half is finite. Real images from an absent
domain drift upward. This is a *calibration* gap, not a representation one — the
ranking stays intact (AUC 0.9969), the scores just shift.

**Which is why the threshold is 0.980, not 0.5:**

| threshold | FPR | TPR |
|---|---|---|
| 0.50 | 19.3% | 99.5% |
| **0.980** | **2.4%** | **98.3%** |

Chosen on a **held-out split** — WildRF pooled over clean plus the CDN-like views
a browser extension actually sees, split by image so no image appears on both
sides, swept in 0.005 steps, picked by F1 on one half and reported on the other:
FPR **.0215** / TPR **.9797**. An ~8× reduction in false positives for 1.2 points
of recall. For this product that trade is correct: telling someone their own
photograph is fake costs far more than missing one AI image among many.

The threshold is a property of the head, not a constant — we re-derive it on
every swap, and it has moved 0.92 → 0.94 → 0.98 across four heads. It is now
derived by a script rather than by hand (`scripts/derive_threshold.py`), whose
`--verify` mode asserts the protocol still reproduces the table it was first
recorded from.

**False negatives — and the most interesting result in the project.**
Per-generator recall at the shipping threshold, weakest first:

| generator | family | era | recall |
|---|---|---|---|
| ADM | diffusion | 2021 | **0.472** |
| DALLE2 | diffusion | 2022 | **0.573** |
| Midjourney (v4-era) | diffusion | 2022 | 0.694 |
| GLIDE | diffusion | 2021 | 0.830 |
| SD14 / SD15 / VQDM / Wukong | diffusion | 2022 | 0.973–0.991 |
| SDXL | diffusion | 2023 | 0.992 |
| **DALL·E 3 (held out)** | diffusion | **2023** | **0.992** |
| GAN family (8 generators) | gan | 2018–21 | 0.982 mean |

**Our worst generators are now the oldest ones, and our best are the newest.**
ADM and DALLE2 — the two weakest — are 2021–22 research models. DALL·E 3 and
SDXL, the two most recent, sit at 0.99. The 2022-era diffusion mean is 0.865
while the modern held-out generator is 0.992.

That inversion matters more than the averages. The deployment question is not
"does it detect four-year-old research models" — it is "will it hold against
what people actually use, and against what ships next year". The evidence says
the newer the generator, the better this does, and the held-out tier says that
is generalisation rather than memorised provenance.

**Trade-offs we accept.** Higher threshold → misses subtle AI. The legacy-recall
regression earlier heads paid for their modern gains (2022-era diffusion 0.859 →
0.811) is now recovered: training on every transform severity puts it at 0.865
without giving the modern gains back. Held at a matched 2.5% WildRF FPR, OOD
clean recall goes 0.883 → 0.899 against the previous head while DALL·E 3 goes
0.944 → 0.958. Frozen backbone → cannot learn generator-specific
artifacts, which is exactly why it generalises. Linear head → leaves
in-distribution accuracy on the table, deliberately.

**Residual weaknesses we have not fixed.** WildRF gains are carried by Reddit;
Twitter and Facebook FPR are slightly worse than the previous head. And the
worst false-positive view is now `jpeg_q70` — a *trained* view, and the closest
in the grid to real CDN recompression, i.e. the actual deployment condition.

---

## 9. The demo

A Chrome extension calling a local FastAPI server (`demo/server.py`) that loads
the checkpoint and scores images in-page. Everything about "which model" comes
from the checkpoint — swap `--head` to a different backbone entirely and the
extension needs no changes.

> **Show, in order:** an AI-art subreddit (flags reliably) → r/itookapicture
> (mostly clean at 0.980) → a Google Images search → a deliberate hard case.

**The required deliverable script (§5.5.2)** is the same code path:

```
uv run main.py predict --input_dir <dir> --output preds.json
```

which writes a JSON array of `{"image_path": ..., "pred": <float 0-1>}`, one
entry per image — the confidence that the image is AIGC.

---

## 10. Honest limitations

1. **Real-domain coverage is still a binding constraint.** Two real domains
   added; more would keep helping, and the residual FPR is concentrated in
   photography registers we do not cover.
2. **Heavy noise and heavy composed degradation** drop to ~0.87 AUC on OOD.
3. **Threshold is global.** Per-domain thresholds would beat one number, and
   temperature calibration is unimplemented.
4. **GAN families are reported separately** and treated as out of scope; the
   brief's threat model is diffusion.
7. **Composition is the remaining failure mode.** `chain_heavy` is the worst
   OOD view at 0.888 — the only degradation class the head never trains on, so
   it is the one number here that is pure extrapolation.
8. **Only two modern generators in training, one held out.** The generalisation
   claim rests on a single held-out modern generator. Three or four would make
   it much stronger, and that is the first thing we would buy with more time.
5. **Compute-bound, not idea-bound.** One RTX 3080. Every ablation is a data
   ablation because that is what fits.
6. **`val` and `heldout` are near-saturated** and no longer discriminate; OOD and
   WildRF are the only tiers that still move.

---

## 11. Takeaway

**A frozen VFM with a linear probe genuinely works.** 0.9979 clean AUC against
ten unseen generators, 0.9969 on real social media, 0.9999 on the brief's
benchmark — from **1,025 trainable parameters** on a consumer GPU. We tested the
obvious upgrade, deeper heads, and it made real-world performance *worse*.

**The strongest evidence we have is the held-out generator.** We pulled three
current models, trained on two, and kept DALL·E 3 back. It scores 0.990 clean
and improves on all 18 views over the previous head. A detector that gets
*better* on the newest generators, including one it has never seen from a
publisher it has never seen, is the one worth deploying.

**And the process finding is the transferable one.** Three of the four things
that moved our numbers were data corrections, and three separate corpora reached
training mislabelled — depth maps as photographs, GAN faces as real, and real
photographs as SD3 output. None was caught by a metric. All three were caught by
*scoring the corpus* or *looking at the pixels*. The audit that runs before
training is worth more than any architecture we tried.

---

## Appendix — tools, models, libraries (§5.5.1)

**Tools:** VS Code, Claude Code, git, `uv` (dependency + venv management).
**Hardware:** one NVIDIA RTX 3080 (12 GB), Windows 11.
**Models:** PE-Core-L (`timm/vit_pe_core_large_patch14_336.fb`), frozen. Raced
against DINOv3-L and MetaCLIP2-H.
**Libraries:** PyTorch, torchvision, timm, HuggingFace `datasets` + `hub`,
scikit-learn, pandas, NumPy, Pillow, tqdm; FastAPI + uvicorn for the demo.
**Datasets:** Tiny-GenImage (training + heldout), AIGC-Detection-Benchmark
(OOD eval; a disjoint later slice for `train_ext`), SID_Set (reals),
Unsplash (`wtcherr/unsplash_5k`), WildRF (arXiv:2406.09398), COCO val2017 +
WildFake DALL·E Advanced (demo-val), `Photoroom/midjourney-v6-recap` and
`bitmind/nano-banana` (modern generators, training),
`OpenDatasets/dalle-3-dataset` (**held out — never trained on**).
*Rejected:* `gmongaras/Stable_Diffusion_3_Recaption` — mislabelled real
photographs, quarantined; see §7(c).
**Papers:** "Simplicity Prevails" (arXiv:2602.01738) — the frozen-backbone +
linear-probe recipe. UniversalFakeDetect, Ojha et al. CVPR 2023 — linear probes
generalise where deep classifiers do not. WildRF, arXiv:2406.09398.
