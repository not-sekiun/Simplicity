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
      ──> threshold 0.95 ──> verdict
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

**Training** (35,800 images):

| corpus | n | role |
|---|---|---|
| Tiny-GenImage `train` | 23,800 | 7 generators + ImageNet reals |
| SID_Set reals | 4,000 | OpenImages-register real photographs |
| Unsplash | 4,000 | curated photography — a real domain ImageNet does not cover |

**Evaluation** (never trained on, enforced by directory layout):

| tier | n | what it answers |
|---|---|---|
| `val` | 4,200 | in-distribution sanity |
| `heldout` | 7,000 | same generators, unseen images |
| `ood` | 8,200 | **10 generators absent from training** |
| `demo_val` | 13,843 | the brief's §5.4 self-reported benchmark |
| `wildrf_test` | 2,503 | **real Reddit/X/Facebook images**, platform re-encoded |

WildRF (arXiv:2406.09398) is the tier we care most about: real photographs and
real AI images as they actually circulate, already carrying platform compression.

**The result that mattered.** Adding SID_Set reals and 4,000 Unsplash photographs
— no modelling change at all — took WildRF false positives from **33.0% to
18.3%** at unchanged recall, and *improved* the unseen-generator tier at the same
time. Domain coverage of the REAL class was worth more than any architecture
change we tried.

> **Show:** the corpus ledger page; the FPR-per-real-source table.

---

## 6. Robustness evaluation (§5.5.4)

Per-view AUC for the shipping head. Views marked ✗ were **never trained on**.

| view | trained | OOD AUC | WildRF AUC |
|---|---|---|---|
| clean | ✓ | 0.9961 | 0.9935 |
| jpeg q90 | ✗ | 0.9918 | 0.9929 |
| jpeg q70 | ✓ | 0.9905 | 0.9928 |
| jpeg q50 | ✗ | 0.9892 | 0.9930 |
| jpeg q30 | ✗ | 0.9786 | 0.9916 |
| blur σ0.5 | ✗ | 0.9956 | 0.9932 |
| blur σ1.0 | ✓ | 0.9886 | 0.9918 |
| blur σ2.0 | ✗ | 0.9587 | 0.9855 |
| resize 0.5× | ✓ | 0.9827 | 0.9908 |
| resize 0.25× | ✗ | 0.9437 | 0.9799 |
| noise σ0.02 | ✗ | 0.9478 | 0.9761 |
| noise σ0.05 | ✓ | 0.9228 | 0.9182 |
| **noise σ0.1** | ✗ | **0.8616** | **0.8373** |
| colour jitter | ✓ | 0.9947 | 0.9928 |
| centre crop 80% | ✓ | 0.9940 | 0.9925 |
| chain light | ✗ | 0.9792 | 0.9903 |
| chain medium | ✗ | 0.9346 | 0.9740 |
| **chain heavy** | ✗ | **0.8665** | **0.9504** |

**Headline:** clean AUC **0.9961** against 10 unseen generators; **0.9935** on
real social media; 18-view mean **0.9620** / **0.9743**.

**Weakest point, stated plainly:** heavy additive noise (σ0.1), at 0.84–0.86.
Composition costs more than any single axis — `chain_heavy` is well below every
transform it is built from, which a single-transform grid cannot see.

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

**What we changed so neither recurs:**
- `scripts/audit_corpora.py` fingerprints every corpus by saturation and
  bytes/pixel and exits non-zero on an outlier. Real photography sits at
  0.30–0.36 saturation; the depth maps sat at 0.000.
- The puller checks channel mode *before* RGB conversion, aborts a
  majority-greyscale pull, and refuses to overwrite a populated directory.
- The index rebuild no longer derives labels at all — it recovers them from the
  existing index and hard-fails on any unaccounted image.

> **Talking point:** metrics did not catch either fault. Both were invisible to
> AUC and to the fingerprint checks. Four images would have saved days — so the
> audit now looks at pixels, and it runs before training.

---

## 8. Error analysis (§5.5.5)

**False positives — what still trips it.** Real social media photography, ~18%
at threshold 0.5, concentrated in enthusiast/edited photography rather than
casual snapshots. Per platform: Facebook 24.4%, Reddit 18.7%, Twitter 14.7%.
Web-snapshot imagery (Google Images register) barely trips at all — that domain
is closest to the ImageNet-derived training reals.

**Why:** the training pool's real half was one dataset. Any real image from an
absent domain drifts upward. This is a *calibration* gap, not a representation
one — the ranking stays intact (AUC 0.9935), the scores just shift.

**Which is why the threshold is 0.95, not 0.5:**

| threshold | FPR | TPR |
|---|---|---|
| 0.50 | 18.8% | 99.5% |
| **0.95** | **2.8%** | **95.8%** |
| 0.99 | 0.9% | 89.0% |

Chosen on a **held-out split** — WildRF split by image, threshold picked by F1 on
one half and reported on the other, so it is not tuned on the tier it is reported
against. A 6.7× reduction in false positives for 3.7 points of recall. For this
product that trade is correct: telling someone their own photograph is fake costs
far more than missing one AI image among many.

**False negatives.** Per-generator recall at 0.5 on the unseen tier, weakest
first: **DALLE2 0.817**, Midjourney 0.876, ADM 0.894 — everything else is ≥0.97.

Worth stating rather than burying: DALLE2 is both the weakest generator *and* a
**diffusion** model held out of training, i.e. squarely in the brief's threat
model. The GAN families we generalise to easily (StyleGAN 0.976, CycleGAN 0.983)
are the out-of-scope ones. So our worst case is the case that matters most, and
it is the clearest argument for the data-scaling work in §11 rather than for any
architectural change.

**Trade-offs we accept.** Higher threshold → misses subtle AI. Frozen backbone →
cannot learn generator-specific artifacts, which is exactly why it generalises.
Linear head → leaves in-distribution accuracy on the table, deliberately.

---

## 9. The demo

A Chrome extension calling a local FastAPI server (`demo/server.py`) that loads
the checkpoint and scores images in-page. Everything about "which model" comes
from the checkpoint — swap `--head` to a different backbone entirely and the
extension needs no changes.

> **Show, in order:** an AI-art subreddit (flags reliably) → r/itookapicture
> (mostly clean at 0.95) → a Google Images search → a deliberate hard case.

---

## 10. Honest limitations

1. **Real-domain coverage is still the binding constraint.** Two real domains
   added; more would keep helping. This is the whole remaining problem.
2. **Heavy noise and heavy composed degradation** drop to ~0.84–0.87 AUC.
3. **Threshold is global.** Per-domain thresholds would beat one number, and
   temperature calibration is unimplemented.
4. **GAN families are reported separately** and treated as out of scope; the
   brief's threat model is diffusion.
5. **Compute-bound, not idea-bound.** One RTX 3080. Every ablation is a data
   ablation because that is what fits.
6. **`val` and `heldout` are near-saturated** and no longer discriminate; OOD and
   WildRF are the only tiers that still move.

---

## 11. Takeaway

**A frozen VFM with a linear probe genuinely works.** 0.9961 clean AUC against
ten generators it has never seen, from 1,025 trainable parameters on a consumer
GPU. We tested the obvious upgrade — deeper heads — and it made real-world
performance *worse*.

The remaining work is not architectural. It is **scaling training-data diversity
and depth**: more real domains, more generators, more images. Every measurable
gain this project made came from data, and every dead end came from assuming a
number meant what it appeared to mean.

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
WildFake DALL·E Advanced (demo-val).
**Papers:** "Simplicity Prevails" (arXiv:2602.01738) — the frozen-backbone +
linear-probe recipe. UniversalFakeDetect, Ojha et al. CVPR 2023 — linear probes
generalise where deep classifiers do not. WildRF, arXiv:2406.09398.
