# Dataset table

Every corpus touched by the project, its category, and whether it was trained on,
held out, or rejected. Shipping head: `pe-core-l__linear__allsev_e1.pt` @
threshold 0.980 — `aigcmodern_nosd3_e1.pt` @ 0.985 was the head when this table
was first written and is now a superseded arm under `models/`.

## Master table

| Dataset | Source | Category | n | Composition | Status |
|---|---|---|---|---|---|
| `train_ext` | Tiny-GenImage + AIGC-Detection-Benchmark (disjoint later slice) | Mixed real + AIGC | 30,919 | 17,078 real / 13,841 AIGC, 13 generators | **Trained** |
| SID_Set reals | SID_Set (real half only) | **Real domain** | 4,000 | 4,000 real | **Trained** |
| Unsplash | `wtcherr/unsplash_5k` | **Real domain** | 4,000 | 4,000 real | **Trained** |
| Midjourney v6 | `Photoroom/midjourney-v6-recap` | **Modern generator (2024)** | 1,500 | 1,500 AIGC | **Trained** |
| nano-banana | `bitmind/nano-banana` | **Modern generator (2025)** | 1,500 | 1,500 AIGC | **Trained** |
| `val` | Tiny-GenImage (15% split) | In-distribution | 4,200 | 2,100 / 2,100, 7 generators | Eval only — saturated |
| `heldout` | Tiny-GenImage official validation split | In-distribution, unseen images | 7,000 | 3,500 / 3,500, 7 generators | Eval only — saturated |
| `demo_val` | COCO val2017 + WildFake "DALL·E Advanced" | **The brief's §5.4 benchmark** | 13,843 | 5,000 real / 8,843 AIGC | **Eval only — never trained, never tuned against** |
| `ood` | AIGC-Detection-Benchmark | Cross-generator | 8,200 | 4,000 real / 4,200 AIGC, 17 generators | **Eval only** — see caveat below |
| `wildrf_test` | WildRF (arXiv:2406.09398) | **Real-world, in-the-wild** | 2,503 | 1,251 real / 1,252 AIGC | **Eval only** — real Reddit/X/Facebook, already platform-compressed |
| `dalle3_holdout` | `OpenDatasets/dalle-3-dataset` | **Held-out modern generator** | 1,500 | 1,500 AIGC | **HELD OUT — the generalisation test** |
| `wildrf_real` | WildRF train split | Real domain | 1,555 | 1,555 real | Built, **not in the shipping recipe** (used to test whether in-domain reals inflate the WildRF result) |

## Generators in the training pool (13)

ADM · BigGAN · CycleGAN · GLIDE · GauGAN · Midjourney · SD14 · SD15 · StarGAN ·
StyleGAN · StyleGAN2 · VQDM · Wukong

## Rejected — evaluated and deliberately not used

| Dataset | Category | Why rejected | Where it is now |
|---|---|---|---|
| `gmongaras/Stable_Diffusion_3_Recaption` | **Mislabelled** | A *recaptioning* corpus — real photographs paired with SD3-authored captions, **not** SD3 output. 1,500 real photos were labelled AIGC; cost DALL·E 2 seven points of recall and contributed ~half the training loss. | `data/quarantine/` with full evidence; removed from the CLI so it cannot be re-pulled |
| Pexels depth-map mirror | **Mislabelled** | `pexels-110k-768p-min-jpg-depth-anything-large-hf` ships Depth Anything *outputs* named after photographs it does not contain. Single-channel; `convert("RGB")` widened them silently. 4,000 trained as REAL. | Removed; `PEXELS_REAL_MANIFEST` never rebuilt |
| SID_Set (paired real/fake) | **Shortcut** | Real and AI halves separable at 0.93 balanced accuracy from an 8×8 greyscale thumbnail alone — an aspect-ratio / composition leak. | Paired split dropped; **real half re-added** once the aspect-preserving resize + square crop closed the leak |
| CIFAKE | **Harmful** | Measured as actively harmful to transfer once mixed in. | Not used |

## Caveat — the "10 unseen generators" label is stale

`ood` is described in older docs as *"10 generators absent from training."* That
was true when the head trained on `train` (7 generators). The **shipping head
trains on `train_ext` (13 generators)**, so relative to the model that actually
ships:

| | Generators |
|---|---|
| **Seen** (13) | ADM, BigGAN, CycleGAN, GLIDE, GauGAN, Midjourney, SD14, SD15, StarGAN, StyleGAN, StyleGAN2, VQDM, Wukong |
| **Genuinely unseen** (4) | **DALLE2, ProGAN, SDXL, WhichFaceIsReal** |

So `ood` is now largely a *seen-generator, unseen-image* tier with four true
holdouts, and **`dalle3_holdout` is the only clean unseen-generator evidence** —
which is exactly why it was held out. Prefer "four truly unseen generators in
`ood`, plus a fully held-out modern generator" over the old "10 unseen" phrasing.

Source: `src/aigc_detect/registry/corpora.yaml` (the corpus registry), the
recipes under `data/manifests/` and their resolved CSVs, and
`data/quarantine/README.md`.
