# QUARANTINE — do not train on, do not re-download

## sd3/ — `gmongaras/Stable_Diffusion_3_Recaption` — MISLABELLED, REJECTED 2026-08-30

**These 1,500 images are REAL PHOTOGRAPHS that were labelled `label=1` (AIGC).**

The HF repo is a *recaptioning* corpus: real images paired with SD3-authored
captions, for training. It is not a dump of SD3 output. The `_Recaption` /
`-recap` suffix means recaptioned-real, not model-generated. (Contrast
`Photoroom/midjourney-v6-recap`, which IS genuine Midjourney output that was
later recaptioned — same suffix, opposite provenance. The suffix is not a
reliable signal in either direction; check the pixels.)

### Evidence

| check | sd3 | midjourney_v6 | nano_banana |
|---|---|---|---|
| exactly square | 7.5% | 100% | 100% |
| distinct resolutions in 400 imgs | **281** | 1 | 1 |
| modal sizes | 500x500, 640x480, 800x600, 500x400 | 1024x1024 | 1024x1024 |
| rejected under-384px at download | **78%** (5,507 / 7,042) | — | — |

No text-to-image model emits 640x480, 800x600, or 281 distinct resolutions,
and a 1024x1024 renderer cannot produce 78% of output below 384px. Visual
confirmation: `sd3_000011.jpg` is a scraped e-commerce product photo with a
burned-in "I wanne Buy" watermark.

The shipping head `pe-core-l__linear__trainext.pt`, which never saw this data,
scores it at mean P(AIGC) = **0.0230** — indistinguishable from genuine
photographs (0.0178) and nowhere near the AIGC cluster (0.9798). It is not
failing to detect these; it is correctly seeing photographs.

### Damage when trained on

Adding it as AIGC forces the boundary to enclose a region of feature space
occupied by real photos, dragging genuine reals across with it:

| ood metric | control | +sd3 | sd3 removed |
|---|---|---|---|
| DALLE2 degraded AUC | 0.7927 | **0.7205** | 0.8203 |
| DALLE2 clean AUC | 0.9789 | **0.9428** | 0.9819 |
| real FPR@t | 0.011 | **0.019** | 0.015 |
| train_loss (epoch 1) | — | **0.2599** | 0.1275 |

SD3 alone contributed roughly half the total training loss — the signature of
unfittable label noise. Removing it recovered the entire regression.

Same fault family as the SID_Set aspect-ratio shortcut and the depth-map fault:
a label that correlates with something other than the generator.
