# Transform table

All 22 views. The head trains on **19**; the robustness grid scores **18**. They
overlap on 15, so **3 of the 18 scored views are held out** — the three composed
chains — and the 4 training chains are never scored.

The shipping head trains on every *severity* of every family. That was measured,
not assumed: the previous head trained one severity per family, and swapping to
full severity coverage improved every tier except a saturated demo_val, with the
largest gains on the composed chains it still does **not** train on.

| # | View | Family | Parameter | Real-world analog | In 18-view grid | Trained on | Status |
|---|---|---|---|---|---|---|---|
| 1 | `clean` | — | none | pristine upload | Yes | Yes | baseline |
| 2 | `jpeg_q90` | JPEG | q = 90 | light social re-encode | Yes | Yes | Trained |
| 3 | `jpeg_q70` | JPEG | q = 70 | typical CDN re-encode | Yes | Yes | Trained |
| 4 | `jpeg_q50` | JPEG | q = 50 | messaging app | Yes | Yes | Trained |
| 5 | `jpeg_q30` | JPEG | q = 30 | aggressive recompression | Yes | Yes | Trained |
| 6 | `blur_sigma0.5` | Gaussian blur | σ = 0.5 | slight defocus | Yes | Yes | Trained |
| 7 | `blur_sigma1.0` | Gaussian blur | σ = 1.0 | out-of-focus | Yes | Yes | Trained |
| 8 | `blur_sigma2.0` | Gaussian blur | σ = 2.0 | heavy defocus | Yes | Yes | Trained |
| 9 | `resize_0.5x` | Resize round-trip | 0.5× then upscale | thumbnail generation | Yes | Yes | Trained |
| 10 | `resize_0.25x` | Resize round-trip | 0.25× then upscale | small thumbnail | Yes | Yes | Trained |
| 11 | `noise_sigma0.02` | Gaussian noise | σ = 0.02 | mild sensor noise | Yes | Yes | Trained |
| 12 | `noise_sigma0.05` | Gaussian noise | σ = 0.05 | low-light sensor noise | Yes | Yes | Trained |
| 13 | `noise_sigma0.1` | Gaussian noise | σ = 0.10 | heavy sensor noise | Yes | Yes | Trained — still the worst SINGLE transform on every tier |
| 14 | `color_jitter` | Colour jitter | ±20% brightness / contrast / saturation | filter apps, auto-enhance | Yes | Yes | Trained |
| 15 | `center_crop_80` | Centre crop | 80% of frame | profile-pic cropping | Yes | Yes | Trained |
| 16 | `chain_light` | Chain (depth 2) | resize 0.5× → JPEG q70 | a single re-upload | Yes | No | **Held out** (composition) |
| 17 | `chain_medium` | Chain (depth 4) | crop 80% → jitter → resize 0.5× → JPEG q50 | screenshot → filter app → re-upload | Yes | No | **Held out** (composition) |
| 18 | `chain_heavy` | Chain (depth 4) | blur σ1.0 → resize 0.25× → noise σ0.05 → JPEG q30 | a repost of a repost | Yes | No | **Held out** (composition) — worst OOD view, 0.8878 |
| 19 | `trainchain_a` | Chain (depth 2) | blur σ1.0 → JPEG q70 | — | No | Yes | Training-only |
| 20 | `trainchain_b` | Chain (depth 2) | noise σ0.05 → JPEG q70 | — | No | Yes | Training-only |
| 21 | `trainchain_c` | Chain (depth 3) | crop 80% → jitter → blur σ1.0 | — | No | Yes | Training-only — **does not end in JPEG** |
| 22 | `trainchain_d` | Chain (depth 4) | jitter → resize 0.5× → noise σ0.05 → JPEG q70 | — | No | Yes | Training-only |

## Severity coverage (the brief's §5.2 table)

| Family | Severities implemented | Trained | Held out |
|---|---|---|---|
| JPEG compression | 90, 70, 50, 30 | **all four** | — |
| Gaussian blur | 0.5, 1.0, 2.0 | **all three** | — |
| Resize (down then up) | 0.5×, 0.25× | **both** | — |
| Gaussian noise | 0.02, 0.05, 0.10 | **all three** | — |
| Colour jitter | ±20% | **±20%** | — |
| Centre crop | 80% | **80%** | — |

**Every severity is trained; composition is what is held out.** The three scored
chains are the only views the head has never seen in any form, and they are where
full severity coverage bought the most — at a matched 2.5% false-positive rate,
OOD recall on those three chains went 0.418 → 0.610 against the previous head.

The earlier one-severity-per-family head is kept as the ablation
(`pe-core-l__linear__aigcmodern_nosd3_e1.pt`), and a 22-view head that trains on
the scored chains too was also built: it is **worse** than this one on the
competition metric (0.99270 vs 0.99357), so training on the evaluation views is
not simply "more coverage is better".

## Design rules enforced in code

- **Training chains use only already-trained severities.** The single new
  variable in the chain columns is *composition*, not new severities.
- **Training chains are disjoint from scored chains.** Training on the exact
  compositions you then report would make those three columns measure
  memorisation, and they are the only evidence about composition at all.
- **`trainchain_c` deliberately does not end in JPEG**, so the head cannot learn
  "a chain is the thing that ends in a re-encode".
- **`_validate_chain_specs()` forbids repeating a transform family inside one
  chain.** Blur σ2.0 twice is σ_eff = 2.83, outside the brief's table; nothing
  would otherwise flag that the claimed robustness envelope had widened.
- **Everything is deterministic.** Training runs on cached embeddings, so each
  image contributes a fixed set of rows. The stochastic `RobustnessAugment` /
  `build_train_transform` exist but are **not** in the training path — the only
  caller is `main.py preview-augment`.

Source: `src/aigc_detect/transforms.py` (`CHAIN_SPECS`, `TRAIN_CHAIN_SPECS`),
`src/aigc_detect/train_head.py` (`TRAIN_VIEWS_ALL_SEVERITIES`).
