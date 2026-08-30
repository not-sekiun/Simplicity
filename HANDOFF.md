# Handoff — the race is DONE and decided; next is deliverable 5.5.5

**Written 2026-08-29, updated after the race completed.**

## THE RACE IS FINISHED AND THE DECISION IS MADE. Do not re-run it.

`pe-core-l` won and ships. Both challengers lost decisively on `ood-s4000`,
against a rule fixed before the numbers were seen (switch only on > +0.010):

| backbone | res | params | ood SCORE | delta |
|---|---|---|---|---|
| **`pe-core-l`** | 336 | 316M | **0.9366** | — |
| `dinov3-l` | 256 | 303M | 0.9041 | -0.0325 |
| `metaclip2-h` | 224 | 630M | 0.8961 | -0.0405 |

Same ordering on the in-scope diffusion-only metric (0.9466 / 0.9140 / 0.9143
clean), on worst-case, and on val. Full write-up: `NARRATIVE.md` Run 7.

**The shipping model is `models/pe-core-l__linear__augchain.pt`**, which is
already `predict.py`'s default. `models/*__linear__race.pt` are the race
artifacts; `pe-core-l__linear__race.pt` is equivalent to the shipping head
(same config, same seed).

**Start at section 0 below** -- it carries the newest findings (the data lever,
the disjoint training slice, and why metaclip2-giant is blocked). Sections 1-2
are retained as the record of how the race decision was made; read them only
if you need to audit it. Deliverable 5.5.5 (error analysis) is still the last
required piece.

---

## 0. LATEST (2026-08-30) — the data lever is the live one; read this first

Two jobs may still be running when you pick this up. Check before assuming:

```
data/embeddings/pe-core-l__train__*.npz   22 files = full-pool embedding DONE
data/train_ext/train_ext_index.csv        exists  = training-slice pull DONE
```

### The finding that changed the plan

I had written that both improvement levers were spent -- augmentation (the
ceiling probe) and backbone (the race). Both true. **But nobody had tested the
DATA lever, and it is the one still paying.** Learning curve, trained on
cached `train-s6000` views, scored on `ood-s4000`:

| train images | rows | ood clean | ood pooled | **ood SCORE** |
|---|---|---|---|---|
| 750 | 8,250 | 0.7889 | 0.6807 | 0.7348 |
| 1,500 | 16,500 | 0.9044 | 0.7753 | 0.8398 |
| 3,000 | 33,000 | 0.9394 | 0.8897 | 0.9145 |
| 6,000 | 66,000 | 0.9532 | 0.9201 | **0.9366** |

**The curve has not flattened** -- 3,000 -> 6,000 still bought +0.022. The
shipping head used **6,000 of the 23,800 available (25%)**. Extrapolating,
the full pool is worth roughly **+0.015-0.025**, larger than the entire
backbone-race spread.

(Caveat: measured on `ood`, which is mild tuning-on-the-benchmark. Safe here
because the curve is monotone and the mechanism is obvious; do NOT use `ood`
to pick a real hyper-parameter.)

### RESULT: the full-pool retrain landed at +0.0041, not the +0.015-0.025 predicted

`models/pe-core-l__linear__fullpool.pt`, trained on all 23,800 images x 11 views
= 261,800 rows (vs the shipping head's 6,000 images).

| | 6k baseline | full pool 23.8k | delta |
|---|---|---|---|
| **ood score** | 0.9366 | **0.9407** | **+0.0041** |
| ood AUC_clean | 0.9532 | 0.9624 | +0.0092 |
| ood AUC_robust (pooled) | 0.9200 | 0.9191 | -0.0009 |
| ood worst (`chain_heavy`) | 0.8099 | 0.8050 | -0.0049 |
| val score | 0.9886 | 0.9870 | -0.0016 |

**The prediction was wrong and the method is the lesson.** +0.015-0.025 was
extrapolated from a 4-point learning curve (3k->6k bought +0.022). The curve had
flattened further out than its visible points implied: a 4x data increase bought
a fifth of what a 2x increase had. Do not put a range on an extrapolation from
four points -- "direction certain, magnitude unknown" was the honest claim.

Two things the breakdown shows:

- **Clean improved, robustness did not** (+0.0092 clean, -0.0009 robust, -0.0049
  worst). More of the SAME distribution sharpens clean discrimination and buys
  no robustness.
- ~~**The portrait bug is completely unmoved.** `WhichFaceIsReal` FPR was 1.000
  before and is 1.000 after.~~ **RETRACTED** -- those 250 images are StyleGAN
  fakes we had relabelled as real, so FPR 1.000 was the model being correct.
  See FINDINGS 2h-CORRECTION. Corrected OOD for the shipping head
  (`photoreal`, re-embedded and re-scored with `eval-grid`):
  clean AUC **0.9961**, 18-view mean **0.9620**, clean FPR **0.0220**,
  clean TPR **0.9660**. Same tier, same head, `augchain`: 0.9940 / 0.9597.

2 epochs is fine; a 1-epoch run scored 0.9404 vs 0.9407, so the epoch-2 dip in
the val log does not carry to ood.

**Volume of the same distribution is an exhausted lever.** The two live ones are
generator diversity (untested) and real-domain coverage (untouched).

### train_ext HOLDS OUT DALLE2 and SDXL -- do not remove that without a replacement

Training on the whole slice would have taken the OOD tier from 9 unseen
generator cells to 1, and from three in-scope unseen DIFFUSION cells to **zero**:

```
unseen now (9):   CycleGAN DALLE2 GauGAN ProGAN SD14 SDXL StarGAN StyleGAN StyleGAN2
if all trained:   ProGAN only -- a GAN, out of scope. No diffusion left.
as shipped (3):   DALLE2, ProGAN, SDXL  -> in-scope diffusion preserved: DALLE2, SDXL
```

The score would have risen with no way to separate generalization from "we now
train on what we test" -- the data/ood trap, one step removed.
`HOLDOUT_GENERATORS` in `scripts/make_train_ext.py` keeps DALLE2 and SDXL
unseen. DALLE2 is the worst cell in the whole grid (clean 0.9278 -> degraded
0.8020), so if generator diversity transfers, **DALLE2 and SDXL must improve
WITHOUT being trained on.** That is the only result that says anything about the
competition's unknown generators. `--no-holdout` exists but destroys this.

Current `train_ext.csv`: 30,919 rows, 14 generators (6 new: CycleGAN, GauGAN,
SD14, StarGAN, StyleGAN, StyleGAN2), 17,078 real / 13,841 fake.

### Multi-machine: repo is on GitHub (private)

https://github.com/not-sekiun/tiktoktechjam2026 -- clone to EXACTLY the path in
`worker.py`'s `CANONICAL_ROOT`. `fingerprint_paths` hashes the manifests'
absolute path strings, so a different root makes every cache computed there
STALE on arrival. `scripts/worker.py --check` verifies path + data before any
GPU time is committed; `--job embed:train-ext` embeds only the 11 training views
rather than the default 22 (the other 11 are held-out EVAL views, never
evaluated on a train manifest -- the primary machine's full-pool run paid that
2x cost, secondary machines should not).

Manifests are committed to git as the reproducibility contract; images are not.
Copy `data/raw/tiny_genimage` (2.4 GB) and `data/train_ext` (545 MB), or re-pull
them -- the downloads are deterministic and land at identical paths.

### Process hazards hit twice this session, both silent

1. **`str.replace` on source silently no-ops when the anchor has moved.** The
   `only_generators` filter was declared, documented, and PRINTED ("keeping only
   generators: [...]") while never filtering, because its insertion anchored on
   a line an earlier audit had deleted. Use an anchored Edit, which fails loudly.
   Never accept a log line as evidence that the behaviour it describes ran.
2. **`nohup` inside a backgrounded call survives TaskStop.** The first train_ext
   pull kept running after being "stopped" and wrote contaminating generators
   alongside the fixed pull for ~20 minutes. Do not use `nohup` when the harness
   already backgrounds; verify the process is gone by PID.

Both were caught only by inspecting state rather than trusting a status line.
`data/train_ext/` was twice contaminated with `WhichFaceIsReal` -- the canary for
the FPR bug. Had it reached training, that detector would have gone green while
the bug remained.


### Job 1 (COMPLETE): full-pool embedding

`embed-views --backbone pe-core-l --manifest train --train-chains` with no
`--sample-rows`, so stem `train`, 23,800 rows x 22 views = 523,600 forwards,
~1.7h. When it finishes:

```
uv run main.py train-head-views --backbone pe-core-l --with-chains     --val-sample-rows 2000 --out models/pe-core-l__linear__fullpool.pt
```
(`--train-sample-rows` omitted so the stem is `train`.) Then score it on
`ood-s4000` and compare to **0.9366**. If it wins, this becomes the shipping
head and `predict.py`'s default should be repointed.

### Job 2 (COMPLETE): disjoint generator-diverse training slice

`data/train_ext/` -- a TRAINING slice from the same HF dataset as the eval
tier, kept **provably disjoint by construction**: streaming order is
deterministic (shuffle off), every filename encodes its stream position, the
eval tier occupies positions 1..8,400, and this pull uses
`skip_rows=8_400`. Disjointness is structural, not verified after the fact.

It keeps only the **9 generators absent from `data/raw/`** (CycleGAN, DALLE2,
GauGAN, ProGAN, SD14, SDXL, StarGAN, StyleGAN, StyleGAN2) plus Real. Two
reasons: those generators are the entire point, and the SHARED generators are
the ones at risk of overlapping upstream with Tiny-GenImage -- both datasets
draw on GenImage -- which would leak training data into `val`.

**Our training pool has 7 generators; this adds 9 unseen ones.** Given the
competition's test generators are unknown, generator diversity is plausibly
worth more than any remaining modelling change.

**How to use it WITHOUT invalidating anything:** build a union manifest at a
NEW path (e.g. `data/processed/train_ext.csv` = `train.csv` rows + the new
index) and embed it under its own stem. Do **not** drop these images into
`data/raw/` and re-run `main.py split` -- that rewrites `train.csv`/`val.csv`
in place, changes their fingerprints, and invalidates every existing cache
(opens E3). The whole point of a new stem is that nothing already computed
becomes stale.

Residual caveat to state in any write-up: the new slice's **reals** could
still overlap upstream with Tiny-GenImage reals, which would inflate `val`
but not `ood` (ood is disjoint by construction). If val rises much more than
ood after adding this data, suspect exactly that.

### Do NOT train on `data/ood/`

It is the only benchmark that still discriminates (0 of 18 views >= 0.99,
against 11 on val and 16 on demo-val). Training on it destroys the one
instrument that can measure anything, and repeats the error the project
already avoided with demo-val. `data/train_ext/` is the legitimate way to get
the same generator diversity into training.

### metaclip2-giant: registered, legal, and BLOCKED on a code change

My earlier rejection of Giant was based on an estimate and was wrong on the
facts. Measured:

```
vision-tower params = 1,843,564,416   -> 92.2% of the 2B cap, LEGAL
forward OK, pooled_dim 1664 at 378px, VRAM 7.38 GB allocated of 12.9
```

It is registered as `metaclip2-giant` and the loader dispatch now keys on the
MetaCLIP2 *family* rather than the exact name.

**But it cannot currently be run, and the blocker is not time.** Benchmarked
in isolation it does 13.2 img/s at a raw forward batch of 8, and collapses to
1.4 img/s at 16 -- despite peak memory staying under the limit. `embed_views`
flattens each batch to `batch_size * n_views`, so even `--batch-size 1` sends
**18 images per forward**, which is past that cliff. Measured end to end:
**1.33 img/s**, i.e. ~50 hours for the race protocol, not the ~5 I first
estimated.

**To run Giant you must first add micro-batching to
`embed_views.precompute_view_embeddings`**: chunk the flattened `b*v` tensor
into pieces of <= 8 and concatenate the results. Without that it is not slow,
it is infeasible.

Priority: **last**. It needs code, costs the most, its own family already lost
on all 18 views, and the data lever above is a larger expected gain for a
third of the compute. Licence is `cc-by-nc-4.0` (non-commercial) -- the user
judged that acceptable, but confirm before shipping it if it ever wins.

### Revised priority order

| # | Work | Cost | Status / expected value |
|---|---|---|---|
| 1 | Full-pool retrain | done | **+0.0041** (predicted +0.015-0.025; see above) |
| 2 | Deliverable 5.5.5 error analysis | done | `main.py error-analysis`, verified |
| 3 | **Real-domain corpus (photography)** | DONE | sid_real + 4,000 Unsplash -> `photoreal.pt`, now the shipping head. WildRF FPR@0.5 **.330 -> .183** at TPR .993, and OOD improved too. |
| 4 | Train on `train_ext` (6 new generators) | manifest built, embed pending | unknown; judge on held-out DALLE2 + SDXL |
| 5 | metaclip2-giant | micro-batching + ~5h | prior says it loses by ~0.02 |

**On (3), as actually done:** the face-source problem was moot -- there was no
portrait bug (FINDINGS 2h-CORRECTION). What did work was ordinary domain
coverage of the REAL class: `sid-real` (SID_Set's OpenImages reals) plus 4,000
Unsplash photographs, concatenated via `--extra-train-manifest` so each keeps its
own cache stem. Shipping head:

```
uv run main.py train-head-views --backbone pe-core-l --with-chains \
  --val-sample-rows 2000 --extra-train-manifest sid-real unsplash-real \
  --balance --out models/pe-core-l__linear__photoreal.pt
```

**Two data hazards this cost us; check both before adding any real corpus.**

1. *Verify the pixels, not the name.* A "pexels" mirror
   (`cj-mills/pexels-110k-768p-min-jpg-depth-anything-large-hf`) ships Depth
   Anything OUTPUTS -- single-channel depth maps named after photos it does not
   contain. `img.convert("RGB")` widens them to three identical channels
   silently, and they cleared the 384px floor at 768p. 4,000 trained as REAL.
   `scripts/audit_corpora.py` now fingerprints every corpus (saturation,
   bytes/px); real photography sits at 0.30-0.36 saturation, the depth maps at
   0.000. Run it before training on anything new.
2. *Compare at matched TPR, not at threshold 0.5.* Those depth maps made
   FPR@0.5 look BETTER (.183 -> .130) while making the ranking worse at every
   matched operating point (.051 -> .070 at TPR=.98). A threshold-0.5 comparison
   would have shipped them.


Read order: this file, then `NARRATIVE.md`'s "Comparability epochs" table
(short, and it will stop you comparing numbers that are not comparable). Only
open `FINDINGS.md` if you need the forensic detail behind a specific claim — it
is long.

---

## 1. Where the race results are

```
reports/race/race_status.json     <- MACHINE-READABLE. Start here.
reports/race/race_console.log     <- full console output of the whole race
reports/race/<backbone>/run.log   <- per-backbone log
reports/race/<backbone>/grid_val.csv
reports/race/<backbone>/grid_ood.csv
models/<backbone>__linear__race.pt
```

`race_status.json` is written **incrementally after every stage**, so it is
valid even if the race was interrupted. Check each backbone's `"status"`:

- `"done"` — complete and trustworthy.
- `"FAILED"` — see its `"error"` (a truncated traceback). One backbone failing
  does not invalidate the others; the runner isolates them.
- **absent, or present without `"status"`** — that backbone was still running
  when the race stopped. **Its numbers are incomplete: do not use them.** Re-run
  with `uv run python scripts/run_race.py --backbones <key>`.

Per backbone, per eval set (`val`, `ood`), the JSON holds `auc_clean`,
`auc_robust` (pooled / mean / worst), `score_pooled`, `score_worst`,
`worst_view`, `threshold`, and `per_view` (all 18 view AUCs).

## 2. The one decision to make

**Which backbone ships.** The incumbent is `pe-core-l`; `models/pe-core-l__linear__augchain.pt`
is what `predict.py` loads by default. The challengers are `metaclip2-h` and
`dinov3-l`.

### Judge on `ood`, not `val`

This matters more than anything else in this file. Under the shipping head:

| eval set | views >= 0.99 | range |
|---|---|---|
| val-s2000 | **11 / 18** | 0.9121 - 0.9988 |
| demo_val-s2000 | **16 / 18** | 0.9414 - 1.0000 |
| **ood-s4000** | **0 / 18** | **0.8099 - 0.9532** |

val and demo-val are saturated — differences there sit inside their own
standard error. Concretely: a seed change alone was enough to flip val's
"worst view" between `chain_heavy` (0.9121) and `noise_sigma0.1`, which
differed by 0.0003 against a per-view SE of ~0.0065. **A benchmark that cannot
rank its own two worst cells cannot rank two backbones.** `ood-s4000` is the
only tier with room.

### Decision rule (agreed before the numbers were seen, deliberately)

Primary metric: **`ood.score_pooled`** = `0.5*AUC_clean + 0.5*AUC_robust(pooled)`.

| Gap vs `pe-core-l` | Action |
|---|---|
| challenger **> +0.010** | **Switch.** Retrain, repoint `predict.py`'s default, update AGENTS/NARRATIVE. |
| **-0.005 to +0.010** | **Keep `pe-core-l`.** It is validated end to end and already wired into the deliverable. A gap this size is not worth the churn. |
| challenger **< -0.005** | Keep `pe-core-l`, and record the result — a confirmed negative is worth keeping. |

Tie-breakers, in order, if the primary lands in the middle band:
1. `ood` **diffusion-family mean** (the `MEAN diffusion` row from
   `eval-grid --by-generator`). The competition is expected to use diffusion
   generators, so this is the in-scope number.
2. `ood` `chain_heavy` AUC — the binding constraint (0.8099 for `pe-core-l`).
3. `ood.score_worst` — the adversarial reading.

**Do not tune on the GAN rows.** They are reported but out of scope. Note they
did *not* collapse — GANs scored *higher* than diffusion (`MEAN gan` clean
0.9637 vs `MEAN diffusion` 0.9466) — so there is no GAN weakness to fix.

### Precision caveat

`ood` is n=4000 (2000 real / 2000 AIGC), so per-view SE is roughly 0.005-0.008.
The comparison is **paired** (every backbone saw byte-identical rows and
identical per-image transform seeds), which resolves smaller differences than
that suggests — but treat anything under ~0.005 as noise.

### The prior, from the paper

`arXiv:2602.01738`, numbers pulled from its actual tables (FINDINGS 2g):
in-the-wild average **DINOv3 0.940 > PE-CLIP 0.899 > MetaCLIP2 0.842**; but
under blur sigma=2.0 **MetaCLIP2 0.932 (improves) > DINOv3 0.891 > PE-CLIP
0.778 (collapses)**. Two caveats: the paper's DINOv3 is **ViT-7B**, which
violates the <2B rule, so our `dinov3-l` (ViT-L, 303M) is a different model
sharing a family name; and its MetaCLIP2 is *Giant* (~1.8-1.9B vision tower,
~95% of the cap — rejected) while ours is *Huge* (630M). **Only `pe-core-l`
matches the paper exactly.** Weight the paper as family-level evidence, not as
a prediction about these checkpoints.

## 3. State of the deliverables

| Deliverable | Status |
|---|---|
| 5.5.2 inference script | **DONE** — `predict.py --input_dir <dir> --output preds.json`, verified 10/10 on held-out rows |
| 5.5.4 robustness summary | **DONE** — `main.py eval-grid`, 18 views, `reports/grid__*.csv` |
| 5.5.5 error analysis | **NOT STARTED** — the last required piece |
| Backbone race | **DONE** — `pe-core-l` wins; see the top of this file and NARRATIVE Run 7 |
| Full-pool retrain | running / see section 0 |
| `data/train_ext/` slice | running / see section 0 |
| `metaclip2-giant` | registered + legal, BLOCKED on micro-batching — section 0 |

**5.5.5 is now the highest-value remaining work.** The
raw material already exists: `reports/race/*/grid_ood.csv` for per-view and
per-generator failures, and `predict.py` to pull concrete false
positives/negatives. The strongest findings to build it on:
- `DALLE2` is the one real in-scope failure: clean 0.9278 -> degraded **0.8020**,
  a 12.6-point collapse, three times worse than any other generator's. Unseen,
  diffusion, in scope.
- Composition is the binding axis and decays monotonically with depth:
  penalty (chain AUC minus its own weakest component) `chain_light` -0.0122,
  `chain_medium` -0.0500, `chain_heavy` -0.0760.
- Unseen **diffusion** generators generalize well — SD14 0.9579 and SDXL 0.9554
  beat trained ADM 0.9208 and Midjourney 0.9296. That is the frozen-backbone
  thesis working, and it is a result worth writing up.

## 4. Things that will bite you

1. **Never cross-compare comparability epochs.** `NARRATIVE.md` has the table.
   Everything current is E2 (commit `abc28a7` onward). Numbers from before that
   are invalid or superseded and are labelled as such.
2. **Runs 4-6 in NARRATIVE predate training seeds** and carry ~+/-0.0005.
   Training is seeded now (`--seed`, default 42) and reproduces exactly;
   the canonical seeded shipping figure is val **0.9886**, not the 0.9892 that
   an earlier unseeded draw produced. Race numbers are all seeded.
3. **Every view cache read must go through `embed_views.load_view_cache`.** It
   verifies the view spec fingerprint and the manifest fingerprint. A cache is
   keyed by *filename*, which does not identify its contents — `main.py split`
   rewrites manifests in place, and editing a transform severity changes what a
   view name means. A stale cache produces confident wrong numbers, not errors.
4. **Prefer adding a NEW view name to redefining an existing one.** Per-view
   fingerprints mean a new name invalidates nothing, while redefining one opens
   a new comparability epoch and forces a full recompute.
5. **`config.IMAGE_SIZE` / `NORM_MEAN` / `NORM_STD` are ImageNet defaults and
   are WRONG for every registered backbone.** Always use the loaded module's
   `native_res` / `norm_mean` / `norm_std`.
6. **ASCII only in `print()`** — the Windows console mojibakes non-ASCII.
   Unicode is fine in Markdown and docstrings.
7. **Never train on `data/demo_val/` or `data/ood/`.** `make_splits.py` globs
   only `data/raw/`, which enforces this structurally.
8. **There is an unresolved noise-semantics inconsistency**, documented and
   deliberately not fixed: chain noise (`pil_noise`, applied at source
   resolution then attenuated ~2x by the resize) is a weaker perturbation than
   single-view noise (`noise`, applied at native_res full strength) at the same
   nominal sigma. Both are defensible readings of the 5.2 table, the spec
   strings distinguish them so fingerprints do not collide, and it is applied
   identically to every backbone so the race is fair. Fixing it would open E3
   for no score gain.

## 5. Useful commands

```bash
uv run main.py check-env
uv run python scripts/run_race.py --backbones <key>          # resume/rerun one
uv run main.py eval-grid --backbone <key> --manifest ood --sample-rows 4000 \
    --head models/<key>__linear__race.pt --by-generator
uv run python predict.py --input_dir <dir> --output preds.json
```

Reminder: this is a **uv** project. Never a bare `python`/`pip`. `data/` is
gitignored and large; check `main.py check-env` before assuming anything is on
disk.
