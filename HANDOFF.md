# Handoff — read the race results, then decide

**Written 2026-08-29.** A backbone race was launched and is (or was) running in
the background when this was written. **Your first job is to read its results
and make one decision.** Everything you need is below; you should not have to
re-derive anything.

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
| Backbone race | running / see `race_status.json` |

**After the race decision, 5.5.5 is the highest-value remaining work.** The
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
