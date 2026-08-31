# Backbone race results

Machine-readable results: **`race_status.json`** — start there.

Written incrementally after every stage, so it is valid even if the race was
interrupted. Per backbone check `"status"`:

- `"done"`     complete, trustworthy
- `"FAILED"`   see `"error"`; other backbones are unaffected
- missing      still running when the race stopped — **numbers incomplete, do not use**

Per backbone, per eval set (`val`, `ood`): `auc_clean`, `auc_robust`
(pooled/mean/worst), `score_pooled`, `score_worst`, `worst_view`, `threshold`,
and `per_view` (all 18 view AUCs).

**Judge on `ood`, not `val`.** val is saturated (11 of 18 views >= 0.99);
ood-s4000 has none and spans 0.8099-0.9532. The decision rule and tie-breakers
are in `../../HANDOFF.md` section 2.

Fairness: every backbone saw byte-identical rows (same seeded stratified
subsample), identical per-(image, view) transform seeds, and identical seeded
head training. The only variable is the backbone.

Re-run one: `uv run python scripts/run_race.py --backbones <key>`
