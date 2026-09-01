"""Race frozen backbones across the robustness grid on val and the OOD tier.

For each backbone, end to end:
  1. embed all 22 views of train-s6000 (18 scored + 4 training chains)
  2. embed the 18 scored views of val-s2000
  3. embed the 18 scored views of ood-s4000
  4. train a seeded head on clean + 6 degraded + 4 chained views
  5. score it on val-s2000 and ood-s4000, with the per-generator breakdown

WHY THE OOD TIER IS THE DECIDING ONE. val and demo-val are saturated: under the
shipping head, val has 11 of 18 grid views at or above 0.99 and demo-val has 16
of 18, so differences between backbones there sit inside their own standard
error. ood-s4000 has ZERO views above 0.99 and spans 0.8099-0.9532. It is the
only one of the three with room to separate candidates.

FAIRNESS. Every backbone faces byte-identical inputs: the same seeded
stratified row subsample (--sample-rows/--sample-seed), the same per-(image,
view) transform seeds (keyed on image path, so independent of row order), and
the same seeded head training. The only variable is the backbone.

RESULTS ARE WRITTEN INCREMENTALLY to reports/race/race_status.json after every
completed stage, and each stage's stdout is teed to reports/race/<backbone>/.
This is deliberate: an earlier long-running job in this project wrote its index
only at the end, so interrupting it would have discarded an hour of work. A
race that takes ~1.5 hours must survive being stopped.

Usage:
    uv run python scripts/run_race.py
    uv run python scripts/run_race.py --backbones metaclip2-h dinov3-l
    uv run python scripts/run_race.py --skip-embed     # re-score only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from datetime import UTC, datetime, timezone

from aigc_detect.config import (
    OOD_MANIFEST,
    RANDOM_SEED,
    ROOT_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)

RACE_DIR = ROOT_DIR / "reports" / "race"
STATUS_JSON = RACE_DIR / "race_status.json"

# (manifest path, sample_rows, include_train_chains)
EVAL_SETS = {
    "val": (VAL_MANIFEST, 2000, False),
    "ood": (OOD_MANIFEST, 4000, False),
}
TRAIN_SET = (TRAIN_MANIFEST, 6000, True)

DEFAULT_BACKBONES = ("pe-core-l", "metaclip2-h", "dinov3-l")


class _Tee(io.TextIOBase):
    """Write to both the real stdout and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _load_status() -> dict:
    if STATUS_JSON.exists():
        try:
            return json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt status file must not kill the race
            pass
    return {"started": datetime.now(UTC).isoformat(), "backbones": {}}


def _save_status(status: dict) -> None:
    RACE_DIR.mkdir(parents=True, exist_ok=True)
    status["updated"] = datetime.now(UTC).isoformat()
    STATUS_JSON.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def run_backbone(key: str, status: dict, skip_embed: bool = False) -> None:
    from aigc_detect.embed.views import cache_stem, precompute_view_embeddings
    from aigc_detect.evaluation.grid import evaluate_grid
    from aigc_detect.train.probe import TRAIN_VIEWS_WITH_CHAINS, train_head_on_views

    out_dir = RACE_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = status["backbones"].setdefault(key, {})
    t_start = time.time()

    log_path = out_dir / "run.log"
    with open(log_path, "a", encoding="utf-8") as log_fh:
        tee = _Tee(sys.__stdout__, log_fh)
        with contextlib.redirect_stdout(tee):
            print(f"\n{'=' * 70}\n[race] {key} starting {datetime.now(UTC).isoformat()}\n{'=' * 70}")

            if not skip_embed:
                for tag, (manifest, rows, chains) in {
                    "train": TRAIN_SET,
                    **{k: v for k, v in EVAL_SETS.items()},
                }.items():
                    t0 = time.time()
                    print(f"\n[race] {key}: embedding {tag} (sample_rows={rows}, train_chains={chains})")
                    precompute_view_embeddings(
                        manifest_path=manifest,
                        backbone_key=key,
                        batch_size=8,
                        num_workers=4,
                        sample_rows=rows,
                        include_train_chains=chains,
                    )
                    entry[f"embed_{tag}_sec"] = round(time.time() - t0, 1)
                    _save_status(status)

            print(f"\n[race] {key}: training seeded head on clean + degraded + chained views")
            head_path = ROOT_DIR / "models" / f"{key}__linear__race.pt"
            train_head_on_views(
                backbone_key=key,
                train_stem=cache_stem(TRAIN_MANIFEST, sample_rows=TRAIN_SET[1]),
                val_stem=cache_stem(VAL_MANIFEST, sample_rows=EVAL_SETS["val"][1]),
                train_views=TRAIN_VIEWS_WITH_CHAINS,
                out_path=head_path,
                train_manifest=TRAIN_MANIFEST,
                train_sample_rows=TRAIN_SET[1],
                val_manifest=VAL_MANIFEST,
                val_sample_rows=EVAL_SETS["val"][1],
                seed=RANDOM_SEED,
            )
            entry["head"] = str(head_path)
            _save_status(status)

            for tag, (manifest, rows, _c) in EVAL_SETS.items():
                print(f"\n[race] {key}: scoring {tag}")
                res = evaluate_grid(
                    backbone_key=key,
                    manifest_path=manifest,
                    head_path=head_path,
                    sample_rows=rows,
                    by_generator=True,
                    out_csv=out_dir / f"grid_{tag}.csv",
                )
                pooled = res["auc_robust"]["pooled"]
                entry[tag] = {
                    "auc_clean": res["auc_clean"],
                    "auc_robust": res["auc_robust"],
                    "score_pooled": 0.5 * res["auc_clean"] + 0.5 * pooled,
                    "score_worst": 0.5 * res["auc_clean"] + 0.5 * res["auc_robust"]["worst"],
                    "worst_view": res["worst_view"],
                    "threshold": res["threshold"],
                    "per_view": {r["view"]: r["auc"] for r in res["rows"]},
                    "csv": str(out_dir / f"grid_{tag}.csv"),
                }
                _save_status(status)

            entry["total_sec"] = round(time.time() - t_start, 1)
            entry["status"] = "done"
            _save_status(status)
            print(f"\n[race] {key} DONE in {entry['total_sec']:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    p.add_argument("--skip-embed", action="store_true", help="Assume caches exist; only train and score.")
    a = p.parse_args()

    RACE_DIR.mkdir(parents=True, exist_ok=True)
    status = _load_status()
    status["backbone_order"] = a.backbones

    for key in a.backbones:
        try:
            run_backbone(key, status, skip_embed=a.skip_embed)
        except Exception:  # noqa: BLE001 - one backbone failing must not abort the race
            status["backbones"].setdefault(key, {})["status"] = "FAILED"
            status["backbones"][key]["error"] = traceback.format_exc()[-4000:]
            _save_status(status)
            print(f"\n[race] {key} FAILED -- continuing with the rest\n{traceback.format_exc()}")

    status["finished"] = datetime.now(UTC).isoformat()
    _save_status(status)
    print(f"\n[race] all done -> {STATUS_JSON}")


if __name__ == "__main__":
    main()
