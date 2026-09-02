"""Run a named data-or-embedding job on a SECONDARY machine, then ship the
embeddings back.

    uv run python scripts/worker.py --check
    uv run python scripts/worker.py --job data:tiny-genimage
    uv run python scripts/worker.py --job embed:train-ext

WHY THIS EXISTS. Embedding is the only expensive step (523k forward passes for
the full pool) and it parallelises perfectly across machines: each (backbone,
manifest, view) cache is independent. Head training and scoring are seconds and
stay on one machine.

THE HARD CONSTRAINT, AND WHAT IS LEFT OF IT.

This script used to open with "THE REPO PATH MUST MATCH", and it meant it: the
cache key was a hash of the manifest's absolute `image_path` STRINGS, so a run
under `D:\\work\\tiktok...` produced vectors that `load_view_cache` correctly
rejected as STALE the moment you copied them home. Hours of GPU, thrown away,
with nothing warning you until it was done.

Tier 4 removed that. An embedding is now keyed on the image's CONTENT, so
vectors computed anywhere are valid everywhere and combine with `cache merge`.

What still ties a worker to one path is narrower and not about the cache: the
committed manifests name their images by absolute path, so a differing root
means the images cannot be FOUND, not that the work would be wasted. Tier 5
switches manifests to relative paths and this check goes away with it. Until
then, clone to the identical absolute path; `--check` verifies it.

WHY MANIFESTS ARE COMMITTED TO GIT RATHER THAN REBUILT. `main.py split` is
seeded and deterministic, but only if it is invoked with the same flags over the
same set of raw indexes -- and getting that subtly wrong produces a manifest
that looks fine and fingerprints differently, wasting the entire run. The CSVs
are a few MB, so they ship as the contract and secondary machines never run
`split` at all. They download IMAGES to the paths the manifests already name.

The image downloads are themselves deterministic: Tiny-GenImage is indexed by
split position (`train_000123.jpg`), and the streamed slices are taken in stream
order with shuffling off and the position baked into the filename
(`ood_0001234.jpg`). Same commands, same files, same paths.

SHIPPING RESULTS BACK: copy this machine's cache root (the `embeddings/` store
and `hashes.sqlite`) and fold it in on the primary with

    uv run aigc cache merge /path/to/the/workers/cache

Only rows the primary does not already hold are copied, so a partial worker run
is still worth shipping and re-shipping the same store twice costs nothing. The
primary regenerates its `data/embeddings/*.npz` from the merged store with a
plain `embed-views` re-run, which does no forward passes.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from aigc_detect.config import (
    DATA_DIR,
    OOD_MANIFEST,
    ROOT_DIR,
    TRAIN_EXT_MANIFEST,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)

# The canonical location. Every machine must clone here or fingerprints diverge.
CANONICAL_ROOT = Path(r"C:\Users\angus\Desktop\tiktoktechjam2026")

# Exact parameters used for the streamed slices on the primary machine. Changing
# any of these changes which images land on disk, so they are constants, not
# defaults to be re-chosen per machine.
OOD_PULL = dict(per_generator=250, max_scan=30_000, min_scan=2_000, skip_rows=0)
TRAIN_EXT_PULL = dict(per_generator=400, max_scan=60_000, min_scan=0, skip_rows=8_400)

JOBS = {
    "data:tiny-genimage": "Download the Tiny-GenImage training pool + heldout split.",
    "data:ood": "Stream the OOD evaluation tier (eval only, never trained on).",
    "data:train-ext": "Stream the disjoint generator-diverse training slice.",
    "embed:train": "Embed the 11 training views of the FULL train pool (23,800 rows).",
    "embed:train-ext": "Embed the 11 training views of the train-ext union manifest (31,567 rows).",
    "embed:ood": "Embed the 18 scored views of ood-s4000.",
    "embed:val": "Embed the 18 scored views of val-s2000.",
}


def check_root() -> bool:
    ok = ROOT_DIR == CANONICAL_ROOT
    print(f"[worker] repo root : {ROOT_DIR}")
    print(f"[worker] canonical : {CANONICAL_ROOT}")
    if ok:
        print("[worker] PATH OK -- the manifests will resolve to images on this machine.")
    else:
        print("[worker] PATH MISMATCH. The committed manifests name their images by ABSOLUTE")
        print("[worker] path, so under this root they point at files that are not there.")
        print("[worker] (The embeddings themselves no longer care -- they are keyed on image")
        print("[worker]  content since Tier 4, and merge across machines. This is about")
        print("[worker]  finding the images at all, and Tier 5's relative paths retire it.)")
        print(f"[worker] Re-clone to exactly: {CANONICAL_ROOT}")
    return ok


def check_data() -> None:
    print("\n[worker] manifest -> image availability")
    for name, mf in [("train", TRAIN_MANIFEST), ("val", VAL_MANIFEST),
                     ("ood", OOD_MANIFEST), ("train-ext", TRAIN_EXT_MANIFEST)]:
        if not mf.exists():
            print(f"  {name:<10} manifest MISSING ({mf.name})")
            continue
        paths = pd.read_csv(mf, usecols=["image_path"])["image_path"]
        sample = paths.sample(min(400, len(paths)), random_state=0)
        missing = sum(1 for p in sample if not Path(str(p)).exists())
        pct = 100 * (1 - missing / len(sample))
        state = "OK" if missing == 0 else f"{missing}/{len(sample)} sampled paths MISSING"
        print(f"  {name:<10} {len(paths):>7,} rows   images present ~{pct:5.1f}%   {state}")


def _run(cmd: list[str]) -> None:
    print(f"\n[worker] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT_DIR)
    if r.returncode != 0:
        raise SystemExit(f"[worker] command failed with exit {r.returncode}")


def run_job(job: str) -> None:
    if not check_root():
        raise SystemExit("[worker] refusing to run: repo path does not match the canonical root.")

    if job == "data:tiny-genimage":
        _run(["uv", "run", "main.py", "download", "tiny-genimage"])
        _run(["uv", "run", "main.py", "build-heldout"])
    elif job in ("data:ood", "data:train-ext"):
        from aigc_detect.config import GENERATOR_FAMILY, TRAIN_GENERATORS
        from scripts.download_ood_benchmark import download_ood_benchmark

        if job == "data:ood":
            download_ood_benchmark(**OOD_PULL)
            _run(["uv", "run", "main.py", "build-ood"])
        else:
            unseen = tuple(sorted(g for g in GENERATOR_FAMILY
                                  if g not in TRAIN_GENERATORS and GENERATOR_FAMILY[g] != "real"))
            out = DATA_DIR / "train_ext"
            download_ood_benchmark(
                out_dir=out, index_path=out / "train_ext_index.csv",
                source_name="aigc_bench_ext", only_generators=(*unseen, "Real"),
                **TRAIN_EXT_PULL,
            )
    elif job.startswith("embed:"):
        from aigc_detect.train.probe import TRAIN_VIEWS_WITH_CHAINS

        target = job.split(":", 1)[1]
        # A TRAINING manifest only needs the 11 views the head actually trains
        # on. The default grid is 22 (18 scored + 4 training chains), and the
        # other 11 are held-out EVALUATION views -- which are never evaluated on
        # a train manifest, so computing them there is half the GPU time for
        # nothing. (The full-pool run on the primary machine was launched
        # without this and paid that cost; don't repeat it here.)
        # If a future ablation needs the held-out severities on a train
        # manifest (as the augmentation ceiling probe did), run main.py
        # embed-views directly with --train-chains and no --views.
        train_views = ["--views", *TRAIN_VIEWS_WITH_CHAINS]
        args = {
            "train": ["--manifest", "train", *train_views],
            "train-ext": ["--manifest", "train-ext", *train_views],
            "ood": ["--manifest", "ood", "--sample-rows", "4000"],
            "val": ["--manifest", "val", "--sample-rows", "2000"],
        }[target]
        _run(["uv", "run", "main.py", "embed-views", "--backbone", "pe-core-l",
              "--batch-size", "8", "--num-workers", "4", *args])
        print("\n[worker] DONE. Copy these back to the primary machine's data/embeddings/:")
        stem = {"train": "train", "train-ext": "train_ext", "ood": "ood-s4000", "val": "val-s2000"}[target]
        print(f"  data/embeddings/pe-core-l__{stem}__*.npz")
    else:
        raise SystemExit(f"[worker] unknown job '{job}'. Known: {list(JOBS)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=list(JOBS), help="Job to run.")
    ap.add_argument("--check", action="store_true", help="Report path + data state and exit.")
    a = ap.parse_args()

    if a.check or not a.job:
        check_root()
        check_data()
        print("\n[worker] jobs:")
        for k, v in JOBS.items():
            print(f"  {k:<22} {v}")
        return
    run_job(a.job)


if __name__ == "__main__":
    main()
