"""Download a capped, generator-balanced slice of AIGC-Detection-Benchmark into
data/ood/ -- a deliberately HARD out-of-distribution evaluation tier.

TheKernel01/AIGC-Detection-Benchmark (HuggingFace, ungated, Apache-2.0):
  split:      test only, 125,026 images, 32GB total
  features:   image (PIL), label (0=real, 1=fake), generator (18 classes)
  generators: Real, ADM, BigGAN, CycleGAN, DALLE2, GauGAN, GLIDE, Midjourney,
              ProGAN, SD14, SD15, SDXL, StarGAN, StyleGAN, StyleGAN2, VQDM,
              WhichFaceIsReal, Wukong

WHY THIS TIER EXISTS. The project's first three evaluation sets stopped being
able to tell models apart. Under the Run 6 head, demo-val has 16 of 18
robustness-grid views at or above 0.99 (11 at or above 0.999) and val has 11 of
18. A backbone race on those is a comparison inside its own standard error --
see NARRATIVE.md's headroom analysis. This tier is chosen to have room.

WHAT MAKES IT HARD. Ten of the eighteen generator classes are absent from our
training pool (which has only Real, ADM, BigGAN, GLIDE, Midjourney, SD15, VQDM,
Wukong): CycleGAN, DALLE2, GauGAN, ProGAN, SD14, SDXL, StarGAN, StyleGAN,
StyleGAN2, WhichFaceIsReal. Five are **GAN** families, architecturally unlike
the mostly-diffusion generators the head has seen. This is the unseen-generator
evaluation FINDINGS has had open since the Tiny-GenImage ingest and which has
never actually been run.

STREAMING + QUOTAS. 32GB is far more than we need, so the test split is
streamed and each generator gets a quota. Three details matter:

  1. **Shuffling is OFF by default (`--shuffle-buffer 0`), deliberately.** The
     obvious guard against a generator-ordered parquet is `.shuffle(seed,
     buffer_size)`, which in streaming mode also shuffles shard order. It was
     tried and is actively harmful here: a streaming shuffle must FILL its
     buffer before yielding a single example, so `buffer_size=10_000` blocked
     for over seven minutes having emitted nothing. It is also unnecessary --
     probing the raw stream showed shard 0 already interleaves generators
     (`Real, ADM, Real, ...`), which is the exact risk shuffle was guarding
     against. Set `--shuffle-buffer N` to re-enable it if a future revision of
     the dataset turns out to be generator-blocked.
     Cost of leaving it off: within each generator we take the first N in file
     order rather than a random N. Acceptable for an eval tier; noted here so
     nobody reads generator-internal ordering as random.
  2. Reals get a global quota equal to the sum of the fake quotas, so the tier
     lands roughly label-balanced. `stratified_sample` downstream balances by
     (label, source) and this tier is a single source, so generator balance has
     to be established HERE -- it cannot be recovered later.
  3. A `--min-scan` floor prevents early-stopping before the rarer generators
     have had a fair chance to appear. Without it the run could stop having
     never seen the unseen-generator classes this tier exists to test. The
     default is small because the stream was MEASURED to be near-perfectly
     round-robin: the first 837 rows contained all 18 generator classes at
     25 each (plus 419 reals). Raise it if that ever stops holding.

MEASURED THROUGHPUT: ~8.4 images/sec from this network. Filling 250 per fake
generator needs roughly 8,400 rows scanned, i.e. ~17 minutes -- the quotas, not
the 32GB total, set the cost.

Every image is re-encoded as JPEG quality 95, exactly as Tiny-GenImage is.
This is not cosmetic: leaving source formats alone reopens the compression
shortcut (real photos are usually native JPEG, AI images usually native PNG),
where a detector scores ~99% by reading compression history and learning
nothing. Both tiers must be re-encoded the same way or a cross-tier comparison
measures format, not content.

EVALUATION ONLY. data/ood/ is a fourth tier alongside raw/ (training pool),
heldout/ and demo_val/. scripts/make_splits.py globs ONLY data/raw/, so this
can never leak into training.

Usage:
    uv run main.py download-ood [--per-generator N] [--max-scan N] [--force]
    uv run main.py build-ood
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aigc_detect.config import (  # noqa: E402
    GENERATOR_FAMILY,
    LABEL_AIGC,
    LABEL_REAL,
    OOD_DIR,
    TRAIN_GENERATORS,
)

OOD_HF_HANDLE = "TheKernel01/AIGC-Detection-Benchmark"
OOD_INDEX = OOD_DIR / "aigc_detect_bench_index.csv"
SOURCE_NAME = "aigc_detect_bench"


def reindex_from_disk() -> Path:
    """Rebuild the index CSV from the images already written under data/ood/images/.

    Exists because the streaming download only writes its index after the loop
    ends, so interrupting it discards every image already saved. That is a bad
    property for a job that streams for an hour, and it bit us: one generator
    (ProGAN) turned out to be rarer in the stream than the round-robin pattern
    suggested, so its quota never filled and the run kept scanning long after
    every in-scope generator was complete.

    Recovery is exact rather than approximate: the directory name IS the
    generator, and the label follows from the generator (only Real and
    WhichFaceIsReal are real sources), so nothing is inferred from the pixels.
    """
    root = OOD_DIR / "images"
    if not root.exists():
        raise SystemExit(f"No images under {root}. Run `uv run main.py download-ood` first.")

    real_gens = {g for g, fam in GENERATOR_FAMILY.items() if fam == "real"}
    records: list[tuple[str, int, str]] = []
    for gen_dir in sorted(root.iterdir()):
        if not gen_dir.is_dir():
            continue
        generator = gen_dir.name
        if generator not in GENERATOR_FAMILY:
            print(f"[ood] WARNING: unknown generator dir '{generator}' -- included, family unknown")
        label = LABEL_REAL if generator in real_gens else LABEL_AIGC
        for img in sorted(gen_dir.glob("*.jpg")):
            records.append((str(img.resolve()), label, generator))

    if not records:
        raise SystemExit(f"No .jpg files found under {root}.")

    OOD_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OOD_INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "source", "generator"])
        for image_path, label, generator in records:
            w.writerow([image_path, label, SOURCE_NAME, generator])

    n_real = sum(1 for _, l, _ in records if l == LABEL_REAL)
    print(f"[ood] reindexed {len(records):,} images from disk ({n_real} real / {len(records)-n_real} aigc)")
    print(f"[ood] per-generator: {dict(sorted(Counter(g for _, _, g in records).items()))}")
    print(f"[ood] UNSEEN generators (absent from data/raw/): "
          f"{sorted({g for _, _, g in records} - set(TRAIN_GENERATORS))}")
    print(f"[ood] wrote index -> {OOD_INDEX}")
    return OOD_INDEX


def download_ood_benchmark(
    per_generator: int = 200,
    max_scan: int | None = 60_000,
    min_scan: int = 2_000,
    shuffle_buffer: int = 0,
    seed: int = 42,
    force: bool = False,
) -> Path:
    from datasets import load_dataset
    from tqdm import tqdm

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ood] streaming '{OOD_HF_HANDLE}' split='test' (32GB total; quotas cap what we keep)")

    ds = load_dataset(OOD_HF_HANDLE, split="test", streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    feats = getattr(ds, "features", None) or {}
    gen_names = feats["generator"].names if "generator" in feats and hasattr(feats["generator"], "names") else None
    lab_names = feats["label"].names if "label" in feats and hasattr(feats["label"], "names") else None
    print(f"[ood] label names={lab_names} generator names={gen_names}")

    def gen_of(ex):
        g = ex["generator"]
        return gen_names[g] if (gen_names is not None and isinstance(g, int)) else str(g)

    label_map = {0: LABEL_REAL, 1: LABEL_AIGC}
    fake_quota: Counter = Counter()
    real_kept = 0
    real_cap = per_generator * 16  # ~16 fake classes; reals capped to match the fake total
    records: list[tuple[str, int, str]] = []
    scanned = 0

    bar = tqdm(desc="[ood] scanning", unit="img")
    for ex in ds:
        scanned += 1
        bar.update(1)
        if max_scan is not None and scanned > max_scan:
            print(f"\n[ood] hit --max-scan {max_scan}; stopping with quotas partially filled")
            break

        raw_label = ex["label"]
        norm_label = label_map[int(raw_label)]
        generator = gen_of(ex)

        if norm_label == LABEL_REAL:
            if real_kept >= real_cap:
                continue
            real_kept += 1
        else:
            if fake_quota[generator] >= per_generator:
                continue
            fake_quota[generator] += 1

        gen_dir = OOD_DIR / "images" / generator
        gen_dir.mkdir(parents=True, exist_ok=True)
        img_path = gen_dir / f"ood_{scanned:07d}.jpg"
        if force or not img_path.exists():
            ex["image"].convert("RGB").save(img_path, format="JPEG", quality=95)
        records.append((str(img_path.resolve()), norm_label, generator))

        # Stop early once reals are full and every generator we have actually
        # observed as fake is at quota -- avoids streaming the remaining tens of
        # GB for nothing. The scanned>=min_scan guard matters: without it we
        # could stop after a few thousand rows having never encountered the
        # rarer generators at all, and silently ship a tier missing exactly the
        # unseen-generator classes this tier exists to test.
        if (
            scanned >= min_scan
            and real_kept >= real_cap
            and fake_quota
            and all(v >= per_generator for v in fake_quota.values())
        ):
            print(f"\n[ood] all quotas filled after {scanned:,} scanned "
                  f"({len(fake_quota)} fake generators)")
            break
    bar.close()

    if not records:
        raise RuntimeError(
            f"No images kept from {OOD_HF_HANDLE}. The schema may have changed -- inspect "
            "the stream and adjust scripts/download_ood_benchmark.py."
        )

    OOD_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OOD_INDEX, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "source", "generator"])
        for image_path, label, generator in records:
            w.writerow([image_path, label, SOURCE_NAME, generator])

    n_real = sum(1 for _, l, _ in records if l == LABEL_REAL)
    print(f"\n[ood] scanned {scanned:,}, kept {len(records):,} ({n_real} real / {len(records)-n_real} aigc)")
    print(f"[ood] per-generator kept: {dict(sorted(Counter(g for _, _, g in records).items()))}")
    unseen = sorted({g for _, _, g in records} - TRAIN_GENERATORS)
    print(f"[ood] UNSEEN generators (absent from data/raw/): {unseen}")
    print(f"[ood] wrote index -> {OOD_INDEX}")
    return OOD_INDEX


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-generator", type=int, default=200, help="Max images kept per fake generator.")
    p.add_argument("--max-scan", type=int, default=60_000, help="Stop after streaming this many rows.")
    p.add_argument("--min-scan", type=int, default=2_000,
                   help="Never early-stop before this many rows, so rare generators get a fair chance.")
    p.add_argument("--shuffle-buffer", type=int, default=0,
                   help="0 (default) disables streaming shuffle -- it blocks until its buffer fills.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Re-encode images already on disk.")
    p.add_argument("--reindex-only", action="store_true",
                   help="Skip downloading; rebuild the index CSV from images already on disk.")
    a = p.parse_args()
    if a.reindex_only:
        reindex_from_disk()
        return
    download_ood_benchmark(a.per_generator, a.max_scan, a.min_scan, a.shuffle_buffer, a.seed, a.force)


if __name__ == "__main__":
    main()
