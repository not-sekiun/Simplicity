"""Pull MODERN AI-generator images -- the one gap the shipping head still has.

WHY. Our AIGC training half is entirely GenImage-era (2022): ADM, GLIDE, SD15,
VQDM, Wukong, Midjourney-v4-ish, BigGAN. Measured on the unseen-generator tier
at the shipping threshold, GAN is done (0.992) and diffusion is not (0.859), and
the misses are concentrated in the models people actually use now:

    DALLE2   0.542      FLUX.1-dev  0.562 (n=73 probe)
    ADM      0.683      Midjourney  0.736

This pulls three CURRENT generators from three different publishers:

    midjourney_v6   Photoroom/midjourney-v6-recap          art / illustration
    sd3             gmongaras/Stable_Diffusion_3_Recaption  photographic
    nano_banana     bitmind/nano-banana                     Gemini 2.5 Flash Image

THREE PUBLISHERS ON PURPOSE. A single dataset is a single provenance -- one
prompt distribution, one resolution, one encoder -- and a detector will happily
learn "this source" instead of "this generator". Mixing generators INSIDE one
source does not fix that; the confound is source-level. Three independent
sources means a source shortcut would have to be learned three times over, which
is harder than learning what they share. Verify it worked with the blind probe
in scripts/audit_data.py: if 16x16 greyscale separates these from our reals
above ~0.70 balanced accuracy, a shortcut survived and this data is unusable.

DALLE3 IS DELIBERATELY NOT HERE. It is the held-out modern-diffusion tier. If we
train on every modern generator we can find, we lose the ability to measure
whether any of this generalizes.

RE-ENCODED TO JPEG q95, like every other corpus in the project. These ship as
pristine generator output; our reals are all q95. Adding raw PNGs as the AI
class would let the head learn a FORMAT, not a generator -- it would look like a
huge win and teach nothing. Same family as the depth-map fault, subtler.

Usage:
    uv run python scripts/download_aigc_modern.py --source midjourney-v6 --limit 1500
    uv run python scripts/download_aigc_modern.py --merge
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

from aigc_detect.config import DATA_DIR, LABEL_AIGC, PROCESSED_DIR

AIGC_EXT_DIR = DATA_DIR / "aigc_ext"
AIGC_MODERN_MANIFEST = PROCESSED_DIR / "aigc_modern.csv"

SOURCES = {
    "midjourney-v6": {
        "repo": "Photoroom/midjourney-v6-recap",
        "split": "train",
        "generator": "MidjourneyV6",
        "dir": "midjourney_v6",
    },
    # "sd3" -- REMOVED 2026-08-30. gmongaras/Stable_Diffusion_3_Recaption is a
    # RECAPTIONING corpus: real photographs paired with SD3-authored captions,
    # NOT SD3 output. Pulling it labelled 1,500 real photos as AIGC and cost
    # DALLE2 -0.072 degraded AUC before it was caught. Do not re-add it. The
    # pulled data and the full evidence are in data/quarantine/README.md.
    # The "-recap"/"_Recaption" suffix is NOT a reliable provenance signal in
    # either direction -- Photoroom/midjourney-v6-recap carries it and IS
    # genuine generator output. Check the pixels: a real generator dump is one
    # resolution (1024x1024, ~100% square); a scraped corpus has hundreds.
    "nano-banana": {
        "repo": "bitmind/nano-banana",
        "split": "train",
        "generator": "NanoBanana",
        "dir": "nano_banana",
    },
    # DALLE3 -- HELD OUT as the modern-diffusion EVAL tier. Pull it only with
    # --source dalle3-holdout, and never add it to the training manifest.
    "dalle3-holdout": {
        "repo": "OpenDatasets/dalle-3-dataset",
        "split": "train",
        "generator": "DALLE3",
        "dir": "dalle3_holdout",
    },
}

# Same gates as scripts/download_real_domains.py, for the same reasons.
MIN_SIDE = 384
MAX_SIDE = 1536
MIN_SATURATION = 0.02
MAX_REJECT_FRACTION = 0.5
MIN_SCANNED_BEFORE_ABORT = 200


def _mean_saturation(img) -> float:
    import numpy as np

    a = np.asarray(img.resize((64, 64))).astype("float32")
    mx, mn = a.max(2), a.min(2)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def _find_image(example):
    """Datasets differ on the image column name; take the first image-like value."""
    from PIL import Image

    for key in ("image", "img", "jpg", "png", "photo"):
        if key in example and example[key] is not None:
            v = example[key]
            if isinstance(v, dict) and v.get("bytes"):
                return Image.open(io.BytesIO(v["bytes"]))
            if hasattr(v, "convert"):
                return v
    return None


def merge() -> Path:
    """Union the per-generator indexes into one TRAINING manifest.

    dalle3_holdout is excluded by construction -- it is the eval tier.
    """
    indexes = sorted(p for p in AIGC_EXT_DIR.glob("*_index.csv") if "dalle3" not in p.name)
    if not indexes:
        raise SystemExit(f"[aigc-ext] no *_index.csv under {AIGC_EXT_DIR}; pull a source first.")
    out = pd.concat([pd.read_csv(p) for p in indexes], ignore_index=True)
    out = out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    out.to_csv(AIGC_MODERN_MANIFEST, index=False)
    print(f"[aigc-ext] {len(out):,} rows -> {AIGC_MODERN_MANIFEST}")
    print(f"[aigc-ext] by generator: {out['generator'].value_counts().to_dict()}")
    return AIGC_MODERN_MANIFEST


def main() -> Path:
    from datasets import load_dataset
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--merge", action="store_true", help="Union pulled indexes into the training manifest and exit.")
    ap.add_argument("--limit", type=int, default=1500, help="Images to keep (default 1500).")
    ap.add_argument("--overwrite", action="store_true", help="Clear the source's image directory first.")
    a = ap.parse_args()

    if a.merge:
        return merge()
    if not a.source:
        raise SystemExit("Pass --source or --merge.")

    cfg = SOURCES[a.source]
    out_dir = AIGC_EXT_DIR / cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = AIGC_EXT_DIR / f"{cfg['dir']}_index.csv"

    existing = sorted(out_dir.glob("*.jpg"))
    if existing and not a.overwrite:
        raise SystemExit(
            f"[aigc-ext] {out_dir} already holds {len(existing):,} images. Filenames are positional, "
            f"so re-pulling would leave a MIX of two corpora under one set of names. Delete the "
            f"directory (and its embeddings) or pass --overwrite."
        )
    for p in existing:
        p.unlink()

    print(f"[aigc-ext] streaming {cfg['repo']} split={cfg['split']} -> {out_dir}")
    print(f"[aigc-ext] generator={cfg['generator']} limit={a.limit:,}")
    ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True)

    rows, kept, seen, small, mono, bad = [], 0, 0, 0, 0, 0
    for ex in ds:
        if kept >= a.limit:
            break
        seen += 1
        img = _find_image(ex)
        if img is None:
            bad += 1
            continue
        source_mode = img.mode
        try:
            img = img.convert("RGB")
        except Exception:
            bad += 1
            continue
        if min(img.size) < MIN_SIDE:
            small += 1
            continue
        if source_mode in {"L", "1", "I", "F", "I;16"} or _mean_saturation(img) < MIN_SATURATION:
            mono += 1
            if seen >= MIN_SCANNED_BEFORE_ABORT and mono / seen > MAX_REJECT_FRACTION:
                raise SystemExit(
                    f"[aigc-ext] ABORT: {mono:,}/{seen:,} images from {cfg['repo']} are greyscale. "
                    f"This is probably not what the name says it is -- inspect before retrying."
                )
            continue
        if max(img.size) > MAX_SIDE:
            s = MAX_SIDE / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)

        dest = out_dir / f"{cfg['dir']}_{kept:06d}.jpg"
        # q95 to match how every other corpus is stored, so JPEG quality is not
        # a source-correlated cue the head can read instead of the generator.
        img.save(dest, "JPEG", quality=95)
        rows.append({
            "image_path": str(dest.resolve()),
            "label": LABEL_AIGC,
            "source": f"aigc_modern_{cfg['dir']}",
            "generator": cfg["generator"],
        })
        kept += 1
        if kept % 250 == 0:
            print(f"[aigc-ext] {kept:,}/{a.limit:,} kept ({seen:,} scanned, {small:,} too small, "
                  f"{mono:,} greyscale, {bad:,} unreadable)", flush=True)

    if not rows:
        raise SystemExit(f"[aigc-ext] nothing kept from {cfg['repo']}")

    df = pd.DataFrame(rows)
    df.to_csv(index_path, index=False)
    # Also emit a per-source MANIFEST under data/processed/. Each source then has
    # its own cache stem and embeds in its own pass, so an interrupted embed
    # costs one source rather than all of them -- embed_views writes every view
    # only at the end of a single pass over the data.
    per_source = PROCESSED_DIR / f"{cfg['dir']}.csv"
    df.to_csv(per_source, index=False)
    print(f"[aigc-ext] {kept:,} images -> {out_dir}")
    print(f"[aigc-ext] index -> {index_path}")
    print(f"[aigc-ext] manifest -> {per_source}")
    print(f"[aigc-ext] scanned {seen:,}; dropped {small:,} under {MIN_SIDE}px, {mono:,} greyscale, "
          f"{bad:,} unreadable")
    return index_path


if __name__ == "__main__":
    main()
