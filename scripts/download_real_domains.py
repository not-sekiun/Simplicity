"""Pull REAL-image corpora from domains our training pool does not cover.

WHY. FINDINGS 2h: the training pool's real half is ImageNet and nothing else,
and any real image from an absent domain is mapped confidently into AIGC
territory. The failure is not "faces" -- it is CURATED PHOTOGRAPHY. Observed
live: enthusiast/nature photography and Pinterest flag heavily, while Google
Images results for "dogs"/"cats" (web-snapshot register, i.e. ImageNet-like)
flag almost nothing. That matches the two correlations measured among true
reals: P(AIGC) rises as images get smoother (edge-energy rho=-0.215) and more
saturated (rho=+0.130) -- shallow depth of field, denoising, colour grading.

SIZING. The SID_Set reals-only run bought +0.0087 ood score from 4,000 images of
ONE new domain, versus +0.0041 for a 4x increase of the SAME domain. Value is in
the first few thousand images PER DOMAIN, so each source is capped rather than
taken whole: more domains beats more images. At the measured 5.1 decodes/s for
11 views, 4,000 images is ~13 min of GPU, so a cap of 4,000 keeps the whole pull
inside one sitting.

SOURCES
  unsplash  wtcherr/unsplash_5k     professional/enthusiast photography
  pexels    cj-mills/pexels-110k-*  same register, 768p, streamed with a cap

NOT INCLUDED, deliberately:
  - LHQ (90k landscapes): multi-GB for one narrow slice, and Unsplash/Pexels are
    already landscape- and nature-heavy. Revisit only if nature stays a weak cell.
  - LAION-aesthetics 6.5+: has real images and scores the right axis, but 6.5+ is
    heavy on illustration and renders. Artwork in the REAL class risks teaching
    "painterly => real", and AI-art detection currently WORKS. Not worth trading.
  - CelebA-HQ: the portrait corner, deferred by user decision -- WhichFaceIsReal
    is one extreme cell of the axis, not the axis itself.

Usage:
    uv run python scripts/download_real_domains.py --source unsplash --limit 4000
    uv run python scripts/download_real_domains.py --source pexels   --limit 4000
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

from aigc_detect.config import DATA_DIR, LABEL_REAL, PROCESSED_DIR

REAL_EXT_DIR = DATA_DIR / "real_ext"

SOURCES = {
    "unsplash": {
        "mode": "hf_stream",
        "repo": "wtcherr/unsplash_5k",
        "split": "train",
        "image_col": "image",
        "generator": "Real_Unsplash",
    },
    # Replacement for the depth-map mirror below. Parquet-embedded images, so it
    # uses the same hf_stream path as unsplash -- no archive handling. Verified
    # before wiring it in, by sampling 20 rows off the stream: mode=RGB, mean
    # saturation 0.313 (unsplash 0.29, the bad mirror 0.0000), full-resolution
    # originals from 4999x3328 to 7680x5120 rather than 768p derivatives.
    "pexels": {
        "mode": "hf_stream",
        "repo": "ujin-song/pexels-image-60k",
        "split": "train",
        "image_col": "image",
        "generator": "Real_Pexels",
    },
    # DO NOT RE-ENABLE THIS MIRROR. Despite the name, the archive's images/
    # folder holds Depth Anything OUTPUTS -- 109,971 single-channel depth maps
    # named after the Pexels photos they were derived from. The photos are not
    # in the ZIP at all; attributes_df.json carries only titles. Every member is
    # mode=L, which convert("RGB") silently widens to 3 identical channels, so
    # the pull "succeeded" with 4,000 depth maps labelled real. Scoring the
    # corpus is what caught it: mean P(AIGC)=0.999 with 100% over threshold,
    # more AI-looking than the actual AIGC training set. The greyscale gate
    # below now aborts this pull at row 200. A replacement needs to be a mirror
    # of the PHOTOGRAPHS; verify with _mean_saturation before trusting it.
    #
    # "pexels": {
    #     "mode": "hf_zip",
    #     "repo": "cj-mills/pexels-110k-768p-min-jpg-depth-anything-large-hf",
    #     "zip_name": "pexels-110k-768p-min-jpg-depth-anything-large-hf.zip",
    #     "member_prefix": "pexels-110k-768p-min-jpg-depth-anything-large-hf/images/",
    #     "generator": "Real_Pexels",
    # },
}

# Below this the image is a thumbnail, and upscaling to the backbone's 336px
# makes it smoother -- which we measured pushes P(AIGC) UP. A blurry real is
# worse than no real here: it teaches the exact artifact we are trying to unlearn.
MIN_SIDE = 384

# Ceiling on the long side. pexels-image-60k ships full-resolution originals
# (6000x4000 and up), which is 16-40 GB for a 4,000-image pull and would make
# pexels the only corpus reaching the backbone's 336px via a ~18x downscale --
# a source-correlated resampling cue, the same class of artifact MIN_SIDE
# guards the other end of. 1536 puts it in the band the other corpora already
# occupy (sid 1024x768, wildrf 2448x2354) and cuts the pull by ~10x.
MAX_SIDE = 1536

# A photography corpus is colour. This gate exists because the first pexels
# mirror (cj-mills/pexels-110k-768p-min-jpg-depth-anything-large-hf) ships
# Depth Anything OUTPUTS under images/, named after the source photos it never
# includes -- 4,000 single-channel depth maps that `img.convert("RGB")` widens
# to 3 identical channels without complaining. They cleared MIN_SIDE (768p),
# decoded as valid JPEG, and were labelled real. The head trained on them got
# worse at every matched operating point, because a smooth detail-free image
# labelled REAL teaches the exact artifact MIN_SIDE is there to keep out.
# Nothing downstream noticed; only scoring the corpus did.
#
# Measured cost: this rejects 2.0% of unsplash, 2.2% of wildrf, 2.8% of sid --
# genuine black-and-white photographs, which are indistinguishable from a depth
# map one image at a time. That is the accepted price. The margin is wide
# (median saturation 0.27-0.29 against a depth map's 0.0000), and the corpus-
# level abort below is what actually catches a wrong mirror.
MIN_SATURATION = 0.02

# If most of what a source yields fails the gate, the mirror is the wrong
# artifact, not the images. Stop rather than quietly keeping the remainder.
MAX_REJECT_FRACTION = 0.5
MIN_SCANNED_BEFORE_ABORT = 200


def _mean_saturation(img) -> float:
    """Mean (max-min)/max over RGB channels, on a downscale -- 0.0 for greyscale."""
    import numpy as np

    a = np.asarray(img.resize((64, 64))).astype("float32")
    mx, mn = a.max(2), a.min(2)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def merge() -> Path:
    """Union every pulled photography index into one training manifest.

    One manifest means one cache stem and one embed pass over all of them,
    instead of a stem per source. WildRF is deliberately NOT merged in here: its
    reals are a separate manifest so that the arm trained without them stays
    available as the honest generalization reading against wildrf_test.
    """
    from aigc_detect.config import PHOTO_REAL_MANIFEST

    indexes = sorted(REAL_EXT_DIR.glob("*_index.csv"))
    if not indexes:
        raise SystemExit(f"[real-ext] no *_index.csv under {REAL_EXT_DIR}; pull a source first.")
    frames = []
    for p in indexes:
        df = pd.read_csv(p)
        present = df["image_path"].map(lambda q: Path(str(q)).is_file())
        if not present.all():
            print(f"[real-ext] {p.name}: dropping {int((~present).sum()):,} row(s) whose file is gone")
        frames.append(df[present])
        print(f"[real-ext] {p.name}: {int(present.sum()):,} rows")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image_path"])
    PHOTO_REAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PHOTO_REAL_MANIFEST, index=False)
    print(f"[real-ext] {len(out):,} rows -> {PHOTO_REAL_MANIFEST}")
    print(f"[real-ext] by source: {out['source'].value_counts().to_dict()}")

    # Also emit one manifest PER SOURCE. embed_views makes a single pass over a
    # manifest and writes every view only at the end, so one 8,000-row job is
    # all-or-nothing: an interruption at minute 25 of 30 leaves nothing on disk.
    # Per-source manifests are ~13 min each and land independently, and
    # train-head-views takes several --extra-train-manifest values, so the
    # training side is unaffected by the split.
    for src, grp in out.groupby("source"):
        per = PROCESSED_DIR / f"{src}.csv"
        grp.to_csv(per, index=False)
        print(f"[real-ext]   {len(grp):,} rows -> {per}")
    return PHOTO_REAL_MANIFEST


def main() -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(SOURCES))
    ap.add_argument("--merge", action="store_true",
                    help="Union all pulled indexes into data/processed/photo_real.csv and exit.")
    ap.add_argument("--limit", type=int, default=4000,
                    help="Images to keep (default 4000 -- see SIZING in the docstring).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Clear the source's image directory before pulling. Required to re-pull "
                         "a source, because filenames are positional and a partial re-pull "
                         "otherwise leaves both corpora mixed under one set of names.")
    a = ap.parse_args()

    if a.merge:
        return merge()
    if not a.source:
        raise SystemExit("[real-ext] pass --source or --merge")

    from PIL import Image

    cfg = SOURCES[a.source]
    out_dir = REAL_EXT_DIR / a.source
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = REAL_EXT_DIR / f"{a.source}_index.csv"

    # Filenames are positional ({source}_{kept:06d}.jpg), so re-pulling a source
    # overwrites index by index rather than replacing the corpus. Repointing
    # `pexels` at a corrected mirror silently produced a directory holding 266
    # new photographs and 3,734 images from the previous pull, all under the
    # names the fresh index claimed. A partial overlap is worse than a total one:
    # the corpus-level saturation check that catches a wholly wrong source reads
    # as a plausible mid-range number when a source is only partly replaced.
    existing = sorted(out_dir.glob("*.jpg"))
    if existing and not a.overwrite:
        raise SystemExit(
            f"[real-ext] {out_dir} already holds {len(existing):,} images. Re-pulling would "
            f"overwrite them index-by-index and leave a MIX of both corpora under one name.\n"
            f"           Delete the directory (and its embeddings) first, or pass --overwrite "
            f"to clear it as part of this pull."
        )
    if existing and a.overwrite:
        print(f"[real-ext] --overwrite: clearing {len(existing):,} existing images from {out_dir}")
        for p in existing:
            p.unlink()

    def _iter_images():
        """Yield PIL images from whichever shape this source ships in."""
        if cfg["mode"] == "hf_stream":
            from datasets import load_dataset

            print(f"[real-ext] streaming {cfg['repo']} split={cfg['split']} -> {out_dir}")
            for item in load_dataset(cfg["repo"], split=cfg["split"], streaming=True):
                img = item[cfg["image_col"]]
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"]))
                yield img
        else:
            import zipfile

            from huggingface_hub import hf_hub_download

            print(f"[real-ext] fetching archive {cfg['repo']}/{cfg['zip_name']}")
            zpath = hf_hub_download(cfg["repo"], cfg["zip_name"], repo_type="dataset")
            zf = zipfile.ZipFile(zpath)
            members = [n for n in zf.namelist()
                       if n.startswith(cfg["member_prefix"]) and not n.endswith("/")]
            print(f"[real-ext] archive holds {len(members):,} image member(s)")
            for name in members:
                yield Image.open(io.BytesIO(zf.read(name)))

    rows, kept, seen, small, bad, mono = [], 0, 0, 0, 0, 0
    for img in _iter_images():
        seen += 1
        if kept >= a.limit:
            break
        # Check the SOURCE mode before convert("RGB") -- conversion widens a
        # single-channel image to 3 identical channels and hides the problem.
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
                    f"[real-ext] ABORT: {mono:,}/{seen:,} images from {cfg['repo']} are "
                    f"greyscale (source mode {source_mode}, saturation < {MIN_SATURATION}). "
                    f"This mirror is almost certainly not photographs -- check whether "
                    f"'{cfg.get('member_prefix', cfg.get('image_col'))}' holds derived maps "
                    f"(depth, segmentation, matte) rather than the source images."
                )
            continue
        if max(img.size) > MAX_SIDE:
            scale = MAX_SIDE / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        dest = out_dir / f"{a.source}_{kept:06d}.jpg"
        # Re-encode at q95 to match how tiny_genimage was stored, so JPEG
        # quality is not a source-correlated cue. The robustness views apply
        # their own compression on top of whatever the file already carries.
        img.save(dest, "JPEG", quality=95)
        rows.append({"image_path": str(dest.resolve()), "label": LABEL_REAL,
                     "source": f"{a.source}_real", "generator": cfg["generator"]})
        kept += 1
        if kept % 500 == 0:
            print(f"[real-ext] {kept:,}/{a.limit:,} kept ({seen:,} scanned, "
                  f"{small:,} too small, {mono:,} greyscale, {bad:,} unreadable)", flush=True)

    if not rows:
        raise SystemExit(f"[real-ext] nothing kept from {cfg['repo']}")

    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"[real-ext] {kept:,} images -> {out_dir}")
    print(f"[real-ext] index -> {index_path}")
    print(f"[real-ext] scanned {seen:,}; dropped {small:,} under {MIN_SIDE}px, "
          f"{mono:,} greyscale, {bad:,} unreadable")
    return index_path


if __name__ == "__main__":
    main()
