"""Demo validation set (challenge brief 5.4): "Validation Dataset (for
Demonstration Purposes Only)". Non-AIGC = COCO val2017 (~4998 imgs). AIGC =
WildFake's "DALL·E Advanced" generator subset (~8843 imgs).

IMPORTANT: per the brief, "Do not use the following data during training."
This is a self-reported, out-of-training benchmark only — it "will not
contribute to the final score." It's kept under data/demo_val/, structurally
separate from data/raw/ (training sources), so scripts/make_splits.py (which
only globs data/raw/*_index.csv) can never pick it up.

Usage:
    uv run main.py download-demo coco-val2017
    uv run main.py download-demo wildfake-dalle-advanced
    uv run main.py build-demo-val

COCO val2017 downloads via a Kaggle mirror by default (same Kaggle
credentials as CIFAKE) since the official S3 bucket can be severely
throttled on some networks; falls back to the official S3 zip (no auth
needed, just slower) if Kaggle isn't set up. WildFake's "DALL·E Advanced"
subset does NOT download automatically: this network
cannot reach ModelScope's API or SDK endpoints at all (confirmed — both hang
indefinitely), which matches the challenge brief's own note that the
ModelScope page needs a manual translate-button step. Fetch it yourself:
  1. Open https://modelscope.cn/datasets/hy2628982280/WildFake/summary
  2. Use the page's translate button if needed
  3. Find/download the "DALL·E Advanced" generator subset
  4. Extract its images into data/demo_val/wildfake_dalle_advanced/
Then run `uv run main.py download-demo wildfake-dalle-advanced` to index them.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from aigc_detect.config import DEMO_VAL_DIR, LABEL_AIGC, LABEL_REAL

COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL2017_KAGGLE_HANDLE = "xthink/coco-2017-val-images"  # same 5000 official val2017 images, faster CDN
COCO_VAL2017_DIR = DEMO_VAL_DIR / "coco_val2017"
WILDFAKE_DALLE_ADVANCED_DIR = DEMO_VAL_DIR / "wildfake_dalle_advanced"

# The brief cites 4998 COCO val2017 images; the standard public val2017.zip
# has 5000. We don't know which 2 the organizers excluded, so we index the
# full standard set — a stated, reasonable assumption (5.3 "Allowed
# assumptions"), off by at most 2 images out of ~5000.


def _write_index(name: str, records: list[tuple[str, int]]) -> Path:
    DEMO_VAL_DIR.mkdir(parents=True, exist_ok=True)
    index_path = DEMO_VAL_DIR / f"{name}_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source"])
        for image_path, label in records:
            writer.writerow([image_path, label, name])
    print(f"[{name}] wrote {len(records)} rows -> {index_path}")
    return index_path


def _download_coco_via_kaggle_mirror() -> list[Path]:
    """Fast path: a Kaggle-hosted re-upload of the same 5000 official val2017
    images. The official S3 bucket (images.cocodataset.org) is reachable but
    can be throttled to single-digit KB/s on some networks (observed: an
    18+ hour ETA for 815MB) — Kaggle's CDN was ~20MB/s (~40s) in comparison,
    matching CIFAKE's download speed. Requires the same Kaggle credentials
    as `download_cifake()`."""
    import kagglehub

    print(f"[coco_val2017] downloading Kaggle mirror '{COCO_VAL2017_KAGGLE_HANDLE}'...")
    mirror_root = Path(kagglehub.dataset_download(COCO_VAL2017_KAGGLE_HANDLE))
    return sorted(mirror_root.rglob("*.jpg"))


def _download_coco_via_official_s3() -> list[Path]:
    """Slow-but-guaranteed fallback: the official COCO S3 bucket, no auth needed."""
    import requests
    from tqdm import tqdm

    zip_path = DEMO_VAL_DIR / "val2017.zip"
    if not zip_path.exists():
        print(f"[coco_val2017] downloading {COCO_VAL2017_URL} (~815MB)...")
        resp = requests.get(COCO_VAL2017_URL, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print("[coco_val2017] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEMO_VAL_DIR)  # zip contains a top-level val2017/ folder
    zip_path.unlink(missing_ok=True)
    return sorted(COCO_VAL2017_DIR.glob("*.jpg"))


def download_coco_val2017(prefer_kaggle: bool = True) -> Path:
    DEMO_VAL_DIR.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    if prefer_kaggle:
        try:
            image_paths = _download_coco_via_kaggle_mirror()
        except Exception as err:  # noqa: BLE001 - any auth/network failure -> fall back
            print(f"[coco_val2017] Kaggle mirror failed ({err!r}), falling back to official S3...")

    if not image_paths:
        image_paths = _download_coco_via_official_s3()

    records = [(str(p.resolve()), LABEL_REAL) for p in image_paths]
    if not records:
        raise RuntimeError("No COCO val2017 images found from either source.")
    print(f"[coco_val2017] indexed {len(records)} images "
          f"(brief cites 4998; standard val2017 is 5000 -- see module docstring).")
    return _write_index("coco_val2017", records)


def index_wildfake_dalle_advanced() -> Path | None:
    images = [
        p for p in sorted(WILDFAKE_DALLE_ADVANCED_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ] if WILDFAKE_DALLE_ADVANCED_DIR.exists() else []

    if not images:
        print(f"""
No images found under {WILDFAKE_DALLE_ADVANCED_DIR}.

This must be fetched manually (see this module's docstring for why):
  1. Open https://modelscope.cn/datasets/hy2628982280/WildFake/summary
  2. Use the page's translate button if needed
  3. Find/download the "DALL·E Advanced" generator subset
  4. Extract its images into: {WILDFAKE_DALLE_ADVANCED_DIR}
Then re-run: uv run main.py download-demo wildfake-dalle-advanced
""".strip())
        return None

    records = [(str(p.resolve()), LABEL_AIGC) for p in images]
    print(f"[wildfake_dalle_advanced] indexed {len(records)} images "
          f"(brief cites 8843 for the full 'DALL-E Advanced' subset).")
    return _write_index("wildfake_dalle_advanced", records)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="which", required=True)
    sub.add_parser("coco-val2017")
    sub.add_parser("wildfake-dalle-advanced")
    args = parser.parse_args()

    if args.which == "coco-val2017":
        download_coco_val2017()
    else:
        index_wildfake_dalle_advanced()


if __name__ == "__main__":
    main()
