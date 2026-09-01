"""Empirical provenance + health audit of every manifest the project uses.

WHY THIS EXISTS. A pexels mirror named `pexels-110k-768p-min-jpg-depth-anything
-large-hf` turned out to ship Depth Anything OUTPUTS under images/, named after
the photos it never included. Every member was mode=L, which convert("RGB")
widens to three identical channels without error, so 4,000 depth maps decoded
as valid JPEGs, cleared the 384px floor, and were written as label=0 REAL. The
manifest fingerprints all passed -- they verify that embeddings match the
manifest, never that the manifest holds photographs. Training on them made the
head worse at every matched operating point.

Nothing in the pipeline was positioned to notice. This is that check: it reads
what is actually on disk and reports the two properties that separated the
depth maps from real photography.

  saturation  mean (max-min)/max over RGB. Real photo corpora sit at 0.30-0.36.
              The depth maps were 0.000. This is the discriminator.
  bytes/px    file size over pixel count. Low means the frame is large but the
              detail is gone -- upscaled or over-compressed, which reads as
              SMOOTH, and smoothness is what drives P(AIGC) up (FINDINGS 2h).

Neither is a pass/fail gate here. They are a fingerprint of each corpus: run it
after adding any source and compare the new row against the established ones.
An outlier means look at the images before training on them.

Usage:
    uv run python scripts/audit_corpora.py
    uv run python scripts/audit_corpora.py --sample 400   # tighter estimates
    uv run python scripts/audit_corpora.py --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from aigc_detect import config as C

ROOT = Path(__file__).resolve().parents[1]

# (name, manifest path, role, upstream origin, what builds it)
CORPORA = [
    ("train", C.TRAIN_MANIFEST, "TRAIN", "TheKernel01/Tiny-GenImage (HF)", "main.py split"),
    ("val", C.VAL_MANIFEST, "EVAL", "TheKernel01/Tiny-GenImage (HF)", "main.py split"),
    ("heldout", C.HELDOUT_MANIFEST, "EVAL", "TheKernel01/Tiny-GenImage (HF)", "main.py build-heldout"),
    ("ood", C.OOD_MANIFEST, "EVAL", "TheKernel01/AIGC-Detection-Benchmark (HF)", "main.py build-ood"),
    ("demo_val", C.DEMO_VAL_MANIFEST, "EVAL", "xthink/coco-2017-val-images (Kaggle)", "main.py build-demo-val"),
    ("wildrf_test", C.WILDRF_TEST_MANIFEST, "EVAL", "WildRF, arXiv:2406.09398", "scripts/make_wildrf.py"),
    ("wildrf_real", C.WILDRF_REAL_MANIFEST, "TRAIN", "WildRF, arXiv:2406.09398", "scripts/make_wildrf.py"),
    ("sid_real", C.SID_REAL_MANIFEST, "TRAIN", "saberzl/SID_Set (HF)", "scripts/make_sid_real.py"),
    ("unsplash_real", C.UNSPLASH_REAL_MANIFEST, "TRAIN", "wtcherr/unsplash_5k (HF)",
     "download_real_domains.py --source unsplash"),
    ("pexels_real", C.PEXELS_REAL_MANIFEST, "TRAIN", "ujin-song/pexels-image-60k (HF)",
     "download_real_domains.py --source pexels"),
    ("photo_real", C.PHOTO_REAL_MANIFEST, "TRAIN", "unsplash + pexels union", "download_real_domains.py --merge"),
    ("train_ext", C.TRAIN_EXT_MANIFEST, "TRAIN", "TheKernel01/Tiny-GenImage, skip_rows=8400", "scripts/make_train_ext.py"),
]

# Stems concatenated into the head that predict.py loads by default.
SHIPPING = {"train", "sid_real", "unsplash_real"}

# Reference bands, measured across the corpora above. A real photography corpus
# has not scored below 0.30 mean saturation; the depth-map corpus scored 0.000.
SAT_SUSPECT = 0.10
BPP_SUSPECT = 0.08


def sample_health(paths, n_sample, seed=0):
    rs = np.random.RandomState(seed)
    pick = paths if len(paths) <= n_sample else [paths[i] for i in rs.choice(len(paths), n_sample, replace=False)]
    sat, bpp, w_, h_, unreadable = [], [], [], [], 0
    for p in pick:
        try:
            with Image.open(p) as im:
                w, h = im.size
                rgb = im.convert("RGB").resize((64, 64))
            a = np.asarray(rgb).astype("float32")
            mx, mn = a.max(2), a.min(2)
            sat.append(float(((mx - mn) / (mx + 1e-6)).mean()))
            bpp.append(os.path.getsize(p) / (w * h))
            w_.append(w)
            h_.append(h)
        except Exception:
            unreadable += 1
    return sat, bpp, w_, h_, unreadable, len(pick)


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=150, help="Images to open per corpus (default 150).")
    ap.add_argument("--json", default=None, help="Also write the full table here as JSON.")
    a = ap.parse_args()

    rows, suspect = [], []
    for name, path, role, origin, builder in CORPORA:
        if not Path(path).exists():
            rows.append({"name": name, "role": role, "present": False, "manifest": rel(path),
                         "origin": origin, "builder": builder})
            continue
        df = pd.read_csv(path)
        sat, bpp, w_, h_, unreadable, checked = sample_health(df.image_path.tolist(), a.sample)
        dirs = collections.Counter(str(Path(p).parent) for p in df.image_path)
        r = {
            "name": name, "role": role, "present": True, "manifest": rel(path),
            "origin": origin, "builder": builder,
            "rows": len(df), "real": int((df.label == 0).sum()), "aigc": int((df.label == 1).sum()),
            "generators": sorted(set(df.generator)) if "generator" in df.columns else [],
            "image_dirs": [rel(d) for d, _ in dirs.most_common()],
            "saturation": float(np.mean(sat)) if sat else float("nan"),
            "bytes_per_px": float(np.median(bpp)) if bpp else float("nan"),
            "median_w": int(np.median(w_)) if w_ else 0,
            "median_h": int(np.median(h_)) if h_ else 0,
            "unreadable": unreadable, "checked": checked,
            "in_shipping_head": name in SHIPPING,
        }
        rows.append(r)
        if sat and (r["saturation"] < SAT_SUSPECT or r["bytes_per_px"] < BPP_SUSPECT):
            suspect.append(r)

    print(f"{'corpus':<15}{'role':<7}{'rows':>7}{'real':>7}{'aigc':>7}{'gen':>5}"
          f"{'sat':>8}{'byte/px':>9}{'median dims':>13}{'bad':>5}")
    print("-" * 93)
    for r in rows:
        if not r["present"]:
            print(f"{r['name']:<15}{r['role']:<7}{'not on disk':>25}")
            continue
        mark = " *" if r["in_shipping_head"] else ""
        if r in suspect:
            mark = " <-- SUSPECT"
        dims = f"{r['median_w']}x{r['median_h']}"
        print(f"{r['name']:<15}{r['role']:<7}{r['rows']:>7,}{r['real']:>7,}{r['aigc']:>7,}"
              f"{len(r['generators']):>5}{r['saturation']:>8.3f}{r['bytes_per_px']:>9.3f}"
              f"{dims:>13}{r['unreadable']:>5}{mark}")

    print(f"\n* feeds the shipping head. Sampled up to {a.sample} images per corpus.")
    print(f"suspect if saturation < {SAT_SUSPECT} (greyscale: derived map, not a photo) "
          f"or bytes/px < {BPP_SUSPECT} (upscaled/over-compressed).")

    print("\nPROVENANCE")
    print("-" * 93)
    for r in rows:
        if not r["present"]:
            continue
        print(f"  {r['name']}")
        print(f"      origin    {r['origin']}")
        print(f"      built by  {r['builder']}")
        print(f"      manifest  {r['manifest']}")
        print(f"      images    {r['image_dirs'][0]}"
              + (f"   (+{len(r['image_dirs']) - 1} more dirs)" if len(r["image_dirs"]) > 1 else ""))

    if suspect:
        print(f"\n!! {len(suspect)} corpus/corpora look wrong: "
              + ", ".join(s["name"] for s in suspect)
              + "\n   Open a few images before training on them.")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\n[audit] wrote {a.json}")
    return 1 if suspect else 0


if __name__ == "__main__":
    raise SystemExit(main())
