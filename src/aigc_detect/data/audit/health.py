"""Per-corpus image health: saturation and bytes/px, the depth-map fingerprint.

A pexels mirror named `...depth-anything-large-hf` turned out to ship Depth
Anything OUTPUTS under `images/`, named after the photos it never included.
Every member was mode=L, which `convert("RGB")` widens to three identical
channels without error, so 4,000 depth maps decoded as valid JPEGs, cleared
the resolution floor, and were written as label=0 REAL. Every manifest
fingerprint still passed -- they verify embeddings match the manifest, never
that the manifest holds photographs. Training on them made the head worse at
every matched operating point, and nothing in the pipeline was positioned to
notice; only scoring the corpus by hand did (see `sources.yaml`'s `pexels`
entry for the full incident).

  saturation  mean (max-min)/max over RGB, downscaled to 64x64. Real photo
              corpora sit at 0.30-0.36; the depth maps were 0.000. This is
              the discriminator, and it is also the `quality_gate.
              min_saturation` check `hf.py` runs live during a pull -- this
              module is the same measurement run again, after the fact, over
              a whole corpus rather than aborting mid-stream.
  bytes/px    file size over pixel count. Low means the frame is large but
              the detail is gone -- upscaled or over-compressed, which reads
              as SMOOTH, and smoothness is what drives P(AIGC) up in a real
              detector (docs/findings.md 2h).

Neither is a pass/fail gate on its own -- unlike the blind probe in
`probe.py`, this is a fingerprint to compare a new corpus against the
established ones, not a threshold that blocks a pull.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: A real photography corpus has not scored below this; the depth-map corpus
#: scored 0.000. Below this, the corpus is probably not photographs.
SAT_SUSPECT = 0.10

#: Below this, images are probably upscaled or over-compressed past the point
#: of carrying real detail.
BPP_SUSPECT = 0.08


def sample_health(paths: list, n_sample: int, seed: int = 0) -> dict:
    """Saturation, bytes/px, and dimensions over a random sample of `paths`."""
    from PIL import Image

    rs = np.random.RandomState(seed)
    pick = paths if len(paths) <= n_sample else [paths[i] for i in rs.choice(len(paths), n_sample, replace=False)]
    sat, bpp, widths, heights, unreadable = [], [], [], [], 0
    for p in pick:
        p = Path(p)
        try:
            with Image.open(p) as im:
                w, h = im.size
                rgb = im.convert("RGB").resize((64, 64))
            a = np.asarray(rgb).astype("float32")
            mx, mn = a.max(2), a.min(2)
            sat.append(float(((mx - mn) / (mx + 1e-6)).mean()))
            bpp.append(os.path.getsize(p) / (w * h))
            widths.append(w)
            heights.append(h)
        except Exception:
            unreadable += 1
    return {
        "checked": len(pick),
        "unreadable": unreadable,
        "saturation": float(np.mean(sat)) if sat else float("nan"),
        "bytes_per_px": float(np.median(bpp)) if bpp else float("nan"),
        "median_w": int(np.median(widths)) if widths else 0,
        "median_h": int(np.median(heights)) if heights else 0,
    }


def is_suspect(health: dict) -> bool:
    sat, bpp = health.get("saturation"), health.get("bytes_per_px")
    if sat != sat or bpp != bpp:  # NaN -- nothing readable, not evidence either way
        return False
    return sat < SAT_SUSPECT or bpp < BPP_SUSPECT
