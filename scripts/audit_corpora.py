"""Empirical health fingerprint of every corpus in the registry: saturation + bytes/px.

WHY THIS EXISTS. A pexels mirror named `pexels-110k-768p-min-jpg-depth-anything
-large-hf` turned out to ship Depth Anything OUTPUTS under images/, named after
the photos it never included. Every member was mode=L, which convert("RGB")
widens to three identical channels without error, so 4,000 depth maps decoded
as valid JPEGs, cleared the 384px floor, and were written as label=0 REAL. The
manifest fingerprints all passed -- they verify that embeddings match the
manifest, never that the manifest holds photographs. Training on them made the
head worse at every matched operating point.

The real check now lives in `aigc_detect.data.audit.health` and runs against
every corpus `registry/corpora.yaml` declares, rather than the fixed manifest
list this script used to carry by hand (most of those constants -- `wildrf_real`,
`sid_real`, `photo_real` -- named pre-Tier-5 paths that no longer exist). This
is a thin shim over that module, kept for `uv run python scripts/audit_corpora.py`.

Usage:
    uv run python scripts/audit_corpora.py
    uv run python scripts/audit_corpora.py --sample 400   # tighter estimates
    uv run python scripts/audit_corpora.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aigc_detect.data.audit.health import is_suspect, sample_health
from aigc_detect.data.corpus import all_corpora
from aigc_detect.data.dataset import resolve_image_path

ROOT = Path(__file__).resolve().parents[1]


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

    rows, suspects = [], []
    for cid, corpus in sorted(all_corpora().items()):
        try:
            df = corpus.rows()
        except SystemExit:
            rows.append({"name": cid, "role": corpus.role, "present": False})
            continue

        paths = [resolve_image_path(p) for p in df["image_path"]]
        health = sample_health(paths, a.sample)
        r = {
            "name": cid, "role": corpus.role, "present": True,
            "rows": len(df),
            "real": int((df.label == 0).sum()), "aigc": int((df.label == 1).sum()),
            "generators": sorted(str(g) for g in set(df["generator"].dropna())) if "generator" in df.columns else [],
            "provenance": corpus.provenance,
            **health,
        }
        rows.append(r)
        if is_suspect(health):
            suspects.append(r)

    print(f"{'corpus':<26}{'role':<11}{'rows':>8}{'real':>8}{'aigc':>8}"
          f"{'sat':>8}{'byte/px':>9}{'median dims':>13}{'bad':>5}")
    print("-" * 100)
    for r in rows:
        if not r["present"]:
            print(f"{r['name']:<26}{r['role']:<11}{'not on disk':>32}")
            continue
        mark = " <-- SUSPECT" if r in suspects else ""
        dims = f"{r['median_w']}x{r['median_h']}"
        print(f"{r['name']:<26}{r['role']:<11}{r['rows']:>8,}{r['real']:>8,}{r['aigc']:>8,}"
              f"{r['saturation']:>8.3f}{r['bytes_per_px']:>9.3f}{dims:>13}{r['unreadable']:>5}{mark}")

    print(f"\nSampled up to {a.sample} images per corpus. Suspect if saturation < "
          f"0.10 (greyscale: derived map, not a photo) or bytes/px < 0.08 (upscaled/over-compressed).")

    if suspects:
        print(f"\n!! {len(suspects)} corpus/corpora look wrong: "
              + ", ".join(s["name"] for s in suspects)
              + "\n   Open a few images before training on them.")

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\n[audit] wrote {a.json}")
    return 1 if suspects else 0


if __name__ == "__main__":
    raise SystemExit(main())
