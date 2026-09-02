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

        # PER-LABEL SATURATION, because the blind probe cannot answer this one.
        # probe.py converts to L before it looks, so colour is gone by the time
        # it measures anything -- a corpus whose AIGC half is monochrome and
        # whose real half is not carries a shortcut the probe is blind to by
        # construction, exactly like aspect ratio. Greyscale images are no
        # longer dropped at pull time (see fetchers/hf.py), so the skew they
        # can create has to be visible somewhere; this is where.
        # FULL --sample PER LABEL, not half of it. Halving was the first thing
        # tried and it made the number useless: at n=20 per label this reported
        # gaps of 0.14 for aigc_detect_bench and 0.21 for wildrf, and at n=500
        # the same two corpora measure 0.02 and 0.10. The estimator is noisy
        # enough that an under-sampled reading invents shortcuts that are not
        # there, which is a worse failure than not measuring at all.
        by_label, by_label_n = {}, {}
        for lab, tag in ((0, "real"), (1, "aigc")):
            sub_paths = [resolve_image_path(q) for q in df.loc[df.label == lab, "image_path"]]
            if sub_paths:
                h = sample_health(sub_paths, a.sample)
                by_label[tag], by_label_n[tag] = h["saturation"], h["checked"]
        r = {
            "name": cid, "role": corpus.role, "present": True,
            "rows": len(df),
            "real": int((df.label == 0).sum()), "aigc": int((df.label == 1).sum()),
            "generators": sorted(str(g) for g in set(df["generator"].dropna())) if "generator" in df.columns else [],
            "provenance": corpus.provenance,
            "sat_real": by_label.get("real"),
            "sat_aigc": by_label.get("aigc"),
            "sat_n_real": by_label_n.get("real", 0),
            "sat_n_aigc": by_label_n.get("aigc", 0),
            **health,
        }
        rows.append(r)
        if is_suspect(health):
            suspects.append(r)

    print(f"{'corpus':<26}{'role':<11}{'rows':>8}{'real':>8}{'aigc':>8}"
          f"{'sat':>8}{'sat|real':>10}{'sat|aigc':>10}{'byte/px':>9}{'median dims':>13}{'bad':>5}")
    print("-" * 120)
    for r in rows:
        if not r["present"]:
            print(f"{r['name']:<26}{r['role']:<11}{'not on disk':>32}")
            continue
        mark = " <-- SUSPECT" if r in suspects else ""
        # A gap this wide means one label is far more monochrome than the other.
        # 0.12 and the n>=100 floor are both calibrated against a measurement,
        # not guessed: at n=500 per label the established corpora sit at
        # tiny_genimage 0.030, aigc_detect_bench 0.017 and wildrf 0.095, so a
        # real gap has to clear the widest of those to mean anything. Below
        # n=100 the estimator swings wider than the threshold itself, so it
        # says "too few" rather than flagging on noise.
        if r["sat_real"] is not None and r["sat_aigc"] is not None:
            if min(r["sat_n_real"], r["sat_n_aigc"]) < 100:
                mark += "  (sample too small to judge colour split)"
            elif abs(r["sat_real"] - r["sat_aigc"]) > 0.12:
                mark += " <-- COLOUR SPLITS THE LABELS"
        dims = f"{r['median_w']}x{r['median_h']}"

        def _sat(v):
            return f"{v:>10.3f}" if v is not None else f"{'-':>10}"

        print(f"{r['name']:<26}{r['role']:<11}{r['rows']:>8,}{r['real']:>8,}{r['aigc']:>8,}"
              f"{r['saturation']:>8.3f}{_sat(r['sat_real'])}{_sat(r['sat_aigc'])}"
              f"{r['bytes_per_px']:>9.3f}{dims:>13}{r['unreadable']:>5}{mark}")

    print(f"\nSampled up to {a.sample} images per corpus. Suspect if saturation < "
          f"0.10 (greyscale: derived map, not a photo) or bytes/px < 0.08 (upscaled/over-compressed).")
    print("sat|real and sat|aigc are that same measurement split by label. A gap above 0.12 means "
          "colour separates")
    print("the classes, which the blind probe cannot detect: it converts to greyscale before it "
          "probes. The estimator")
    print("needs n>=100 per label to be worth reading -- raise --sample before believing a gap.")

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
