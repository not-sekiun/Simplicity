"""The shortcut audit, promoted from two ad-hoc scripts into the pull pipeline itself.

`scripts/audit_data.py` (the blind probe) and `scripts/audit_corpora.py` (the
saturation/bytes-per-px fingerprint) were both written after the fact, to
explain a corpus that had already gone bad: SID_Set's composition shortcut,
the SD3 recaptioning corpus mislabelled as generator output, the depth-map
pexels mirror. Each incident shipped, trained, and cost real time before
someone thought to run the check that would have caught it in one pass. This
package is that check moved BEFORE the fact -- `cli/pull.py` runs
:func:`audit_corpus` at the end of every pull unless `--no-audit`, and
`aigc_detect.data.corpus.assert_trainable` refuses to resolve a suspect
corpus into a training manifest without the deliberate override
`gate.is_suspect` looks for. See `probe.py` for the statistical check itself,
`gate.py` for where its verdict lives and how it gates, and `health.py` for
the (non-gating) saturation fingerprint.

WHY A SINGLE-LABEL PULL NEEDS A REFERENCE POOL. Most sources in `sources.yaml`
are two-class on their own (`tiny_genimage`, `cifake`, `wildrf`,
`aigc_detect_bench`) and the probe runs on exactly their own rows. Several are
not: `nano_banana`, `midjourney_v6` and `dalle3_holdout` are entirely AIGC;
`unsplash` is entirely real. A probe fit on one class alone cannot report
anything. `download_aigc_modern.py`'s own docstring already names the right
comparison -- "if 16x16 greyscale separates these from OUR REALS" -- so a
single-label pull is probed against a sample of the opposite label drawn from
every OTHER corpus already in the registry, which is the population it will
actually sit beside in a training manifest.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from aigc_detect.config import LABEL_NAMES, RANDOM_SEED
from aigc_detect.data.audit import gate
from aigc_detect.data.audit.health import BPP_SUSPECT, SAT_SUSPECT, sample_health
from aigc_detect.data.audit.probe import (
    PROBE_SIDE,
    SHORTCUT_THRESHOLD,
    BlindProbeResult,
    image_probe_vector,
    run_blind_probe,
    tensor_probe_vector,
)
from aigc_detect.data.dataset import resolve_image_path
from aigc_detect.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "BPP_SUSPECT",
    "PROBE_SIDE",
    "SAT_SUSPECT",
    "SHORTCUT_THRESHOLD",
    "BlindProbeResult",
    "audit_corpus",
    "describe_group",
    "gate",
    "image_probe_vector",
    "run_blind_probe",
    "run_registry_audit",
    "sample_health",
    "tensor_probe_vector",
]


def _sample_records(df: pd.DataFrame, n: int, seed: int) -> list[tuple]:
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return [(resolve_image_path(p), int(lbl)) for p, lbl in zip(df["image_path"], df["label"], strict=True)]


def _reference_rows(exclude_id: str, label: int, seed: int) -> pd.DataFrame:
    """Rows of `label` from every OTHER on-disk corpus -- see the module
    docstring's section on single-label pulls."""
    from aigc_detect.data.corpus import all_corpora

    frames = []
    for cid, corpus in all_corpora().items():
        if cid == exclude_id:
            continue
        try:
            rows = corpus.rows()
        except SystemExit:
            continue
        frames.append(rows[rows["label"] == label])
    if not frames:
        return pd.DataFrame(columns=["image_path", "label", "source", "generator"])
    return pd.concat(frames, ignore_index=True)


def audit_corpus(
    corpus_id: str,
    *,
    sample: int = 600,
    seed: int = RANDOM_SEED,
    use_transform: bool = False,
    write: bool = True,
) -> BlindProbeResult:
    """Run the blind probe for one corpus and (by default) write its verdict
    into that corpus's own `corpus.yaml`. This is what `cli/pull.py` calls
    after every successful fetch."""
    from aigc_detect.data.corpus import get_corpus

    corpus = get_corpus(corpus_id)
    own = corpus.rows()
    labels_present = set(int(x) for x in own["label"].dropna().unique())

    if len(labels_present) >= 2:
        records = _sample_records(own, sample, seed)
        basis = "self (both labels present)"
    elif len(labels_present) == 1:
        only = labels_present.pop()
        other = 1 - only
        ref = _reference_rows(corpus_id, other, seed)
        records = _sample_records(own, sample, seed) + _sample_records(ref, sample, seed)
        basis = f"self ({LABEL_NAMES[only]}) vs. the rest of the registry's {LABEL_NAMES[other]}"
    else:
        records = []
        basis = "no labelled rows"

    result = run_blind_probe(records, seed, use_transform)
    if write:
        verdict = result.as_dict()
        verdict["basis"] = basis
        verdict["sample"] = sample
        verdict["seed"] = seed
        verdict["audited_at"] = date.today().isoformat()
        gate.write_verdict(corpus_id, verdict)
    return result


def describe_group(paths: list) -> dict:
    """Per-(source,label) descriptive stats: format mix, top resolutions,
    aspect ratio spread, percent exactly square -- the aspect-ratio shortcut
    this project has already hit once (SID_Set) is exactly what these catch."""
    from collections import Counter

    import numpy as np
    from PIL import Image

    formats: Counter = Counter()
    resolutions: Counter = Counter()
    ratios: list[float] = []
    n_square = n_ok = 0
    for p in paths:
        try:
            with Image.open(p) as img:
                formats[img.format or "?"] += 1
                w, h = img.size
                resolutions[(w, h)] += 1
                ratios.append(w / h)
                n_square += int(w == h)
                n_ok += 1
        except Exception:
            continue

    ratios_arr = np.array(ratios) if ratios else np.array([float("nan")])
    return {
        "n_sampled": n_ok,
        "formats": formats.most_common(),
        "top_resolutions": resolutions.most_common(5),
        "aspect_min": float(np.min(ratios_arr)),
        "aspect_median": float(np.median(ratios_arr)),
        "aspect_max": float(np.max(ratios_arr)),
        "pct_square": 100.0 * n_square / n_ok if n_ok else float("nan"),
    }


def run_registry_audit(sample: int = 600, use_transform: bool = False, seed: int = RANDOM_SEED) -> int:
    """The full report `aigc audit-data` prints: per-corpus descriptive stats
    and blind probe, then one probe pooled across the whole registry -- the
    direct successor to `scripts/audit_data.py`'s `run_audit`, sourced from
    the corpus registry instead of the pre-Tier-5 `data/raw` / `aigc_ext` /
    `real_ext` directories (none of which exist on disk any more). Returns
    the count of suspect corpora, so a caller can treat >0 as a failing exit.
    """
    from aigc_detect.data.corpus import all_corpora

    print("=" * 78)
    print("DATA SHORTCUT AUDIT" + (" (transformed tensors)" if use_transform else " (raw images)"))
    print("=" * 78)

    pooled_records: list[tuple] = []
    suspects: list[str] = []

    for cid, corpus in sorted(all_corpora().items()):
        try:
            df = corpus.rows()
        except SystemExit:
            continue
        print(f"\n--- corpus: {cid} (role={corpus.role}) ---")

        source_records: list[tuple] = []
        for label, group in df.groupby("label"):
            label_name = LABEL_NAMES.get(int(label), str(label))
            sampled = group if len(group) <= sample else group.sample(n=sample, random_state=seed)
            paths = [resolve_image_path(p) for p in sampled["image_path"]]
            print(f"  [{label_name} label={label}] total={len(group)} sampled={len(paths)}")
            stats = describe_group(paths)
            print(f"    formats:          {stats['formats']}")
            print(f"    top resolutions:  {stats['top_resolutions']}")
            print(f"    aspect ratio:     min={stats['aspect_min']:.3f} "
                  f"median={stats['aspect_median']:.3f} max={stats['aspect_max']:.3f}")
            print(f"    pct exactly sq.:  {stats['pct_square']:.1f}%")
            source_records.extend((p, int(label)) for p in paths)

        pooled_records.extend(source_records)
        probe = run_blind_probe(source_records, seed, use_transform)
        if probe.skipped:
            print(f"  [blind probe] n={probe.n} -- skipped (single label; see pooled probe below)")
        else:
            print(f"  [blind probe] n={probe.n} balanced_acc={probe.balanced_acc:.4f} "
                  f"roc_auc={probe.roc_auc:.4f} -> {probe.verdict}")
            if probe.suspect:
                suspects.append(cid)

    print(f"\n--- pooled (every corpus, n={len(pooled_records)}) ---")
    pooled_probe = run_blind_probe(pooled_records, seed, use_transform)
    if pooled_probe.skipped:
        print(f"  [blind probe] n={pooled_probe.n} -- skipped")
    else:
        print(f"  [blind probe] n={pooled_probe.n} balanced_acc={pooled_probe.balanced_acc:.4f} "
              f"roc_auc={pooled_probe.roc_auc:.4f} -> {pooled_probe.verdict}")

    print("\n" + "=" * 78)
    print(f"Verdict threshold: balanced_acc >= {SHORTCUT_THRESHOLD:.2f} on a {PROBE_SIDE}x{PROBE_SIDE}-grayscale")
    print("blind probe means a label shortcut (e.g. aspect ratio / resolution) survives.")
    if suspects:
        print(f"SUSPECT corpora: {sorted(suspects)}")
    print("=" * 78)
    return len(suspects)
