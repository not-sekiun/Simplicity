"""Hugging Face backends: `hf_streaming` and `hf_parquet`.

STREAMING VS PARQUET, AND WHY BOTH EXIST. Most sources here are far larger than
the few thousand images this project actually keeps (SID_Set alone is ~140GB),
so `hf_streaming` reads a `datasets` `IterableDataset` and stops once its cap
or quota is filled -- see `download_data.py` and `download_ood_benchmark.py`.
Tiny-GenImage is the exception: at ~8.4GB for the whole `train` split it is
cheaper to let `datasets.load_dataset(..., streaming=False)` pull the full
split once, which buys HF's own resumable download cache for free (a
`Server disconnected` mid-pull resumes from HF's cache on retry, independent
of this module's own `.pull_state.json` resume) -- see
`download_tiny_genimage.py`'s module docstring. `hf_parquet` is that path.

TWO LABEL SHAPES. A source config carries either a fixed `label` (every row
pulled gets the same one -- nano_banana is entirely AIGC) or a per-row
`label_field` + `label_map` (Tiny-GenImage and the OOD benchmark ship a
`label` column of their own, and `sid_set` uses the same shape to keep only
ONE of the source's own label values -- see sources.yaml's comment on why
`sid_set` filters to reals only). Both shapes reduce to the same normalized
0/1 by the time a row reaches :class:`~aigc_detect.data.fetchers.base.
IncrementalIndexWriter`.

QUOTA MODE. A config carrying `per_generator` is the `download_ood_benchmark.py`
shape: a generator-balanced slice with reals capped to
`per_generator * real_cap_multiplier`, an optional `only_generators` allow-list,
and a `skip_rows` starting offset -- see `sources.yaml`'s `aigc_bench_ext` /
`aigc_detect_bench` entries for why the offset is load-bearing (it is what
keeps the training slice and the eval tier disjoint by construction). Resuming
a quota pull reconstructs its per-generator counts by reading back the rows
already committed to `index.csv`, rather than serializing a `Counter` into
`.pull_state.json` -- the index is already the source of truth for "what has
been kept", so a second copy of the same fact is one more place for it to
drift from the first.
"""

from __future__ import annotations

import io
from collections import Counter

from aigc_detect.config import LABEL_AIGC, LABEL_REAL
from aigc_detect.data.fetchers.base import CorpusPaths, IncrementalIndexWriter, PullResult, PullState
from aigc_detect.data.relocate import _rel_posix
from aigc_detect.log import get_logger

logger = get_logger(__name__)

#: Datasets differ on which feature name carries the image; try the declared
#: `image_col` first, then this list -- mirrors `_find_image` in
#: download_aigc_modern.py / download_real_domains.py.
_IMAGE_KEY_FALLBACKS = ("image", "img", "jpg", "png", "photo")

_LABEL_NAME_TO_INT = {"real": LABEL_REAL, "aigc": LABEL_AIGC}


def _find_image(example: dict, image_col: str):
    from PIL import Image

    for key in (image_col, *_IMAGE_KEY_FALLBACKS):
        if key in example and example[key] is not None:
            v = example[key]
            if isinstance(v, dict) and v.get("bytes"):
                return Image.open(io.BytesIO(v["bytes"]))
            if hasattr(v, "convert"):
                return v
    return None


def _mean_saturation(img) -> float:
    import numpy as np

    a = np.asarray(img.resize((64, 64))).astype("float32")
    mx, mn = a.max(2), a.min(2)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def _reencode(img, dest_file, reencode: dict | None) -> None:
    img = img.convert("RGB")
    if reencode:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_file, reencode.get("format", "jpeg").upper(), quality=reencode.get("quality", 95))
    else:
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_file, "JPEG", quality=95)


class HFStreamingFetcher:
    def pull(self, source, dest: CorpusPaths, state: PullState) -> PullResult:
        cfg = source.config
        if "per_generator" in cfg:
            return _pull_quota(source, dest, state)
        return _pull_simple(source, dest, state)


class HFParquetFetcher:
    def pull(self, source, dest: CorpusPaths, state: PullState) -> PullResult:
        return _pull_parquet(source, dest, state)


def _row_label(example: dict, cfg: dict) -> int | None:
    """The normalized 0/1 label for one example, or None if this source keeps
    only some of the upstream label values (sid_set's reals-only filter)."""
    if "label" in cfg:
        return _LABEL_NAME_TO_INT[cfg["label"]]
    raw = example[cfg["label_field"]]
    mapped = cfg["label_map"].get(raw) or cfg["label_map"].get(str(raw))
    return None if mapped is None else _LABEL_NAME_TO_INT[mapped]


def _pull_simple(source, dest: CorpusPaths, state: PullState) -> PullResult:
    from datasets import load_dataset

    cfg = source.config
    cap = cfg.get("cap")
    quality_gate = cfg.get("quality_gate") or {}
    resumed = state.rows_written > 0
    writer = IncrementalIndexWriter(dest, state)

    ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True)
    scanned = state.rows_scanned
    if scanned:
        ds = ds.skip(scanned)
    kept = state.rows_written

    small = mono = bad = 0
    for example in ds:
        if cap is not None and kept >= cap:
            break
        scanned += 1

        img = _find_image(example, cfg.get("image_col", "image"))
        if img is None:
            bad += 1
            continue
        label = _row_label(example, cfg)
        if label is None:
            continue  # this upstream label value is not part of this source

        source_mode = img.mode
        try:
            img = img.convert("RGB")
        except Exception:
            bad += 1
            continue

        if quality_gate:
            if min(img.size) < quality_gate["min_side"]:
                small += 1
                continue
            # MONOCHROME IS COUNTED, NOT DROPPED. This used to `continue`, and
            # that was over-fitted to the incident that produced it: the
            # Depth Anything mirror was ~100% mode=L at saturation 0.000, a
            # CORPUS-level property, and the abort below is what catches it.
            # Discarding individual images contributed nothing to that catch
            # and cost real data -- on the first corpus pulled after the rule
            # existed it dropped 19 images, 16 monochrome AI art and 3 real
            # black-and-white photographs, and zero depth maps. Greyscale is
            # legitimate content a deployed detector meets; a training set
            # that excludes it is narrower than the world it is scored in.
            #
            # WHAT REPLACES THE DROP IS VISIBILITY, NOT NOTHING. Greyscale can
            # still become a LABEL-correlated cue, and the blind probe cannot
            # see that one -- probe.py converts to L, so colour is gone before
            # it looks, the same structural blindness that lets an aspect-ratio
            # shortcut through. `aigc corpus audit` reports saturation per
            # label so a skew is legible; see scripts/audit_corpora.py.
            if source_mode in {"L", "1", "I", "F", "I;16"} or _mean_saturation(img) < quality_gate["min_saturation"]:
                mono += 1
                if scanned >= quality_gate["min_scanned_before_abort"] and mono / scanned > quality_gate[
                    "max_mono_fraction"
                ]:
                    raise SystemExit(
                        f"[pull] ABORT: {mono:,}/{scanned:,} images from {cfg['repo']} are greyscale. "
                        f"A corpus that is mostly monochrome is probably not what it claims to be -- "
                        f"see sources.yaml's quality_gate comment before retrying.\n"
                        f"        {kept:,} rows are already indexed and are NOT trustworthy: discard "
                        f"them with `aigc pull run {source.id} --force` once the source is fixed."
                    )
            if max(img.size) > quality_gate["max_side"]:
                from PIL import Image

                s = quality_gate["max_side"] / max(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)

        prefix = cfg.get("filename_prefix", source.id)
        dest_file = dest.images / f"{prefix}_{kept:06d}.jpg"
        _reencode(img, dest_file, cfg.get("reencode"))
        writer.add({
            "image_path": _rel_posix(dest_file),
            "label": label,
            "source": cfg.get("source_name", source.id),
            "generator": cfg.get("generator") or "",
        })
        kept += 1
        writer.maybe_checkpoint(rows_scanned=scanned, cursor={})

    if not cap or kept < cap:
        completed = True  # exhausted the stream
    else:
        completed = kept >= cap
    writer.finish(rows_scanned=scanned, cursor={}, completed=completed)
    return PullResult(rows_written=state.rows_written, rows_scanned=state.rows_scanned,
                      resumed=resumed, completed=completed,
                      note=f"{small:,} too small, {mono:,} greyscale/mono (kept), {bad:,} unreadable, this run")


def _existing_quota_counts(dest: CorpusPaths, generator_field_present: bool) -> tuple[Counter, int]:
    """Reconstruct per-generator counts and the real count from what is
    already committed, so resuming a quota pull does not need its own copy
    of that bookkeeping in `.pull_state.json` -- see the module docstring."""
    import csv

    fake_quota: Counter = Counter()
    real_kept = 0
    if not dest.index_csv.is_file():
        return fake_quota, real_kept
    with open(dest.index_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["label"]) == LABEL_REAL:
                real_kept += 1
            else:
                fake_quota[row["generator"]] += 1
    return fake_quota, real_kept


def _pull_quota(source, dest: CorpusPaths, state: PullState) -> PullResult:
    from datasets import load_dataset

    cfg = source.config
    resumed = state.rows_written > 0
    writer = IncrementalIndexWriter(dest, state)

    ds = load_dataset(cfg["repo"], split=cfg["split"], streaming=True)
    if cfg.get("shuffle_buffer", 0) > 0:
        ds = ds.shuffle(seed=cfg.get("seed", 42), buffer_size=cfg["shuffle_buffer"])

    feats = getattr(ds, "features", None) or {}
    gen_field = cfg.get("generator_field", "generator")
    gen_names = feats[gen_field].names if gen_field in feats and hasattr(feats[gen_field], "names") else None

    def gen_of(ex):
        g = ex[gen_field]
        return gen_names[g] if (gen_names is not None and isinstance(g, int)) else str(g)

    per_generator = cfg["per_generator"]
    real_cap = per_generator * cfg.get("real_cap_multiplier", 16)
    only_generators = set(cfg["only_generators"]) if cfg.get("only_generators") else None
    skip_rows = cfg.get("skip_rows", 0)
    min_scan = cfg.get("min_scan", 0)
    max_scan = cfg.get("max_scan")

    fake_quota, real_kept = _existing_quota_counts(dest, gen_field in feats)
    scanned = state.rows_scanned
    if scanned:
        ds = ds.skip(scanned)

    for example in ds:
        scanned += 1
        if scanned <= skip_rows:
            continue
        if max_scan is not None and scanned > max_scan:
            break

        label = _row_label(example, cfg)
        generator = gen_of(example)
        if only_generators is not None and generator not in only_generators:
            continue

        if label == LABEL_REAL:
            if real_kept >= real_cap:
                continue
            real_kept += 1
        else:
            if fake_quota[generator] >= per_generator:
                continue
            fake_quota[generator] += 1

        img = _find_image(example, cfg.get("image_col", "image"))
        if img is None:
            continue

        prefix = cfg.get("filename_prefix", source.id)
        gen_dir = dest.images / generator
        dest_file = gen_dir / f"{prefix}_{scanned:07d}.jpg"
        _reencode(img, dest_file, cfg.get("reencode"))
        writer.add({
            "image_path": _rel_posix(dest_file),
            "label": label,
            "source": cfg.get("source_name", source.id),
            "generator": generator,
        })
        writer.maybe_checkpoint(rows_scanned=scanned, cursor={})

        if (
            scanned >= min_scan
            and real_kept >= real_cap
            and fake_quota
            and all(v >= per_generator for v in fake_quota.values())
        ):
            break

    completed = True
    writer.finish(rows_scanned=scanned, cursor={}, completed=completed)
    return PullResult(rows_written=state.rows_written, rows_scanned=state.rows_scanned,
                      resumed=resumed, completed=completed,
                      note=f"per-generator: {dict(sorted(fake_quota.items()))}, real: {real_kept:,}")


def _pull_parquet(source, dest: CorpusPaths, state: PullState) -> PullResult:
    from datasets import load_dataset

    cfg = source.config
    resumed = state.rows_written > 0
    writer = IncrementalIndexWriter(dest, state)

    ds = load_dataset(cfg["repo"], split=cfg["split"])  # non-streaming: HF's own resumable cache
    label_names = ds.features[cfg["label_field"]].names if hasattr(ds.features[cfg["label_field"]], "names") else None
    gen_field = cfg.get("generator_field")
    gen_names = (
        ds.features[gen_field].names if gen_field and hasattr(ds.features[gen_field], "names") else None
    )

    start = state.rows_written  # index-addressable, so resume is just "start later"
    prefix = cfg.get("filename_prefix", source.id)
    for idx in range(start, len(ds)):
        example = ds[idx]
        raw_label = example[cfg["label_field"]]
        label_name = label_names[raw_label] if label_names is not None else raw_label
        label = _LABEL_NAME_TO_INT.get(str(label_name).lower(), cfg["label_map"].get(raw_label))
        generator = ""
        if gen_field:
            raw_gen = example[gen_field]
            generator = gen_names[raw_gen] if gen_names is not None else str(raw_gen)

        gen_dir = dest.images / (generator or "_")
        dest_file = gen_dir / f"{prefix}_{idx:06d}.jpg"
        _reencode(example[cfg.get("image_col", "image")], dest_file, cfg.get("reencode"))
        writer.add({
            "image_path": _rel_posix(dest_file),
            "label": label,
            "source": cfg.get("source_name", source.id),
            "generator": generator,
        })
        writer.maybe_checkpoint(rows_scanned=idx + 1, cursor={})

    writer.finish(rows_scanned=len(ds), cursor={}, completed=True)
    return PullResult(rows_written=state.rows_written, rows_scanned=state.rows_scanned,
                      resumed=resumed, completed=True)
