"""Fold the legacy .npz view caches into the content-addressed store.

The old layout is one .npz per (backbone, manifest-selection, view), holding
N x dim floats under a single fingerprint over absolute image paths. Those
files represent close to two million forward passes of real GPU time, and every
one of them is still valid work -- only its *addressing* was wrong. Migration
re-derives each row's identity by hashing the file it came from and inserts the
vectors unchanged.

WHAT MIGRATES AND WHAT DOES NOT.

Deterministic views (`clean`, `jpeg_*`, `blur_*`, `resize_*`, `center_crop_80`,
`chain_light`) migrate as-is: the transform is a pure function of the pixels, so
the cached vector is exactly what the new pipeline would compute.

Stochastic views (`noise_*`, `color_jitter`, `chain_medium`, `chain_heavy`, the
`trainchain_*` set) do NOT migrate. Their per-image RNG seed used to be derived
from the image's absolute path, which is the same defect as the cache key: a
rename silently changed the noise draw. Seeding on the content id fixes it, and
that necessarily makes the old vectors unreproducible -- they were drawn from a
different seed. They are recomputed on first use, per manifest, rather than in
one eager batch.

Mixing the two schemes was considered and rejected. `scripts/run_race.py`
depends on every backbone facing byte-identical per-(image, view) degradations;
a store holding some path-seeded and some content-seeded rows for one view would
break that fairness property silently, which is precisely the class of fault
this project has been bitten by before.

Plain `embed.py` caches (no view dimension) are also skipped. They are 2.5% of
the total and the view-based path has superseded them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aigc_detect.cache.hashing import HashCache
from aigc_detect.cache.store import EmbeddingStore, backbone_id, view_id
from aigc_detect.registry.backbones import BACKBONE_REGISTRY


def _revision_for(backbone_key: str, checkpoint: str) -> str | None:
    """The pinned revision from the registry, if this checkpoint is the pinned one.

    The legacy .npz recorded which checkpoint produced it but never a revision,
    so we recover it from the registry -- and only when the checkpoint string
    still matches, otherwise the entry has been repointed since and we must not
    claim the vectors came from the pinned weights.
    """
    entry = BACKBONE_REGISTRY.get(backbone_key)
    if not entry or entry.get("checkpoint") != checkpoint:
        return None
    return entry.get("revision")


def scan(embeddings_dir: str | Path) -> dict:
    """Inventory the legacy cache without touching it."""
    det: list[Path] = []
    sto: list[Path] = []
    plain: list[Path] = []
    rows_det = rows_sto = rows_plain = 0

    for path in sorted(Path(embeddings_dir).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=True) as d:
                if "view" not in d:
                    plain.append(path)
                    rows_plain += int(d["n_rows"]) if "n_rows" in d else len(d["labels"])
                    continue
                spec = str(d["view_spec"]) if "view_spec" in d else ""
                n = int(d["n_rows"]) if "n_rows" in d else len(d["labels"])
                has_paths = "image_paths" in d
        except Exception as exc:  # a corrupt cache file must not abort the inventory
            print(f"[migrate] unreadable, skipping: {path.name} ({type(exc).__name__}: {exc})")
            continue

        if not has_paths:
            # Written before image_paths was recorded; identity is unrecoverable.
            print(f"[migrate] no image_paths, cannot migrate: {path.name}")
            continue
        if "|stochastic" in spec:
            sto.append(path)
            rows_sto += n
        else:
            det.append(path)
            rows_det += n

    return {
        "deterministic": det, "stochastic": sto, "plain": plain,
        "rows_deterministic": rows_det, "rows_stochastic": rows_sto, "rows_plain": rows_plain,
    }


def migrate(
    embeddings_dir: str | Path,
    store: EmbeddingStore,
    hashes: HashCache,
    *,
    dry_run: bool = False,
    limit_files: int | None = None,
) -> dict:
    """Insert every migratable legacy row into `store`.

    Idempotent: `put_batch` skips ids already present, so an interrupted
    migration is resumed by re-running it.
    """
    inv = scan(embeddings_dir)
    files = inv["deterministic"][: limit_files] if limit_files else inv["deterministic"]

    print(
        f"[migrate] {len(inv['deterministic'])} deterministic files ({inv['rows_deterministic']:,} rows) "
        f"-> migrating\n"
        f"[migrate] {len(inv['stochastic'])} stochastic files ({inv['rows_stochastic']:,} rows) "
        f"-> recompute on first use (path-seeded, see module docstring)\n"
        f"[migrate] {len(inv['plain'])} plain embed.py files ({inv['rows_plain']:,} rows) -> skipped"
    )
    if dry_run:
        return {"migrated_rows": 0, "files": len(files), "dry_run": True}

    written = 0
    for i, path in enumerate(files, 1):
        with np.load(path, allow_pickle=True) as d:
            key = str(d["backbone"])
            checkpoint = str(d["checkpoint"])
            dim = int(d["pooled_dim"])
            res = int(d["native_res"])
            norm_mean = [float(x) for x in np.atleast_1d(d["norm_mean"])]
            norm_std = [float(x) for x in np.atleast_1d(d["norm_std"])]
            view = str(d["view"])
            spec = str(d["view_spec"])
            paths = [str(p) for p in d["image_paths"]]
            vectors = d["embeddings"].astype(np.float32)

        revision = _revision_for(key, checkpoint)
        bb = backbone_id(key, checkpoint, revision, dim, res, norm_mean, norm_std)
        vid = view_id(view, spec, None)
        store.register_backbone(bb, key=key, checkpoint=checkpoint, revision=revision,
                                dim=dim, native_res=res, norm_mean=norm_mean, norm_std=norm_std)
        store.register_view(vid, name=view, spec=spec, seed_scheme=None)

        ids = hashes.ids_for(paths)
        n = store.put_batch(bb, vid, ids, vectors)
        written += n
        print(f"[migrate] {i:>3}/{len(files)}  {path.name:<62} +{n:>6} rows")

    return {"migrated_rows": written, "files": len(files),
            "deferred_rows": inv["rows_stochastic"], "skipped_rows": inv["rows_plain"]}
