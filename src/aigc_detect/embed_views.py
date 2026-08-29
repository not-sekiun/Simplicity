"""Precompute frozen-backbone embeddings for every robustness view of a manifest.

This is the measuring instrument for the robustness half of the competition
score (`0.5*AUC_clean + 0.5*AUC_robust`). `embed.py` caches one embedding per
image; this caches **eighteen** -- one per view in `build_robustness_views()`:
"clean", the 14 single-transform rows of the brief's 5.2 table, and 3 chained
rows that compose transforms the way a real redistribution path does.

A degraded image is a *different image* to a frozen backbone: there is no way
to derive the blurred embedding from the clean one, so each view costs its own
forward pass. What this module avoids is paying for each view's *decode*.

    naive:  for view in views: for image in manifest: decode -> transform -> fwd
            => 18 decodes per image, 17 of them producing an identical PIL image

    here:   for image in manifest: decode once -> 18 transforms -> 18 fwds
            => 1 decode per image; the GPU work is unchanged and irreducible

Measured on this machine the pipeline is GPU-bound, not decode-bound (~84
forward passes/s at both batch 4 and batch 16, matching embed.py's 82 img/s),
so the single decode is not a speedup. It is kept because it is a *correctness*
property: every view of an image provably derives from identical source pixels,
and one pass writes every view's cache, so no view can be quietly out of date
relative to its siblings.

Cache location: data/embeddings/<backbone>__<stem>__<view>.npz, where <stem> is
the manifest stem plus a subsample tag (see `cache_stem`).

DETERMINISM (FINDINGS trap 10). Several views are random by construction:
`color_jitter` samples fresh brightness/contrast/saturation factors per call,
the noise views draw from the torch RNG, and two of the chains contain both.
For a cached eval set that is a correctness problem -- re-running would score a
different test, and two backbones raced against each other would not face the
same images.

Every stochastic view is therefore seeded, and seeded **on the image's path**,
not on its row index (`SEED_SCHEME`). Index seeding was the obvious choice and
is wrong for the workflow this module is actually used in: a 2,000-row
stratified subsample gives every image a different row index than the full
manifest does, so the same photo would get a different noise realization in the
subsample than in the full run. The subsample's numbers would then differ from
the full run's for a reason that looks exactly like a real effect. Keyed on the
path, an image's degradations are the same wherever it appears.

CACHE KEYING (FINDINGS trap 7, extended twice).

  1. *Which images.* `manifest_fingerprint` is not enough once subsampling
     exists -- a stratified sample is not a prefix of the manifest -- so the
     fingerprint is taken over the **selected rows**, in their final order.
  2. *What each view means.* "blur_sigma1.0" is a *name*: editing BLUR_SIGMAS
     would silently change what it refers to while the cache file keeps
     loading. Each view's canonical spec string (from `build_robustness_views`)
     is hashed into its .npz as `view_fingerprint` and checked on reuse.

The view fingerprint is per-view rather than one hash over the whole 5.2 table,
so adding the chained views -- or later retuning one severity -- invalidates
only the views that actually changed, not all eighteen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc_detect.backbones import load_backbone
from aigc_detect.config import EMBEDDINGS_DIR, RANDOM_SEED, ROOT_DIR
from aigc_detect.dataset import ManifestImageDataset
from aigc_detect.embed import fingerprint_paths
from aigc_detect.transforms import (
    build_robustness_views,
    eval_view_names,
    train_chain_view_names,
)

# Bumped only if the *seeding scheme* changes. Folded into the fingerprint of
# stochastic views only -- a deterministic view's output does not depend on it,
# so bumping this must not invalidate `clean`, `jpeg_*`, `blur_*` etc.
SEED_SCHEME = "path-v1"


def cache_stem(manifest_path: str | Path, limit: int | None = None, sample_rows: int | None = None) -> str:
    """Cache filename stem for a (manifest, row-selection) pair.

    The selection is part of the *identity* of the cache, not just its
    contents: without a tag, a 2,000-row subsample and the full manifest write
    to the same filename. The fingerprint check would catch that and recompute,
    so nothing would be wrong -- but alternating between the two would thrash,
    each run destroying the other's cache. Tagging keeps both on disk.
    """
    stem = Path(manifest_path).stem
    if sample_rows is not None:
        return f"{stem}-s{sample_rows}"
    if limit is not None:
        return f"{stem}-l{limit}"
    return stem


def view_embeddings_path(backbone_key: str, stem: str, view: str) -> Path:
    return EMBEDDINGS_DIR / f"{backbone_key}__{stem}__{view}.npz"


def view_fingerprint(spec: str) -> str:
    """SHA1 over one view's canonical spec string (see build_robustness_views).

    Stochastic views additionally commit to the seeding scheme, because for
    those the seed is part of what the cached numbers mean.
    """
    payload = spec
    if "|stochastic" in spec:
        payload = f"{spec}|seed={SEED_SCHEME}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _view_seed(image_path: str, view: str) -> int:
    """Stable per-(image, view) seed. Derived from a hash of the path rather
    than from the row index, so an image's degradations do not change when it
    appears in a different subsample -- see the module docstring."""
    h = hashlib.sha1(f"{image_path}::{view}::{SEED_SCHEME}".encode("utf-8", "replace")).digest()
    return int.from_bytes(h[:4], "little")


def stratified_sample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Label-balanced, source-proportional subsample of `n` rows.

    `--limit N` takes the manifest's *first* N rows, which is not a valid eval
    set: the manifest is ordered by however the split was written, so a prefix
    can be arbitrarily skewed in both label and source. This draws n/2 per
    label and allocates each label's quota across sources in proportion to that
    source's share, so the subsample is a miniature of the manifest.

    Rows are returned sorted by image_path, so the row *order* -- and therefore
    the fingerprint -- depends only on which rows were chosen, not on the
    internals of how they were chosen.
    """
    if n >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    per_label = n // 2
    parts = []

    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        want = min(per_label, len(sub))
        groups = [(src, g) for src, g in sub.groupby("source", sort=True)]
        sizes = np.array([len(g) for _, g in groups], dtype=float)
        alloc = np.floor(sizes / sizes.sum() * want).astype(int)

        # Hand out the rounding remainder to the largest groups that still have
        # room, so the allocation sums to exactly `want`.
        while alloc.sum() < want:
            room = sizes - alloc
            if room.max() <= 0:
                break
            alloc[int(np.argmax(np.where(room > 0, sizes, -1.0)))] += 1

        for (_src, g), k in zip(groups, alloc):
            if k > 0:
                picked = np.sort(rng.choice(len(g), size=int(k), replace=False))
                parts.append(g.iloc[picked])

    return pd.concat(parts).sort_values("image_path", kind="mergesort").reset_index(drop=True)


def select_rows(
    manifest_path: str | Path,
    limit: int | None = None,
    sample_rows: int | None = None,
    sample_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Resolve a manifest plus row-selection flags into the exact DataFrame to
    embed. `sample_rows` wins if both are given."""
    df = ManifestImageDataset(manifest_path).df
    if sample_rows is not None:
        return stratified_sample(df, sample_rows, sample_seed)
    if limit is not None:
        return df.iloc[:limit].reset_index(drop=True)
    return df


class MultiViewDataset(Dataset):
    """Yields (views, label) where views is a (n_views, 3, S, S) tensor built
    from a SINGLE decode of the source image."""

    def __init__(self, df: pd.DataFrame, pipelines: dict):
        self.df = df.reset_index(drop=True)
        self.view_names = list(pipelines.keys())
        self.pipelines = pipelines

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        raw_path = str(row["image_path"])
        p = Path(raw_path)
        img = Image.open(p if p.is_absolute() else ROOT_DIR / p).convert("RGB")  # ONE decode

        tensors = []
        for name in self.view_names:
            # Seeded per (image path, view): color_jitter, the noise views and
            # two of the chains are random by construction, and a cached eval
            # set must be reproducible across runs, workers, backbones -- and
            # across subsamples (see the module docstring).
            torch.manual_seed(_view_seed(raw_path, name))
            tensors.append(self.pipelines[name](img))
        return torch.stack(tensors), int(row["label"])


def precompute_view_embeddings(
    manifest_path: str | Path,
    backbone_key: str,
    views: list[str] | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    force: bool = False,
    limit: int | None = None,
    sample_rows: int | None = None,
    sample_seed: int = RANDOM_SEED,
    include_train_chains: bool = False,
    dtype: str = "float16",
) -> list[Path]:
    """Embed every requested view of ``manifest_path`` in one pass over the data.

    ``batch_size`` is in IMAGES, not forward-pass rows -- the effective GPU batch
    is batch_size * n_views. The default of 8 gives 144 images per forward at 18
    views, which is the right order for a 336px ViT on a 3080.

    ``sample_rows`` draws a label-balanced, source-proportional subsample (see
    ``stratified_sample``) and tags the cache filenames with it. This is the
    intended path for racing backbones: at ~2,000 rows an AUC has a standard
    error near +/-0.005-0.01, far tighter than the gaps between backbones the
    grid is meant to resolve, and it costs minutes instead of hours per
    backbone. Keep every *view* -- backbones are reported to fail on different
    transforms, so dropping views removes the very signal being measured.

    Embeddings are stored as float16 by default. Not an approximation here: the
    forward pass runs under AMP, so the values already carry fp16 precision and
    the halved file size is free (verified: max |float32 - float16| == 0).
    """
    manifest_path = Path(manifest_path)
    module, pooled_dim, native_res = load_backbone(backbone_key)
    print(
        f"[embed-views] backbone={backbone_key} native_res={native_res} pooled_dim={pooled_dim} "
        f"norm_source={module.norm_source} mean={module.norm_mean} std={module.norm_std}"
    )

    # Per-backbone resolution and normalization stats -- NOT config's defaults.
    # See FINDINGS trap 9: the builders default to 224 + ImageNet stats, which
    # would evaluate every view under normalization the model never saw.
    all_pipelines, all_specs = build_robustness_views(
        image_size=native_res, norm_mean=module.norm_mean, norm_std=module.norm_std
    )
    if views is not None:
        unknown = set(views) - set(all_pipelines)
        if unknown:
            raise KeyError(f"Unknown view(s) {sorted(unknown)}. Available: {list(all_pipelines)}")
        all_pipelines = {k: all_pipelines[k] for k in views}
    else:
        # Default to the 18 SCORED views. The trainchain_* views are
        # augmentation material for the train manifest only -- caching them for
        # val/demo-val would be pure waste, and they must never enter
        # AUC_robust.
        keep = set(eval_view_names()) | (set(train_chain_view_names()) if include_train_chains else set())
        all_pipelines = {k: v for k, v in all_pipelines.items() if k in keep}

    df = select_rows(manifest_path, limit=limit, sample_rows=sample_rows, sample_seed=sample_seed)
    stem = cache_stem(manifest_path, limit=limit, sample_rows=sample_rows)
    m_fp = fingerprint_paths(df["image_path"])
    n = len(df)

    if sample_rows is not None:
        counts = df["label"].value_counts().to_dict()
        print(
            f"[embed-views] --sample-rows {sample_rows} (seed {sample_seed}) -> {n} rows, "
            f"label counts {counts}, sources {df['source'].value_counts().to_dict()}"
        )
    elif limit is not None:
        print(f"[embed-views] --limit {limit} applied -- using first {n} of the manifest's rows")

    # Skip views already cached against BOTH fingerprints; recompute the rest.
    todo: dict = {}
    for name, pipe in all_pipelines.items():
        v_fp = view_fingerprint(all_specs[name])
        out = view_embeddings_path(backbone_key, stem, name)
        if out.exists() and not force:
            try:
                with np.load(out, allow_pickle=True) as d:
                    ok = (
                        str(d["manifest_fingerprint"]) == m_fp
                        and "view_fingerprint" in d
                        and str(d["view_fingerprint"]) == v_fp
                    )
            except Exception:  # noqa: BLE001 - unreadable cache is treated as stale
                ok = False
            if ok:
                print(f"[embed-views] {out.name} matches the manifest and view spec -- skipping")
                continue
            print(f"[embed-views] {out.name} is STALE -- recomputing")
        todo[name] = pipe

    if not todo:
        print("[embed-views] every requested view is already cached and current -- nothing to do")
        return [view_embeddings_path(backbone_key, stem, v) for v in all_pipelines]

    view_names = list(todo)
    ds = MultiViewDataset(df, todo)
    print(f"[embed-views] manifest={manifest_path} stem={stem} rows={n} views={len(view_names)}")
    print(f"[embed-views] computing: {', '.join(view_names)}")
    print(f"[embed-views] {n * len(view_names):,} forward passes from {n:,} decodes")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = next(module.parameters()).device
    use_amp = device.type == "cuda"
    np_dtype = np.float16 if dtype == "float16" else np.float32

    out_arrays = {v: np.empty((n, pooled_dim), dtype=np_dtype) for v in view_names}
    all_labels = np.empty((n,), dtype=np.int64)

    offset = 0
    with torch.no_grad():
        for batched_views, labels in tqdm(loader, desc=f"[embed-views] {backbone_key}"):
            b, v = batched_views.shape[0], batched_views.shape[1]
            flat = batched_views.reshape(b * v, *batched_views.shape[2:]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                feats = module(flat)
            feats = feats.float().cpu().numpy().reshape(b, v, pooled_dim)
            for i, name in enumerate(view_names):
                out_arrays[name][offset : offset + b] = feats[:, i, :].astype(np_dtype)
            all_labels[offset : offset + b] = labels.numpy()
            offset += b

    sources = np.array(ds.df["source"].tolist(), dtype=str)
    # Tiny-GenImage carries a per-image generator tag; keep it so the grid can
    # be broken down by generator without re-reading the manifest.
    generators = (
        np.array(ds.df["generator"].astype(str).tolist(), dtype=str)
        if "generator" in ds.df.columns
        else np.array([""] * n, dtype=str)
    )
    paths = np.array(ds.df["image_path"].astype(str).tolist(), dtype=str)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name in view_names:
        out = view_embeddings_path(backbone_key, stem, name)
        np.savez(
            out,
            embeddings=out_arrays[name],
            labels=all_labels,
            sources=sources,
            generators=generators,
            image_paths=paths,
            view=name,
            view_spec=all_specs[name],
            backbone=backbone_key,
            checkpoint=module.checkpoint_used,
            native_res=native_res,
            pooled_dim=pooled_dim,
            manifest_path=str(manifest_path),
            cache_stem=stem,
            n_rows=n,
            sample_rows=-1 if sample_rows is None else sample_rows,
            sample_seed=sample_seed,
            manifest_fingerprint=m_fp,
            view_fingerprint=view_fingerprint(all_specs[name]),
            seed_scheme=SEED_SCHEME,
            norm_mean=module.norm_mean,
            norm_std=module.norm_std,
            norm_source=module.norm_source,
        )
        written.append(out)
    total_mb = sum(p.stat().st_size for p in written) / 1e6
    print(f"[embed-views] wrote {len(written)} views x {n} rows ({total_mb:.0f} MB) -> {EMBEDDINGS_DIR}")
    return written
