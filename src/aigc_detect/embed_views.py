"""Precompute frozen-backbone embeddings for every robustness view of a manifest.

This is the measuring instrument for the robustness half of the competition
score (`0.5*AUC_clean + 0.5*AUC_robust`). `embed.py` caches one embedding per
image; this caches **fifteen** -- one per (transform, severity) pipeline in
`build_robustness_eval_transforms()`, which implements the brief's 5.2 table.

A degraded image is a *different image* to a frozen backbone: there is no way
to derive the blurred embedding from the clean one, so each view costs its own
forward pass. What this module avoids is paying for each view's *decode*.

    naive:  for view in views: for image in manifest: decode -> transform -> fwd
            => 15 decodes per image, 14 of them producing an identical PIL image

    here:   for image in manifest: decode once -> 15 transforms -> 15 fwds
            => 1 decode per image; the GPU work is unchanged and irreducible

The three noise views are close to free on top of that: noise is the only
tensor-domain transform in the table, so `clean` and `noise_sigma*` share the
entire resize/crop chain (see transforms.py's note on ordering -- FINDINGS trap 8).

Cache location: data/embeddings/<backbone>__<manifest stem>__<view>.npz

DETERMINISM (FINDINGS trap 10). Two views are random by construction:
`color_jitter` samples fresh brightness/contrast/saturation factors per call,
and the noise views draw from the global torch RNG. For a cached eval set that
is a correctness problem -- re-running would score a different test, and two
backbones raced against each other would not face the same images. Every view
is therefore seeded per (row index, view name), so the cache is reproducible
across runs, dataloader workers, and backbones.

CACHE KEYING (FINDINGS trap 7, extended). The manifest fingerprint alone is not
enough here: "blur_sigma1.0" is a *name*, and editing BLUR_SIGMAS in
transforms.py would silently change what that name refers to while the cache
file keeps working. The 5.2 parameter table is hashed into every .npz as
`transform_fingerprint` and checked on reuse.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc_detect.backbones import load_backbone
from aigc_detect.config import EMBEDDINGS_DIR, ROOT_DIR
from aigc_detect.dataset import ManifestImageDataset
from aigc_detect.embed import manifest_fingerprint
from aigc_detect.transforms import (
    BLUR_SIGMAS,
    CENTER_CROP_FRACTION,
    COLOR_JITTER_STRENGTH,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    build_robustness_eval_transforms,
)


def view_embeddings_path(backbone_key: str, manifest_path: str | Path, view: str) -> Path:
    return EMBEDDINGS_DIR / f"{backbone_key}__{Path(manifest_path).stem}__{view}.npz"


def transform_fingerprint() -> str:
    """SHA1 over the 5.2 parameter table, so a cached view is invalidated if the
    parameters behind its name change. See the module docstring."""
    table = (
        f"jpeg={JPEG_QUALITIES}|blur={BLUR_SIGMAS}|resize={RESIZE_SCALES}|"
        f"noise={NOISE_SIGMAS}|jitter={COLOR_JITTER_STRENGTH}|crop={CENTER_CROP_FRACTION}"
    )
    return hashlib.sha1(table.encode("utf-8")).hexdigest()


def _view_seed(idx: int, view: str) -> int:
    """Stable per-(row, view) seed. Derived from a hash rather than idx*k so that
    adjacent rows do not get correlated noise realizations."""
    h = hashlib.sha1(f"{idx}:{view}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


class MultiViewDataset(Dataset):
    """Yields (views, label) where views is a (n_views, 3, S, S) tensor built
    from a SINGLE decode of the source image."""

    def __init__(self, manifest_csv: str | Path, pipelines: dict, limit: int | None = None):
        base = ManifestImageDataset(manifest_csv)
        self.df = base.df.iloc[:limit].reset_index(drop=True) if limit is not None else base.df
        self.view_names = list(pipelines.keys())
        self.pipelines = pipelines

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = Path(row["image_path"])
        img = Image.open(p if p.is_absolute() else ROOT_DIR / p).convert("RGB")  # ONE decode

        tensors = []
        for name in self.view_names:
            # Seeded per (row, view): color_jitter and the noise views are random
            # by construction and a cached eval set must be reproducible.
            torch.manual_seed(_view_seed(idx, name))
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
    dtype: str = "float16",
) -> list[Path]:
    """Embed every requested view of ``manifest_path`` in one pass over the data.

    ``batch_size`` is in IMAGES, not forward-pass rows -- the effective GPU batch
    is batch_size * n_views. The default of 8 gives 120 images per forward at 15
    views, which is the right order for a 336px ViT on a 3080.

    Embeddings are stored as float16 by default: a ~1k-parameter linear probe
    cannot tell the difference, and it halves what is otherwise ~850MB per
    (backbone, manifest) pair.
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
    all_pipelines = build_robustness_eval_transforms(
        image_size=native_res, norm_mean=module.norm_mean, norm_std=module.norm_std
    )
    if views is not None:
        unknown = set(views) - set(all_pipelines)
        if unknown:
            raise KeyError(f"Unknown view(s) {sorted(unknown)}. Available: {list(all_pipelines)}")
        all_pipelines = {k: all_pipelines[k] for k in views}

    m_fp = manifest_fingerprint(manifest_path, limit)
    t_fp = transform_fingerprint()

    # Skip views already cached against BOTH fingerprints; recompute the rest.
    todo: dict = {}
    for name, pipe in all_pipelines.items():
        out = view_embeddings_path(backbone_key, manifest_path, name)
        if out.exists() and not force:
            try:
                with np.load(out, allow_pickle=True) as d:
                    ok = (
                        str(d["manifest_fingerprint"]) == m_fp
                        and "transform_fingerprint" in d
                        and str(d["transform_fingerprint"]) == t_fp
                    )
            except Exception:  # noqa: BLE001 - unreadable cache is treated as stale
                ok = False
            if ok:
                print(f"[embed-views] {out.name} matches the manifest and transform table -- skipping")
                continue
            print(f"[embed-views] {out.name} is STALE -- recomputing")
        todo[name] = pipe

    if not todo:
        print("[embed-views] every requested view is already cached and current -- nothing to do")
        return [view_embeddings_path(backbone_key, manifest_path, v) for v in all_pipelines]

    view_names = list(todo)
    ds = MultiViewDataset(manifest_path, todo, limit=limit)
    n = len(ds)
    if limit is not None:
        print(f"[embed-views] --limit {limit} applied -- using first {n} of the manifest's rows")
    print(f"[embed-views] manifest={manifest_path} rows={n} views={len(view_names)}")
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

    sources = ds.df["source"].tolist()
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name in view_names:
        out = view_embeddings_path(backbone_key, manifest_path, name)
        np.savez(
            out,
            embeddings=out_arrays[name],
            labels=all_labels,
            sources=np.array(sources, dtype=str),
            view=name,
            backbone=backbone_key,
            checkpoint=module.checkpoint_used,
            native_res=native_res,
            pooled_dim=pooled_dim,
            manifest_path=str(manifest_path),
            n_rows=n,
            manifest_fingerprint=m_fp,
            transform_fingerprint=t_fp,
            norm_mean=module.norm_mean,
            norm_std=module.norm_std,
            norm_source=module.norm_source,
        )
        written.append(out)
    total_mb = sum(p.stat().st_size for p in written) / 1e6
    print(f"[embed-views] wrote {len(written)} views x {n} rows ({total_mb:.0f} MB) -> {EMBEDDINGS_DIR}")
    return written
