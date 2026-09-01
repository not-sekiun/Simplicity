"""Precompute and cache frozen-backbone pooled embeddings for a manifest.

Per the "Simplicity Prevails" recipe (arXiv:2602.01738), preprocessing is
"resized and center-cropped to the native resolution of each model" with no
additional augmentation -- see aigc_detect.data.transforms.build_backbone_transform
for the aspect-preserving resize + center-crop tail already used elsewhere in
this project, reused here at each backbone's own native resolution.

Cache location: data/embeddings/<backbone_key>__<manifest_stem>.npz
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from aigc_detect.config import EMBEDDINGS_DIR
from aigc_detect.data.dataset import ManifestImageDataset
from aigc_detect.data.transforms import build_backbone_transform
from aigc_detect.registry.backbones import load_backbone


def embeddings_path(backbone_key: str, manifest_path: str | Path) -> Path:
    manifest_stem = Path(manifest_path).stem
    return EMBEDDINGS_DIR / f"{backbone_key}__{manifest_stem}.npz"


def fingerprint_paths(paths) -> str:
    """SHA1 over an ordered sequence of image paths.

    Shared by ``manifest_fingerprint`` (whole manifest) and the per-view
    embedder (which fingerprints the *selected subset* of rows, since a
    stratified sample is not a prefix of the manifest).
    """
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\n")
    return h.hexdigest()


def manifest_fingerprint(manifest_path: str | Path, limit: int | None = None) -> str:
    """SHA1 over the manifest's image_path column (in order), so a cached .npz
    can be invalidated when the manifest changes.

    The cache filename is only <backbone>__<manifest stem>.npz, and re-running
    `main.py split` with different flags rewrites train.csv/val.csv in place --
    same filename, different images. Without this check a stale cache would be
    silently reused and every downstream metric would be wrong.
    """
    df = pd.read_csv(manifest_path, usecols=["image_path"])
    if limit is not None and limit < len(df):
        df = df.iloc[:limit]
    return fingerprint_paths(df["image_path"])


def precompute_embeddings(
    manifest_path: str | Path,
    backbone_key: str,
    out_path: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    force: bool = False,
    limit: int | None = None,
) -> Path:
    manifest_path = Path(manifest_path)
    out_path = Path(out_path) if out_path is not None else embeddings_path(backbone_key, manifest_path)

    fingerprint = manifest_fingerprint(manifest_path, limit)
    if out_path.exists() and not force:
        try:
            cached = np.load(out_path, allow_pickle=True)
            cached_fp = str(cached["manifest_fingerprint"]) if "manifest_fingerprint" in cached else None
        except Exception:
            cached_fp = None
        if cached_fp == fingerprint:
            print(f"[embed] {out_path} already exists and matches the manifest -- skipping")
            return out_path
        reason = "no fingerprint (written before this check existed)" if cached_fp is None else "manifest changed"
        print(f"[embed] {out_path} is STALE ({reason}) -- recomputing")

    module, pooled_dim, native_res = load_backbone(backbone_key)
    print(
        f"[embed] backbone={backbone_key} native_res={native_res} pooled_dim={pooled_dim} "
        f"norm_source={module.norm_source} mean={module.norm_mean} std={module.norm_std}"
    )

    transform = v2.Compose(
        [
            *build_backbone_transform(native_res),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=module.norm_mean, std=module.norm_std),
        ]
    )

    ds = ManifestImageDataset(manifest_path, transform=transform)
    if limit is not None and limit < len(ds):
        ds.df = ds.df.iloc[:limit].reset_index(drop=True)
        print(f"[embed] --limit {limit} applied -- using first {len(ds)} of the manifest's rows")

    sources = ds.df["source"].tolist()
    n = len(ds)
    print(f"[embed] manifest={manifest_path} rows={n}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = next(module.parameters()).device
    use_amp = device.type == "cuda"

    all_embeddings = np.empty((n, pooled_dim), dtype=np.float32)
    all_labels = np.empty((n,), dtype=np.int64)

    offset = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[embed] {backbone_key}"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                feats = module(images)
            feats = feats.float().cpu().numpy()
            bsz = feats.shape[0]
            all_embeddings[offset : offset + bsz] = feats
            all_labels[offset : offset + bsz] = labels.numpy()
            offset += bsz

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=all_embeddings,
        labels=all_labels,
        sources=np.array(sources, dtype=str),
        backbone=backbone_key,
        checkpoint=module.checkpoint_used,
        native_res=native_res,
        pooled_dim=pooled_dim,
        manifest_path=str(manifest_path),
        n_rows=n,
        manifest_fingerprint=fingerprint,
        norm_mean=module.norm_mean,
        norm_std=module.norm_std,
        norm_source=module.norm_source,
    )
    print(f"[embed] saved embeddings {all_embeddings.shape} -> {out_path}")
    return out_path
