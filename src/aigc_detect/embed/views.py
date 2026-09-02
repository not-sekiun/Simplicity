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

WHERE THE VECTORS LIVE (Tier 4). The content-addressed store is authoritative:
every vector is written to the store under (backbone id, view id, image content
id), and `data/embeddings/<backbone>__<stem>__<view>.npz` is a *projection* of
it in manifest order, kept because every reader in the project still loads that
shape. Three consequences follow, and they are the point of the tier:

  - RESUMABLE. Each batch is committed durably before the next one runs, so a
    killed run restarts by asking the store which ids are absent. It does not
    begin again.
  - INCREMENTAL. Re-encode one image and exactly one content id changes, so
    exactly one forward pass runs. The .npz alone could only say "this whole
    file is stale".
  - FREE RE-PROJECTION. Rebuilding an .npz whose rows are all in the store costs
    a gather, not a GPU. Moving the repo, or rebuilding a split, no longer costs
    hours -- which is what makes the Tier 5 corpus move affordable at all.

DETERMINISM (FINDINGS trap 10). Several views are random by construction:
`color_jitter` samples fresh brightness/contrast/saturation factors per call,
the noise views draw from the torch RNG, and two of the chains contain both.
For a cached eval set that is a correctness problem -- re-running would score a
different test, and two backbones raced against each other would not face the
same images.

Every stochastic view is therefore seeded, and seeded **on the image's content
id**, not on its row index and no longer on its path (`SEED_SCHEME`). Index
seeding was the obvious choice and is wrong for the workflow this module is
actually used in: a 2,000-row stratified subsample gives every image a different
row index than the full manifest does, so the same photo would get a different
noise realization in the subsample than in the full run. The subsample's numbers
would then differ from the full run's for a reason that looks exactly like a
real effect.

Path seeding fixed that and carried its own version of the same defect one level
down: renaming the repository silently redrew every noise realization, so the
"same" eval set was quietly a different test. Seeding on content makes an
image's degradations a property of the image, which is the only thing they were
ever meant to be. It also, unavoidably, makes every path-seeded vector
unreproducible -- which is why the Tier 4 migration deliberately did not import
them.

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

from aigc_detect.cache.hashing import HashCache
from aigc_detect.cache.identity import identity_from_module
from aigc_detect.cache.store import EmbeddingStore, view_id
from aigc_detect.config import EMBEDDINGS_DIR, RANDOM_SEED, get_settings
from aigc_detect.data.dataset import ManifestImageDataset, resolve_image_path
from aigc_detect.data.transforms import (
    build_robustness_views,
    eval_view_names,
    train_chain_view_names,
)
from aigc_detect.embed.embeddings import fingerprint_paths
from aigc_detect.log import get_logger
from aigc_detect.registry.backbones import load_backbone

logger = get_logger(__name__)

# Bumped only if the *seeding scheme* changes. Folded into the fingerprint of
# stochastic views only -- a deterministic view's output does not depend on it,
# so bumping this must not invalidate `clean`, `jpeg_*`, `blur_*` etc.
#
# path-v1 -> content-v1 at Tier 4. Every stochastic vector computed under
# path-v1 is now unreproducible, by design: its seed came from a string the
# refactor is in the business of making irrelevant. Those views recompute; the
# deterministic ones, which never depended on this, do not.
SEED_SCHEME = "content-v1"


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


def load_view_cache(
    backbone_key: str,
    stem: str,
    view: str,
    spec: str,
    expected_manifest_fp: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one cached view, verifying its integrity fields.

    Every consumer of a view cache must go through this: the cache filename
    (<backbone>__<stem>__<view>.npz) does not uniquely identify its contents --
    `main.py split` rewrites manifests in place under the same filename, and
    editing a transform severity changes what a view name means. A stale cache
    therefore produces confident, plausible, WRONG numbers rather than an
    error, so both fingerprints are checked before anything is returned.

    Returns ``(embeddings, labels, meta)``: embeddings as float32, labels as
    int64, and ``meta`` a dict with at least ``"manifest_fingerprint"``, plus
    ``"sources"``/``"generators"``/``"image_paths"`` when present in the file.

    Raises ``SystemExit`` with an actionable message naming the file and the
    re-run command if: the file is missing, it has no ``view_fingerprint``,
    its ``view_fingerprint`` != ``view_fingerprint(spec)``, or
    ``expected_manifest_fp`` is given and does not match the cached
    ``manifest_fingerprint``.
    """
    path = view_embeddings_path(backbone_key, stem, view)
    if not path.exists():
        raise SystemExit(
            f"[load-view-cache] missing {path.name}. Run: uv run main.py embed-views "
            f"--backbone {backbone_key} --views {view} (plus --manifest/--sample-rows matching "
            f"cache_stem '{stem}')"
        )
    with np.load(path, allow_pickle=True) as d:
        if "view_fingerprint" not in d or str(d["view_fingerprint"]) != view_fingerprint(spec):
            raise SystemExit(
                f"[load-view-cache] STALE: {path.name} was computed under a different definition "
                f"of view '{view}' (spec is now {spec!r}). Re-run: uv run main.py embed-views "
                f"--backbone {backbone_key} --force"
            )
        m_fp = str(d["manifest_fingerprint"]) if "manifest_fingerprint" in d else None
        if expected_manifest_fp is not None and m_fp != expected_manifest_fp:
            raise SystemExit(
                f"[load-view-cache] STALE: {path.name} was computed from a different row "
                f"selection than the manifest yields now (the split was rebuilt). "
                f"Re-run: uv run main.py embed-views --backbone {backbone_key} --force"
            )
        embeddings = d["embeddings"].astype(np.float32)
        labels = d["labels"].astype(np.int64)
        meta: dict = {"manifest_fingerprint": m_fp}
        for key in ("sources", "generators", "image_paths"):
            if key in d:
                meta[key] = d[key].astype(str)
    return embeddings, labels, meta


def view_seed(image_id: str, view: str) -> int:
    """Stable per-(image, view) seed, derived from the image's CONTENT id.

    Not the row index (a subsample would redraw every realization) and no longer
    the path (a repo rename would redraw every realization). An image's
    degradations are now a property of its bytes, so they are the same in every
    subsample, on every machine, at every path -- see the module docstring.
    """
    h = hashlib.sha1(f"{image_id}::{view}::{SEED_SCHEME}".encode("utf-8", "replace")).digest()
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

        for (_src, g), k in zip(groups, alloc, strict=True):
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
    """Yields (views, label, idx) where views is a (n_views, 3, S, S) tensor
    built from a SINGLE decode of the source image.

    `idx` is returned so the writer can name the rows a batch corresponds to
    without relying on the loader's ordering. It is redundant today (shuffle is
    off) and it is one assumption fewer standing between a forward pass and the
    content id its vector gets filed under.
    """

    def __init__(self, df: pd.DataFrame, pipelines: dict, img_ids: list[str]):
        self.df = df.reset_index(drop=True)
        self.view_names = list(pipelines.keys())
        self.pipelines = pipelines
        if len(img_ids) != len(self.df):
            raise ValueError(f"[embed-views] {len(img_ids)} ids for {len(self.df)} rows")
        self.img_ids = img_ids

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(resolve_image_path(row["image_path"])).convert("RGB")  # ONE decode

        tensors = []
        for name in self.view_names:
            # Seeded per (content id, view): color_jitter, the noise views and
            # two of the chains are random by construction, and a cached eval
            # set must be reproducible across runs, workers, backbones, machines
            # -- and across subsamples (see the module docstring).
            torch.manual_seed(view_seed(self.img_ids[idx], name))
            tensors.append(self.pipelines[name](img))
        return torch.stack(tensors), int(row["label"]), idx


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
        # An explicit --views list wins, which means include_train_chains has no
        # effect here. Say so instead of silently dropping the 4 chain views:
        # the caller asked for them, the cache would come back 7 views deep, and
        # train-head-views --with-chains would then fail on a missing view with
        # no hint that a flag had been ignored. (Same failure shape as the
        # only_generators filter that was declared, printed, and never applied.)
        if include_train_chains:
            missing = [v for v in train_chain_view_names() if v not in views]
            if missing:
                raise SystemExit(
                    f"[embed-views] --train-chains was passed together with an explicit --views "
                    f"list that omits {missing}. An explicit list is used verbatim, so the chain "
                    f"views would be silently skipped.\n"
                    f"          Either drop --views (the default set honours --train-chains), or "
                    f"name all of them:\n"
                    f"          --views {' '.join(list(views) + missing)}"
                )
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

    if dtype == "float32":
        logger.warning(
            "--dtype float32 controls the .npz container only; the store holds float16. "
            "Under AMP that is lossless (max |f32-f16| == 0 on this pipeline), but a CPU "
            "run without autocast would be widened back from fp16, not recovered."
        )

    settings = get_settings()
    with (
        EmbeddingStore(settings.store_root) as store,
        HashCache(settings.hash_db_path) as hashes,
    ):
        ident = identity_from_module(backbone_key, module, pooled_dim, native_res)
        ident.register(store)

        # One stat() per file, and a read only for files the memo has not seen.
        img_ids = hashes.ids_for(
            [resolve_image_path(v) for v in df["image_path"]], progress=True
        )
        n_unique = len(set(img_ids))
        if n_unique != n:
            # Duplicate BYTES under different paths. Worth saying out loud: the
            # store computes each once, so the forward-pass count will not match
            # the row count, and that is not a bug.
            logger.info("%d of %d rows are byte-duplicates; each is embedded once", n - n_unique, n)

        vids: dict[str, str] = {}
        for name in all_pipelines:
            spec = all_specs[name]
            scheme = SEED_SCHEME if "|stochastic" in spec else None
            vid = view_id(name, spec, scheme)
            store.register_view(vid, name=name, spec=spec, seed_scheme=scheme)
            vids[name] = vid

        if force:
            for name, vid in vids.items():
                dropped = store.drop(ident.bb_id, vid, img_ids)
                if dropped:
                    logger.info("--force: dropped %s stored rows for view '%s'", f"{dropped:,}", name)

        # THE ONLY QUESTION THAT COSTS GPU: which (view, image) pairs are absent.
        # Not "is this file stale" -- a manifest that gained ten rows now costs
        # ten forward passes, and a repo that moved costs none.
        gaps = {name: store.missing(ident.bb_id, vid, img_ids) for name, vid in vids.items()}
        compute = {name: pipe for name, pipe in all_pipelines.items() if gaps[name]}

        if compute:
            wanted = set().union(*(set(gaps[name]) for name in compute))
            # First row per absent id: a byte-duplicate does not earn a second
            # forward pass just because it appears twice in the manifest.
            first_row: dict[str, int] = {}
            for row_i, iid in enumerate(img_ids):
                if iid in wanted and iid not in first_row:
                    first_row[iid] = row_i
            rows = sorted(first_row.values())
            sub_df = df.iloc[rows].reset_index(drop=True)
            sub_ids = [img_ids[i] for i in rows]

            view_names = list(compute)
            print(f"[embed-views] manifest={manifest_path} stem={stem} rows={n} views={len(all_pipelines)}")
            print(f"[embed-views] computing {len(view_names)} view(s): {', '.join(view_names)}")
            print(
                f"[embed-views] {len(rows):,} of {n:,} rows are missing from the store -- "
                f"{len(rows) * len(view_names):,} forward passes from {len(rows):,} decodes"
            )
            for name in view_names:
                if len(gaps[name]) != len(rows):
                    print(f"[embed-views]   {name}: {len(gaps[name]):,} missing")

            loader = DataLoader(
                MultiViewDataset(sub_df, compute, sub_ids),
                batch_size=batch_size, shuffle=False, num_workers=num_workers,
            )
            device = next(module.parameters()).device
            use_amp = device.type == "cuda"
            committed = 0
            with torch.no_grad():
                for batched_views, _labels, idx in tqdm(loader, desc=f"[embed-views] {backbone_key}"):
                    b, v = batched_views.shape[0], batched_views.shape[1]
                    flat = batched_views.reshape(b * v, *batched_views.shape[2:]).to(
                        device, non_blocking=True
                    )
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        feats = module(flat)
                    feats = feats.float().cpu().numpy().reshape(b, v, pooled_dim)
                    batch_ids = [sub_ids[j] for j in idx.tolist()]
                    # Committed per batch, not per run: this is the whole of
                    # "resume". Kill the process here and everything up to the
                    # last batch survives.
                    for i, name in enumerate(view_names):
                        committed += store.put_batch(ident.bb_id, vids[name], batch_ids, feats[:, i, :])
            print(f"[embed-views] committed {committed:,} vectors to the store")
        else:
            print("[embed-views] every requested view is already in the store -- no forward passes")

        return _project_to_npz(
            store=store, ident=ident, vids=vids, specs=all_specs, views=list(all_pipelines),
            backbone_key=backbone_key, stem=stem, manifest_path=manifest_path, df=df,
            img_ids=img_ids, m_fp=m_fp, module=module, native_res=native_res,
            pooled_dim=pooled_dim, sample_rows=sample_rows, sample_seed=sample_seed,
            dtype=dtype, force=force, recomputed=set(compute),
        )


def _npz_is_current(path: Path, m_fp: str, v_fp: str) -> bool:
    """Does this .npz already describe exactly this row selection and view spec?

    A corrupt or truncated file answers False rather than raising: the only
    consequence of rebuilding one is a gather, which is now cheap.
    """
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            return (
                str(d["manifest_fingerprint"]) == m_fp
                and "view_fingerprint" in d
                and str(d["view_fingerprint"]) == v_fp
            )
    except Exception:
        return False


def _project_to_npz(
    *, store, ident, vids, specs, views, backbone_key, stem, manifest_path, df, img_ids,
    m_fp, module, native_res, pooled_dim, sample_rows, sample_seed, dtype, force, recomputed,
) -> list[Path]:
    """Write the manifest-ordered .npz view of what the store holds.

    The store is authoritative; this is the shape every reader in the project
    still expects (`load_view_cache`, the grid, the trainer, error analysis).
    Keeping it means Tier 4 changes how vectors are addressed without touching a
    single consumer -- the readers move in Tier 7, on their own schedule.

    Rebuilding one costs a gather. That is the point: after the Tier 5 corpus
    move every fingerprint here changes and every file below is rewritten, with
    no GPU involved at all.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    labels = df["label"].to_numpy(dtype=np.int64)
    sources = np.array(df["source"].astype(str).tolist(), dtype=str)
    # Tiny-GenImage carries a per-image generator tag; keep it so the grid can
    # be broken down by generator without re-reading the manifest.
    generators = (
        np.array(df["generator"].astype(str).tolist(), dtype=str)
        if "generator" in df.columns
        else np.array([""] * len(df), dtype=str)
    )
    paths = np.array(df["image_path"].astype(str).tolist(), dtype=str)
    np_dtype = np.float16 if dtype == "float16" else np.float32

    written: list[Path] = []
    rebuilt: list[Path] = []
    for name in views:
        out = view_embeddings_path(backbone_key, stem, name)
        v_fp = view_fingerprint(specs[name])
        if name not in recomputed and not force and _npz_is_current(out, m_fp, v_fp):
            print(f"[embed-views] {out.name} matches the manifest and view spec -- skipping")
            written.append(out)
            continue

        matrix, missing = store.gather(ident.bb_id, vids[name], img_ids)
        if missing:
            raise SystemExit(
                f"[embed-views] {len(missing)} of {len(img_ids)} rows for view '{name}' are "
                f"absent from the store after the compute pass. This should not happen; the "
                f"store may be damaged. Check: uv run aigc cache status"
            )
        np.savez(
            out,
            embeddings=matrix.astype(np_dtype),
            labels=labels,
            sources=sources,
            generators=generators,
            image_paths=paths,
            image_ids=np.array(img_ids, dtype=str),
            view=name,
            view_spec=specs[name],
            backbone=backbone_key,
            checkpoint=module.checkpoint_used,
            bb_id=ident.bb_id,
            view_id=vids[name],
            native_res=native_res,
            pooled_dim=pooled_dim,
            manifest_path=str(manifest_path),
            cache_stem=stem,
            n_rows=len(df),
            sample_rows=-1 if sample_rows is None else sample_rows,
            sample_seed=sample_seed,
            manifest_fingerprint=m_fp,
            view_fingerprint=v_fp,
            seed_scheme=SEED_SCHEME,
            norm_mean=module.norm_mean,
            norm_std=module.norm_std,
            norm_source=module.norm_source,
        )
        written.append(out)
        rebuilt.append(out)

    if rebuilt:
        total_mb = sum(p.stat().st_size for p in rebuilt) / 1e6
        print(
            f"[embed-views] projected {len(rebuilt)} view(s) x {len(df)} rows "
            f"({total_mb:.0f} MB) -> {EMBEDDINGS_DIR}"
        )
    return written
