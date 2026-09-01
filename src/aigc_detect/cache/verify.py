"""Prove a migrated store actually holds what it claims.

Migration re-derives each row's identity by hashing the file named in the legacy
.npz. That assumes the file on disk today is the file that produced the vector.
It usually is -- but "usually" is not a basis for trusting 1.1 million rows that
back every published number in this project, and the failure mode is silent: a
wrong vector produces a plausible AUC, not an error.

So we re-embed a random sample through the live pipeline and compare against
what the store returns. Any drift -- a re-encoded image, a transform spec that
changed meaning since the cache was written, a mis-sharded row -- shows up as a
vector that does not match.

TOLERANCE. The forward pass runs under autocast, so bit-exactness is not
promised across runs; the comparison is on cosine similarity, which is what the
downstream linear head actually sees. A genuine mismatch (different image,
different transform) lands far below the threshold rather than near it, so this
does not need a delicately tuned epsilon.
"""

from __future__ import annotations

import numpy as np

# A correct row re-embeds to essentially the same direction. Autocast jitter
# lives around 1 - 1e-6; a different image or transform is typically < 0.9.
COSINE_MIN = 0.9995


def _paths_for_ids(hashes, img_ids: list[str]) -> dict[str, str]:
    """Reverse the memo: content id -> a path that currently holds those bytes."""
    from aigc_detect.config import DATA_DIR

    found: dict[str, str] = {}
    conn = hashes._conn
    for start in range(0, len(img_ids), 800):
        batch = img_ids[start : start + 800]
        q = ",".join("?" * len(batch))
        for iid, rel in conn.execute(
            f"SELECT img_id, path FROM file_ids WHERE img_id IN ({q})", batch
        ):
            if iid not in found:
                p = DATA_DIR / rel
                found[iid] = str(p if p.exists() else rel)
    return found


def verify_sample(store, hashes, *, n_images: int = 200, seed: int = 0) -> bool:
    """Re-embed a sample and compare. Returns True if every checked row matches."""
    import torch
    from PIL import Image

    from aigc_detect.data.transforms import build_robustness_views
    from aigc_detect.registry.backbones import load_backbone

    conn = store._conn
    groups = conn.execute(
        "SELECT b.bb_id, b.key, b.dim, b.native_res, b.norm, v.view_id, v.name, v.spec "
        "FROM rows_ r JOIN backbones b ON b.bb_id=r.bb_id JOIN views v ON v.view_id=r.view_id "
        "GROUP BY r.bb_id, r.view_id"
    ).fetchall()
    if not groups:
        print("[verify] store is empty -- nothing to check")
        return True

    rng = np.random.default_rng(seed)
    # Spread the budget across (backbone, view) groups so no view goes unchecked.
    per_group = max(1, n_images // len(groups))
    print(f"[verify] {len(groups)} (backbone, view) groups, ~{per_group} images each")

    loaded: dict[str, tuple] = {}
    failures: list[str] = []
    checked = 0

    for bb_id, key, _dim, _native_res, _norm, vid, view_name, spec in groups:
        ids = [r[0] for r in conn.execute(
            "SELECT img_id FROM rows_ WHERE bb_id=? AND view_id=?", (bb_id, vid)
        )]
        if not ids:
            continue
        pick = [ids[i] for i in rng.choice(len(ids), size=min(per_group, len(ids)), replace=False)]
        paths = _paths_for_ids(hashes, pick)
        pick = [i for i in pick if i in paths]
        if not pick:
            continue

        if key not in loaded:
            loaded[key] = load_backbone(key)
        module, _pooled_dim, res = loaded[key]

        pipelines, specs = build_robustness_views(
            image_size=res, norm_mean=module.norm_mean, norm_std=module.norm_std
        )
        if view_name not in pipelines:
            failures.append(f"{key}/{view_name}: view no longer exists in the transform table")
            continue
        if specs[view_name] != spec:
            failures.append(
                f"{key}/{view_name}: SPEC DRIFT -- store has {spec!r}, "
                f"current table says {specs[view_name]!r}"
            )
            continue

        pipe = pipelines[view_name]
        device = next(module.parameters()).device
        batch = []
        kept = []
        for iid in pick:
            try:
                img = Image.open(paths[iid]).convert("RGB")
            except OSError:
                continue
            batch.append(pipe(img))
            kept.append(iid)
        if not batch:
            continue

        with torch.no_grad():
            x = torch.stack(batch).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                fresh = module(x)
            fresh = fresh.float().cpu().numpy()

        stored, missing = store.gather(bb_id, vid, kept)
        if missing:
            failures.append(f"{key}/{view_name}: {len(missing)} sampled rows absent from the store")
            continue

        a = fresh / (np.linalg.norm(fresh, axis=1, keepdims=True) + 1e-12)
        b = stored / (np.linalg.norm(stored, axis=1, keepdims=True) + 1e-12)
        cos = (a * b).sum(axis=1)
        checked += len(cos)
        bad = int((cos < COSINE_MIN).sum())
        if bad:
            failures.append(
                f"{key}/{view_name}: {bad}/{len(cos)} rows mismatch "
                f"(min cosine {cos.min():.6f}, median {np.median(cos):.6f})"
            )

    print(f"[verify] compared {checked:,} rows across {len(groups)} groups")
    if failures:
        print(f"[verify] FAILED -- {len(failures)} group(s):")
        for f in failures:
            print(f"  {f}")
        return False
    print("[verify] PASS -- every sampled row re-embeds to its stored vector")
    return True


