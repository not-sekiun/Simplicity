"""Resolve "which model produced this vector" without paying to load the model.

`backbone_id` (store.py) is a digest over the backbone's key, its resolved
checkpoint and revision, its pooled dimension, its native resolution, and its
normalization statistics. Five of those six come straight from
BACKBONE_REGISTRY. The sixth does not: `norm_mean`/`norm_std` are read off the
loaded checkpoint (timm's `resolve_model_data_config`, or the transformers image
processor), which is the correct source -- FINDINGS trap 9 is precisely what
happens when normalization is assumed rather than resolved.

That leaves a read path with an awkward cost: asking "is this image already
embedded?" would require instantiating a 1.2 GB vision tower purely to learn two
tuples of three floats.

So the store's own `backbones` table doubles as the memo, exactly the way
`hashes.sqlite` memoises file hashes. Normalization is a pure function of the
checkpoint, so a row matching on (key, checkpoint, revision, dim, native_res)
carries the norm this checkpoint resolves to, and no load is needed. A miss --
a new backbone, or a checkpoint repointed since -- falls back to loading, which
is the honest cost of not having seen those weights before.

WHY NOT KEY THE LOOKUP ON `key` ALONE. Because a repointed checkpoint or a
changed revision pin must MISS. Matching on the five registry-known fields means
the memo can only ever answer for the exact model the registry currently
describes; anything else loads and gets its own id.
"""

from __future__ import annotations

from dataclasses import dataclass

from aigc_detect.cache.store import EmbeddingStore, backbone_id
from aigc_detect.log import get_logger
from aigc_detect.registry.backbones import BACKBONE_REGISTRY

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackboneIdentity:
    """Everything that takes part in a vector's "which model" half."""

    bb_id: str
    key: str
    checkpoint: str
    revision: str | None
    dim: int
    native_res: int
    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]

    def register(self, store: EmbeddingStore) -> None:
        store.register_backbone(
            self.bb_id, key=self.key, checkpoint=self.checkpoint, revision=self.revision,
            dim=self.dim, native_res=self.native_res,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
        )


def identity_from_module(key: str, module, dim: int, native_res: int) -> BackboneIdentity:
    """Build an identity from an already-loaded backbone.

    `checkpoint_used` rather than the registry's checkpoint string: metaclip2-h
    can fall back to a timm mirror, and vectors produced by the mirror must not
    claim to have come from the entry that failed to load.
    """
    entry = BACKBONE_REGISTRY.get(key, {})
    checkpoint = str(getattr(module, "checkpoint_used", entry.get("checkpoint", key)))
    revision = entry.get("revision") if checkpoint == entry.get("checkpoint") else None
    mean = tuple(float(x) for x in module.norm_mean)
    std = tuple(float(x) for x in module.norm_std)
    return BackboneIdentity(
        bb_id=backbone_id(key, checkpoint, revision, dim, native_res, mean, std),
        key=key, checkpoint=checkpoint, revision=revision,
        dim=dim, native_res=native_res, norm_mean=mean, norm_std=std,
    )


def identity_from_store(store: EmbeddingStore, key: str) -> BackboneIdentity | None:
    """The identity for `key` if the store has already seen exactly this model.

    Returns None when the store holds no row matching the registry's current
    description of `key`, or -- defensively -- more than one, which would mean
    two different normalizations were recorded for one checkpoint and is not a
    thing the caller should be allowed to guess about.
    """
    import json

    entry = BACKBONE_REGISTRY.get(key)
    if entry is None:
        return None
    rows = store._conn.execute(
        "SELECT bb_id, checkpoint, revision, dim, native_res, norm FROM backbones "
        "WHERE key=? AND checkpoint=? AND revision IS ? AND dim=? AND native_res=?",
        (key, entry["checkpoint"], entry.get("revision"), entry["pooled_dim"], entry["native_res"]),
    ).fetchall()
    if len(rows) != 1:
        return None
    bb_id, checkpoint, revision, dim, res, norm = rows[0]
    n = json.loads(norm)
    return BackboneIdentity(
        bb_id=bb_id, key=key, checkpoint=checkpoint, revision=revision,
        dim=dim, native_res=res,
        norm_mean=tuple(float(x) for x in n["mean"]),
        norm_std=tuple(float(x) for x in n["std"]),
    )


def resolve_identity(store: EmbeddingStore, key: str, *, allow_load: bool = True) -> BackboneIdentity:
    """Identity for `key`, from the store's memo if possible, else by loading.

    Raises SystemExit rather than loading when `allow_load` is False, naming the
    command that would populate the memo -- a read-only caller asking for a
    backbone the store has never seen has a data problem, not a lazy-loading
    problem, and silently spending two minutes and 1.2 GB on it would hide that.
    """
    known = identity_from_store(store, key)
    if known is not None:
        return known
    if not allow_load:
        raise SystemExit(
            f"[cache] the store has no vectors from backbone '{key}' as the registry currently "
            f"describes it, so its identity cannot be resolved without loading the checkpoint.\n"
            f"        Run: uv run aigc embed-views --backbone {key} --manifest <manifest>"
        )
    from aigc_detect.registry.backbones import load_backbone

    logger.info("resolving %s identity by loading the checkpoint (not yet in the store)", key)
    module, dim, native_res = load_backbone(key)
    ident = identity_from_module(key, module, dim, native_res)
    ident.register(store)
    return ident
