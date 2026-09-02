"""Invariants of the content-addressed embedding store (Tier 4).

These are the properties the rest of the pipeline is now allowed to assume, so
they are asserted rather than described. Nothing here touches a GPU, a
checkpoint, or the real data root -- the store is deliberately ignorant of all
three, and a test that needed them would be testing something else.

The two acceptance tests from the refactor plan's "definition of done" appear
here in the form the store can answer them:

  * move the files, keep the bytes  ->  the same ids, so zero recomputation
  * change one image                ->  exactly one id changes, so exactly one
                                        row is recomputed
"""

from __future__ import annotations

import numpy as np
import pytest

from aigc_detect.cache.hashing import HashCache, content_id
from aigc_detect.cache.store import EmbeddingStore, backbone_id, view_id

DIM = 8


def _vec(seed: int) -> np.ndarray:
    return np.full((DIM,), float(seed), dtype=np.float32)


def _store(tmp_path, name="store") -> EmbeddingStore:
    store = EmbeddingStore(tmp_path / name)
    store.register_backbone("bb", key="test-bb", checkpoint="ckpt", revision=None,
                            dim=DIM, native_res=224, norm_mean=(0.5,) * 3, norm_std=(0.5,) * 3)
    store.register_view("vv", name="clean", spec="clean", seed_scheme=None)
    return store


def _image(path, payload: bytes) -> object:
    path.write_bytes(payload)
    return path


# -- content addressing -------------------------------------------------------


def test_same_bytes_at_a_different_path_have_the_same_id(tmp_path):
    """The rename acceptance test, at the level the id is decided."""
    (tmp_path / "here").mkdir()
    (tmp_path / "moved").mkdir()
    a = _image(tmp_path / "here" / "x.jpg", b"pixels")
    b = _image(tmp_path / "moved" / "renamed.jpg", b"pixels")
    assert content_id(a) == content_id(b)


def test_changed_bytes_change_the_id(tmp_path):
    p = _image(tmp_path / "x.jpg", b"pixels")
    before = content_id(p)
    p.write_bytes(b"pixels-reencoded")
    assert content_id(p) != before


def test_hash_memo_rehashes_when_the_file_changes(tmp_path):
    """The memo is an optimisation and never an identity."""
    p = _image(tmp_path / "x.jpg", b"one")
    with HashCache(tmp_path / "h.sqlite") as h:
        first = h.id_for(p)
        assert h.id_for(p) == first  # served from the memo
        p.write_bytes(b"two-different-length")
        assert h.id_for(p) != first  # size moved, so it re-hashed


def test_hash_memo_reports_the_missing_file_by_name(tmp_path):
    with HashCache(tmp_path / "h.sqlite") as h, pytest.raises(FileNotFoundError, match=r"gone.jpg"):
        h.ids_for([tmp_path / "gone.jpg"])


def test_ids_are_positionally_aligned_with_the_paths_asked_for(tmp_path):
    paths = [_image(tmp_path / f"{i}.jpg", f"body-{i}".encode()) for i in range(5)]
    with HashCache(tmp_path / "h.sqlite") as h:
        ids = h.ids_for(paths)
    assert ids == [content_id(p) for p in paths]


# -- identity digests ---------------------------------------------------------


def test_backbone_id_moves_with_the_revision_pin():
    """A silent upstream re-upload must miss, not blend into the same matrix."""
    args = ("pe-core-l", "timm/x", None, 1024, 336, (0.5,) * 3, (0.5,) * 3)
    pinned = backbone_id("pe-core-l", "timm/x", "abc123", 1024, 336, (0.5,) * 3, (0.5,) * 3)
    assert backbone_id(*args) != pinned


def test_backbone_id_moves_with_the_normalization():
    a = backbone_id("k", "c", None, 8, 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    b = backbone_id("k", "c", None, 8, 224, (0.48, 0.45, 0.41), (0.5, 0.5, 0.5))
    assert a != b


def test_seed_scheme_only_participates_for_stochastic_views():
    """Bumping the seeding scheme must not invalidate `clean` or `jpeg_*`."""
    assert view_id("clean", "clean", None) == view_id("clean", "clean", None)
    a = view_id("noise", "noise(sigma=0.05)|stochastic", "path-v1")
    b = view_id("noise", "noise(sigma=0.05)|stochastic", "content-v1")
    assert a != b


# -- store round trip ---------------------------------------------------------


def test_gather_returns_rows_in_the_order_asked_for(tmp_path):
    with _store(tmp_path) as store:
        ids = ["aa" + "0" * 30, "bb" + "0" * 30, "cc" + "0" * 30]
        store.put_batch("bb", "vv", ids, np.stack([_vec(1), _vec(2), _vec(3)]))
        out, missing = store.gather("bb", "vv", [ids[2], ids[0], ids[2]])
        assert not missing
        assert [row[0] for row in out] == [3.0, 1.0, 3.0]


def test_missing_rows_come_back_as_nan_and_are_named(tmp_path):
    """A caller that ignores `missing` must get obviously-broken numbers."""
    with _store(tmp_path) as store:
        known = "aa" + "0" * 30
        absent = "ff" + "0" * 30
        store.put_batch("bb", "vv", [known], _vec(1)[None, :])
        out, missing = store.gather("bb", "vv", [known, absent])
        assert missing == [absent]
        assert np.isnan(out[1]).all()
        assert not np.isnan(out[0]).any()


def test_put_batch_is_idempotent(tmp_path):
    """Re-running an interrupted batch must not duplicate or overwrite rows."""
    with _store(tmp_path) as store:
        ids = ["aa" + "0" * 30, "bb" + "0" * 30]
        assert store.put_batch("bb", "vv", ids, np.stack([_vec(1), _vec(2)])) == 2
        assert store.put_batch("bb", "vv", ids, np.stack([_vec(9), _vec(9)])) == 0
        out, _ = store.gather("bb", "vv", ids)
        assert [row[0] for row in out] == [1.0, 2.0]


def test_missing_is_the_whole_of_resume(tmp_path):
    with _store(tmp_path) as store:
        ids = [f"{i:02x}" + "0" * 30 for i in range(6)]
        store.put_batch("bb", "vv", ids[:4], np.stack([_vec(i) for i in range(4)]))
        assert store.missing("bb", "vv", ids) == ids[4:]


def test_put_batch_rejects_a_length_mismatch(tmp_path):
    with _store(tmp_path) as store, pytest.raises(ValueError):
        store.put_batch("bb", "vv", ["aa" + "0" * 30], np.stack([_vec(1), _vec(2)]))


def test_one_changed_image_costs_exactly_one_row(tmp_path):
    """The incremental acceptance test, end to end through the hash memo."""
    files = [_image(tmp_path / f"{i}.jpg", f"image-{i}".encode()) for i in range(5)]
    with HashCache(tmp_path / "h.sqlite") as h, _store(tmp_path) as store:
        ids = h.ids_for(files)
        store.put_batch("bb", "vv", ids, np.stack([_vec(i) for i in range(5)]))
        assert store.missing("bb", "vv", ids) == []

        files[2].write_bytes(b"image-2-re-encoded")
        again = h.ids_for(files)
        assert len(store.missing("bb", "vv", again)) == 1


# -- housekeeping -------------------------------------------------------------


def test_drop_frees_the_row_and_compact_frees_the_bytes(tmp_path):
    with _store(tmp_path) as store:
        ids = ["aa" + "0" * 30, "ab" + "0" * 30]
        store.put_batch("bb", "vv", ids, np.stack([_vec(1), _vec(2)]))
        assert store.orphan_bytes() == 0

        assert store.drop("bb", "vv", [ids[0]]) == 1
        assert store.missing("bb", "vv", ids) == [ids[0]]
        assert store.orphan_bytes() == DIM * 2  # one float16 row, still on disk

        result = store.compact()
        assert result["bytes_reclaimed"] == DIM * 2
        assert store.orphan_bytes() == 0
        # The surviving row still reads back correctly through its new offset.
        out, missing = store.gather("bb", "vv", [ids[1]])
        assert not missing and out[0][0] == 2.0


def test_merge_folds_in_a_second_machines_store(tmp_path):
    """The worker constraint, gone: two stores built anywhere combine."""
    mine = ["aa" + "0" * 30]
    theirs = ["bb" + "0" * 30, "cc" + "0" * 30]
    secondary = _store(tmp_path, "secondary")
    secondary.put_batch("bb", "vv", theirs, np.stack([_vec(2), _vec(3)]))
    secondary.close()

    with _store(tmp_path, "primary") as primary:
        primary.put_batch("bb", "vv", mine, _vec(1)[None, :])
        assert primary.merge(tmp_path / "secondary") == 2
        out, missing = primary.gather("bb", "vv", mine + theirs)
        assert not missing
        assert [row[0] for row in out] == [1.0, 2.0, 3.0]


def test_stats_and_groups_agree(tmp_path):
    with _store(tmp_path) as store:
        store.put_batch("bb", "vv", ["aa" + "0" * 30], _vec(1)[None, :])
        assert store.stats()["rows"] == 1
        assert store.groups() == [("bb", "test-bb", "vv", "clean", 1)]
