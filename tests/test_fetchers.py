"""Tier 6's acceptance tests: resumable pulls, the source registry, and the audit gate.

Hermetic throughout -- nothing here touches the network or a real corpus on
disk. Resumability is exercised against `FakeFetcher`, an in-memory backend
built only for this file; it obeys exactly the contract every real backend
does (`aigc_detect.data.fetchers.base.IncrementalIndexWriter`/`open_state`),
which is what makes testing it a real test of that contract rather than a
test of `FakeFetcher` itself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from aigc_detect.data import sources as sources_mod
from aigc_detect.data.corpus import assert_trainable
from aigc_detect.data.fetchers import (
    CorpusPaths,
    IncrementalIndexWriter,
    PullResult,
    get_fetcher,
    open_state,
)
from aigc_detect.data.fetchers.hf import HFStreamingFetcher
from aigc_detect.data.sources import UNREGISTERED, Source, all_sources


@dataclass
class FakeFetcher:
    """An in-memory `Fetcher`: `total` deterministic rows, batched checkpoints,
    and a `fail_at` kill switch. `fail_at` lives on the fetcher instance, not
    in `source.config` -- a real kill (Ctrl+C, an OOM, a network drop) isn't
    part of a source's configuration either, and putting it there would change
    `config_hash` between the killed run and the resumed one, which is exactly
    the "two different pull configurations under one corpus id" case
    `open_state` exists to refuse.
    """

    fail_at: int | None = None
    batch_size: int = 2

    def pull(self, source: Source, dest: CorpusPaths, state) -> PullResult:
        total = source.config["total"]
        resumed = state.rows_written > 0
        writer = IncrementalIndexWriter(dest, state, batch_size=self.batch_size)
        scanned = state.rows_scanned
        for i in range(scanned, total):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError(f"simulated kill at row {i}")
            p = dest.images / f"img_{i:03d}.jpg"
            p.write_bytes(f"row-{i}".encode())
            writer.add({"image_path": str(p), "label": i % 2, "source": source.id, "generator": ""})
            scanned = i + 1
            writer.maybe_checkpoint(rows_scanned=scanned, cursor={})
        writer.finish(rows_scanned=scanned, cursor={}, completed=True)
        return PullResult(
            rows_written=state.rows_written,
            rows_scanned=state.rows_scanned,
            resumed=resumed,
            completed=True,
        )


def _read_index(dest: CorpusPaths) -> list[dict]:
    with open(dest.index_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalize(rows: list[dict]) -> list[dict]:
    """Rows with `image_path` reduced to its filename, so two pulls into two
    different tmp roots can be compared on everything BUT the incidental
    absolute-path prefix that comes from where the test happened to put them."""
    out = []
    for r in rows:
        r = dict(r)
        r["image_path"] = Path(r["image_path"]).name
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Acceptance test 1: kill a pull at 40%, resume it, compare to an
# uninterrupted run.
# ---------------------------------------------------------------------------


def test_a_killed_pull_resumes_to_an_identical_result(tmp_path):
    ref_dest = CorpusPaths(root=tmp_path / "ref")
    ref_source = Source(id="fake_corpus", fetcher="fake", license="unverified", config={"total": 10})
    ref_state, ref_resumed = open_state(ref_source, ref_dest, force=False)
    assert not ref_resumed
    FakeFetcher().pull(ref_source, ref_dest, ref_state)
    reference = _normalize(_read_index(ref_dest))
    assert len(reference) == 10

    # Same source, same corpus id, a fresh destination -- killed at row 4 (40%).
    kill_dest = CorpusPaths(root=tmp_path / "kill")
    kill_source = Source(id="fake_corpus", fetcher="fake", license="unverified", config={"total": 10})
    state, resumed = open_state(kill_source, kill_dest, force=False)
    assert not resumed
    with pytest.raises(RuntimeError, match="simulated kill at row 4"):
        FakeFetcher(fail_at=4).pull(kill_source, kill_dest, state)

    partial = _read_index(kill_dest)
    assert 0 < len(partial) < 10, "the crash-recovery contract: some rows committed, not all"
    assert len(partial) == 4  # two checkpoints (batch_size=2) landed before the kill

    # A fresh process re-opens state from disk and resumes -- no `fail_at` this time.
    state2, resumed2 = open_state(kill_source, kill_dest, force=False)
    assert resumed2
    assert state2.rows_written == len(partial)
    result = FakeFetcher().pull(kill_source, kill_dest, state2)
    assert result.completed

    resumed_final = _normalize(_read_index(kill_dest))
    assert resumed_final == reference, "a resumed pull must reproduce an uninterrupted one, row for row"


# ---------------------------------------------------------------------------
# Acceptance test 2: a new HF source needs eight lines of YAML, no Python.
# ---------------------------------------------------------------------------


def test_new_hf_source_from_a_tmp_yaml_needs_no_python(tmp_path, monkeypatch):
    registry_path = tmp_path / "sources.yaml"
    registry_path.write_text(
        "version: 1\n"
        "sources:\n"
        "  test_new_source:\n"
        "    fetcher: hf_streaming\n"
        "    repo: some-org/some-dataset\n"
        "    split: train\n"
        "    image_col: image\n"
        "    label: aigc\n"
        "    source_name: test_new_source\n"
        "    license: unverified\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sources_mod, "REGISTRY_PATH", registry_path)
    sources_mod._load_registry.cache_clear()
    try:
        source = sources_mod.get_source("test_new_source")
        assert source.fetcher == "hf_streaming"
        assert source.config["repo"] == "some-org/some-dataset"
        assert source.config["label"] == "aigc"
        # The registry path resolves all the way to a real, runnable backend --
        # no Python was written for this source, only YAML.
        assert isinstance(get_fetcher(source.fetcher), HFStreamingFetcher)
    finally:
        sources_mod._load_registry.cache_clear()


# ---------------------------------------------------------------------------
# Acceptance test 3: a config-hash change refuses to resume.
# ---------------------------------------------------------------------------


def test_config_hash_change_refuses_to_resume_without_force(tmp_path):
    dest = CorpusPaths(root=tmp_path / "corp")
    source_a = Source(id="x", fetcher="fake", license="unverified", config={"total": 10})
    state, _ = open_state(source_a, dest, force=False)
    writer = IncrementalIndexWriter(dest, state, batch_size=1)
    (dest.images / "a.jpg").write_bytes(b"x")
    writer.add({"image_path": str(dest.images / "a.jpg"), "label": 0, "source": "x", "generator": ""})
    writer.finish(rows_scanned=1, cursor={}, completed=False)
    assert dest.state_path.is_file()

    source_b = Source(id="x", fetcher="fake", license="unverified", config={"total": 999})
    with pytest.raises(SystemExit, match="config has changed"):
        open_state(source_b, dest, force=False)

    # The partial pull is untouched by the refusal.
    assert dest.index_csv.is_file()

    # --force is the only way past it, and it starts clean.
    state2, resumed2 = open_state(source_b, dest, force=True)
    assert not resumed2
    assert state2.rows_written == 0
    assert not dest.index_csv.exists()


# ---------------------------------------------------------------------------
# Acceptance test 4: every corpus with a real fetcher is registered, or is a
# declared gap.
# ---------------------------------------------------------------------------


def test_every_corpus_is_registered_or_declared_unregistered():
    from aigc_detect.data.corpus import all_corpora

    registered = set(all_sources())
    gaps = [cid for cid in all_corpora() if cid not in registered and cid not in UNREGISTERED]
    assert not gaps, f"corpora with no sources.yaml entry and not in sources.UNREGISTERED: {gaps}"


# ---------------------------------------------------------------------------
# Acceptance test 5: a suspect corpus refuses to resolve into a training
# manifest without an override.
# ---------------------------------------------------------------------------


def test_suspect_corpus_refuses_to_train_without_override(tmp_path, monkeypatch):
    from aigc_detect.data.audit import gate as audit_gate
    from aigc_detect.data.corpus import all_corpora

    # A real, on-registry, role=train corpus id -- assert_trainable's role
    # check must pass so the audit check is what's actually being exercised.
    corpus_id = next(cid for cid, c in all_corpora().items() if c.role == "train")

    # Point the gate at a scratch directory so nothing here touches the real
    # data/corpora/<id>/corpus.yaml.
    monkeypatch.setattr(audit_gate, "CORPORA_DIR", tmp_path)

    # No verdict on file yet: absence of an audit is not itself suspect.
    assert_trainable(corpus_id, manifest_name="test_manifest")

    corpus_dir = tmp_path / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.yaml").write_text(
        "id: " + corpus_id + "\naudit:\n  suspect: true\n  balanced_acc: 0.81\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="blind-probe audit cleared"):
        assert_trainable(corpus_id, manifest_name="test_manifest")

    # The deliberate override, recorded in the corpus's own corpus.yaml.
    (corpus_dir / "corpus.yaml").write_text(
        "id: " + corpus_id + "\naudit:\n  suspect: true\n  balanced_acc: 0.81\n  override: true\n"
        "  override_reason: test override\n",
        encoding="utf-8",
    )
    assert_trainable(corpus_id, manifest_name="test_manifest")
