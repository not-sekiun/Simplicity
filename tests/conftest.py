"""Shared fixtures: telling "the data isn't here" apart from "the code is wrong".

A fresh clone commits every ``corpus.yaml``, ``index.csv``, manifest recipe and
resolved CSV under ``data/`` -- 64 files (see AGENTS.md's ``data/`` hierarchy).
What it does NOT have is the images themselves: every corpus's ``images/``
directory is gitignored, since the tree is ~24 GB and irreproducible without a
real pull.

Before this fixture existed, a test that needed pixels failed the same way a
test with a genuine bug fails -- an uncaught ``SystemExit`` or a wrong count --
on a machine that had simply never run a corpus pull. That is the wrong
signal: it reads as "the manifest is broken" when the honest statement is "the
images are not on this machine". ``needs_images`` (and ``images_present`` for
tests that only need a plain boolean) makes that distinction explicit, so CI
on a bare checkout gets a clean, informative skip instead of a red build that
looks like a regression.

The check itself looks at exactly one corpus (``tiny_genimage``'s first
indexed row) rather than probing every manifest: a machine either has pulled
corpora or it hasn't, and walking the whole tree to be more thorough would
just make every test importing this file slower for no better an answer.
"""

from __future__ import annotations

from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def images_present() -> bool:
    """True if real image bytes are on this machine, not just the index CSVs.

    Cached for the life of the test process: the answer cannot change mid-run
    (nothing here pulls a corpus), and this gets called from many tests.
    """
    try:
        from aigc_detect.data.corpus import get_corpus
        from aigc_detect.data.dataset import resolve_image_path

        rows = get_corpus("tiny_genimage").rows()
    except SystemExit:
        # No registered corpus, or its index.csv is missing -- an even barer
        # checkout than "images not pulled yet". Same verdict either way: no
        # pixels to test against.
        return False
    if rows.empty:
        return False
    return resolve_image_path(rows.iloc[0]["image_path"]).is_file()


@pytest.fixture
def needs_images():
    """Skip the test cleanly when this machine has no pulled image bytes.

    Use as a plain fixture argument: ``def test_x(needs_images): ...``. Only
    reach for this in a test that actually opens/stats an image file or a
    manifest row whose resolution depends on one being present
    (``require_on_disk``) -- a test that only reads committed YAML/CSV
    structure needs no skip and should keep running on a bare checkout.
    """
    if not images_present():
        pytest.skip(
            "data/corpora/*/images are not on this machine (gitignored) -- "
            "pull a corpus first (see AGENTS.md) to run pixel-dependent checks"
        )
