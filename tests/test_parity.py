"""Tier 8b's actual deliverable: proof, not prose, that two inference entry
points agree.

WHAT THIS REPLACES. `demo/server.py` used to carry a docstring reading
"Reuses the same backbone-loading and preprocessing-parity conventions as
`aigc_detect.predict` ... so a prediction from this server means the same
thing as one from `predict.py`" -- an assertion nobody tested. It happened to
be true because the server's `Model` class was a careful hand-copy of
`predict.py`'s logic, which is exactly the kind of thing that stays true
right up until one of the two copies changes and the other doesn't (see
`inference.detector`'s module docstring for the class it actually was). Tier
8b deleted that class in favor of `inference.detector.FrozenProbeDetector`,
which `demo/server.py` (via `apps/server/app.py`) now holds instead of
reimplementing anything. This file is what makes "the two agree" a fact a CI
run checks instead of a claim a docstring makes.

TOLERANCE: atol=1e-4 on the [0, 1] sigmoid output, not exact equality.
`run_inference` (predict.py's path) and `FrozenProbeDetector.load` (the
server's path) each call `registry.backbones.load_backbone` independently --
two separately constructed model instances holding the same weights, both in
eval() mode, no dropout, nothing stochastic in the forward path. On CPU that
*should* reproduce to the last bit, but floating-point addition is not
associative, and nothing in torch or this project promises that two
independently loaded model instances sum a batch's activations in the exact
same order (buffer reuse, BLAS thread scheduling can differ run to run) --
demanding bit-exact equality would make this test flaky for reasons that
have nothing to do with what it exists to catch. 1e-4 is:
  - three orders of magnitude below the shipping decision threshold's own
    distance from a neutral 0.5 (0.980 -- see
    `inference.bundle.LEGACY_DEFAULT_THRESHOLD`'s docstring),
  - two orders below the smallest step `train.calibrate` sweeps the
    threshold in (0.005),
  - and roughly the scale of ordinary float32 accumulation noise across a
    ~1024-dim pooled embedding and a few hundred million parameters.
A genuine preprocessing divergence -- wrong norm stats, a skipped resize,
the old hand-rolled scaler this tier deleted disagreeing with
`FeaturePipeline` -- moves scores by tenths, not 1e-4 (see FINDINGS-adjacent
numbers in `inference.predict`'s module docstring: FPR moves from 0.0215 to
0.1875 between threshold 0.980 and 0.5, a difference the scale of "wrong
model entirely", not floating-point jitter). So this tolerance stays tight
enough to catch the failure mode this test is for, while not flagging
ordinary numerical noise as one.

SKIPS CLEANLY on a fresh clone, never fails: neither the shipping checkpoint
(models/pe-core-l__linear__allsev_e1.pt) nor `data/corpora/wildrf`'s actual
image bytes are committed to git -- `data/manifests/resolved/wildrf_test.csv`
is (per AGENTS.md's data/ hierarchy: a manifest's CSV is committed, the
corpus images it points at are gitignored). The extension-build check at the
bottom of this file skips the same way when Node/node_modules aren't present.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from aigc_detect.config import ROOT_DIR
from aigc_detect.config.paths import WILDRF_TEST_MANIFEST
from aigc_detect.data.dataset import resolve_image_path
from aigc_detect.inference.bundle import load_bundle
from aigc_detect.inference.detector import FrozenProbeDetector
from aigc_detect.inference.predict import run_inference

CHECKPOINT = ROOT_DIR / "models" / "pe-core-l__linear__allsev_e1.pt"
N_IMAGES = 12
TOLERANCE = 1e-4  # see module docstring


def _sampled_fixture_paths(n: int) -> list[Path]:
    """Up to `n` real image files from wildrf_test, evenly spaced across the
    manifest (not just its first rows) so the sample spans multiple sources
    (reddit/x/facebook) and both labels -- not that label matters to a
    preprocessing-parity check, but a sample that happens to be all-one-thing
    would be a weaker fixture. Only rows whose file actually exists on this
    machine are returned; the caller decides what an empty/short result means.
    """
    if not WILDRF_TEST_MANIFEST.exists():
        return []
    df = pd.read_csv(WILDRF_TEST_MANIFEST)
    if df.empty:
        return []
    stride = max(len(df) // (n * 4), 1)
    paths: list[Path] = []
    for raw in df.iloc[::stride]["image_path"]:
        p = resolve_image_path(raw)
        if p.exists():
            paths.append(p)
        if len(paths) >= n:
            break
    return paths


@pytest.fixture(scope="module")
def fixture_images() -> list[Path]:
    if not CHECKPOINT.exists():
        pytest.skip(f"shipping checkpoint not on this machine: {CHECKPOINT}")
    paths = _sampled_fixture_paths(N_IMAGES)
    if not paths:
        pytest.skip("wildrf_test image tree not on this machine (fresh clone: data/ is gitignored)")
    return paths


def test_predict_py_and_detector_agree_within_tolerance(tmp_path: Path, fixture_images: list[Path]):
    """The deliverable: score the same N images through `predict.py`'s
    `run_inference` (unmodified -- this test does not reimplement or shortcut
    it) and through `inference.detector.FrozenProbeDetector` (what
    `apps/server/app.py` holds and calls for every `/score*` request -- see
    that module's `/health` handler and `score`/`score_batch`/`score_frame`,
    which do nothing to an image beyond decode-to-PIL before calling
    `detector.score`), and assert every prediction lands within `TOLERANCE`.

    Images are copied into a scratch directory with deterministic
    zero-padded names so `run_inference`'s sorted `find_images` traversal
    lines up index-for-index with `fixture_images`' own order -- the pairing
    between the two result sets is positional, not by re-parsing paths.
    """
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    copied: list[Path] = []
    for i, src in enumerate(fixture_images):
        dst = input_dir / f"{i:03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(dst)

    # -- predict.py's path --------------------------------------------------
    output_path = tmp_path / "preds.json"
    run_inference(input_dir=input_dir, head_path=CHECKPOINT, output_path=output_path, batch_size=len(copied), num_workers=0)
    predict_by_key = {r["image_path"]: r["pred"] for r in json.loads(output_path.read_text())}
    assert len(predict_by_key) == len(copied), "predict.py skipped a fixture image as unreadable"

    # -- the Detector/server path --------------------------------------------
    bundle = load_bundle(CHECKPOINT)
    detector = FrozenProbeDetector.load(bundle)
    images = [Image.open(p).convert("RGB") for p in copied]
    detector_preds = detector.score(images)

    max_diff = 0.0
    for dst, det_pred in zip(copied, detector_preds, strict=True):
        key = dst.relative_to(input_dir).as_posix()
        predict_pred = predict_by_key[key]
        diff = abs(predict_pred - det_pred)
        max_diff = max(max_diff, diff)
        assert predict_pred == pytest.approx(det_pred, abs=TOLERANCE), (
            f"{key}: predict.py={predict_pred:.6f} detector={det_pred:.6f} diff={diff:.2e} "
            f"(tolerance={TOLERANCE:.0e})"
        )
    print(f"[test_parity] {len(copied)} images, max |predict.py - detector| = {max_diff:.2e}")


def test_detector_describe_matches_bundle_threshold():
    """`/health` in `apps/server/app.py` renders `Detector.describe()`
    directly for the extension's `resolveThreshold` -- this pins that
    `describe()` actually carries the bundle's own calibrated threshold
    rather than a stale or invented one."""
    if not CHECKPOINT.exists():
        pytest.skip(f"shipping checkpoint not on this machine: {CHECKPOINT}")
    bundle = load_bundle(CHECKPOINT)
    detector = FrozenProbeDetector.load(bundle)
    info = detector.describe()
    assert info["threshold"] == bundle.threshold
    assert info["threshold_source"] == bundle.threshold_source
    assert info["backbone"] == bundle.backbone.key
    assert info["backbone_revision"] == bundle.backbone.revision
    assert info["head_kind"] == bundle.head_kind


# -- E: the extension-build acceptance test, wired into the Python suite -----
#
# Tier 8a's plan named its own acceptance test: "the extension build produces
# working Chrome and Firefox artifacts from one source tree." That lived only
# as a manual `npm run build` + `npm test` until now. `tests/test_parity.py`
# is the one test file this tier owns, so it lives here rather than in a new
# file -- see this project's Tier 8b file-ownership list. Skips cleanly (not
# fails) when Node or `demo/extension/node_modules` isn't present, same as
# the model/data skips above: a machine set up only for the Python side of
# this project must still get a green `uv run pytest`.

EXTENSION_DIR = ROOT_DIR / "demo" / "extension"

_node_missing_reason = None if shutil.which("node") else "node is not on PATH"
_node_modules_missing_reason = (
    None if (EXTENSION_DIR / "node_modules").is_dir() else "demo/extension/node_modules not installed (run `npm install`)"
)


def _run_npm(script: str, timeout: int) -> subprocess.CompletedProcess:
    # shell=True on Windows is what lets `npm` resolve to `npm.cmd` without
    # hardcoding an interpreter path; harmless on POSIX, where npm is a
    # regular executable either way. No untrusted input reaches this string.
    return subprocess.run(
        f"npm run {script}" if script != "test" else "npm test",
        cwd=EXTENSION_DIR,
        capture_output=True,
        text=True,
        shell=True,
        timeout=timeout,
    )


@pytest.mark.skipif(_node_missing_reason, reason=str(_node_missing_reason))
@pytest.mark.skipif(_node_modules_missing_reason, reason=str(_node_modules_missing_reason))
def test_extension_build_produces_chrome_and_firefox_artifacts():
    """Rebuild from a clean `dist/` and check both targets came out --
    `build.js`'s TARGETS dict is the source of truth for what "both targets"
    means; this test doesn't hardcode a third one."""
    dist_dir = EXTENSION_DIR / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)

    result = _run_npm("build", timeout=120)
    assert result.returncode == 0, f"npm run build failed:\n{result.stdout}\n{result.stderr}"

    for target in ("chrome", "firefox"):
        target_dir = dist_dir / target
        for name in ("manifest.json", "content.js", "background.js", "popup.js", "popup.html", "overlay.css"):
            f = target_dir / name
            assert f.is_file() and f.stat().st_size > 0, f"{target} build is missing/empty {name}"
        manifest = json.loads((target_dir / "manifest.json").read_text())
        assert manifest["manifest_version"] == 3

    chrome_bg = json.loads((dist_dir / "chrome" / "manifest.json").read_text())["background"]
    firefox_manifest = json.loads((dist_dir / "firefox" / "manifest.json").read_text())
    assert "service_worker" in chrome_bg, "chrome build must declare an MV3 service_worker"
    assert "scripts" in firefox_manifest["background"], "firefox build must declare background.scripts"
    assert "browser_specific_settings" in firefox_manifest, "firefox build must carry gecko.id for AMO"


@pytest.mark.skipif(_node_missing_reason, reason=str(_node_missing_reason))
@pytest.mark.skipif(_node_modules_missing_reason, reason=str(_node_modules_missing_reason))
def test_extension_unit_tests_pass():
    """`npm test` (node:test over test/heuristics.test.js) alongside the
    build check above -- a build that succeeds but ships broken heuristics
    logic is not "working artifacts"."""
    result = _run_npm("test", timeout=60)
    assert result.returncode == 0, f"npm test failed:\n{result.stdout}\n{result.stderr}"
