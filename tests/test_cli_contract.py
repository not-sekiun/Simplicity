"""The CLI surface, pinned.

This exists to make restructuring safe rather than to test behaviour. The
package is being reorganised into subpackages and `main.py` is being split into
one module per subcommand; none of that is allowed to change what a user can
type. So the subcommand list and each subcommand's option names are asserted
here, and the tests are deliberately written against the *invocation* (a
subprocess running the entry point) rather than against any import path, so
they keep working no matter where the code moves.

Run: uv run pytest
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every subcommand `main.py --help` advertised before the restructure began.
EXPECTED_SUBCOMMANDS = [
    "check-env",
    "download",
    "split",
    "preview-augment",
    "download-demo",
    "build-demo-val",
    "download-ood",
    "build-ood",
    "build-heldout",
    "audit-data",
    "list-backbones",
    "embed",
    "embed-views",
    "train-head-views",
    "eval-grid",
    "error-analysis",
    "train-head",
    "predict",
]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_root_help_succeeds():
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("command", EXPECTED_SUBCOMMANDS)
def test_subcommand_is_still_reachable(command: str):
    """Each subcommand still parses `--help` and exits cleanly.

    Catches the most likely restructuring failure: a command whose module moved
    but whose registration was not updated, which argparse reports as an
    'invalid choice' exit rather than an import error.
    """
    proc = run_cli(command, "--help")
    assert proc.returncode == 0, f"`main.py {command} --help` failed:\n{proc.stderr}"


def test_no_subcommand_was_silently_added_or_dropped():
    proc = run_cli("--help")
    advertised = {c for c in EXPECTED_SUBCOMMANDS if c in proc.stdout}
    missing = set(EXPECTED_SUBCOMMANDS) - advertised
    assert not missing, f"subcommands no longer advertised in --help: {sorted(missing)}"


def test_every_package_module_imports():
    """Import every module in the package.

    A moved module that nothing imports yet will not fail any other test; this
    walks the whole tree so a broken import surfaces immediately.
    """
    import aigc_detect

    failures = []
    for mod in pkgutil.walk_packages(aigc_detect.__path__, prefix="aigc_detect."):
        try:
            __import__(mod.name)
        except Exception as exc:  # noqa: BLE001 - reporting all failures beats the first
            failures.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_package_is_importable_without_path_manipulation():
    """`import aigc_detect` works on its own.

    Before the packaging fix, 23 files opened with a `sys.path.insert(0,
    .../src)` preamble. This asserts the fix stays in place: a plain subprocess
    with no such preamble can import the package.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import aigc_detect; print(aigc_detect.__name__)"],
        capture_output=True,
        text=True,
        cwd=ROOT.parent,  # deliberately NOT the repo root
    )
    assert proc.returncode == 0, proc.stderr
    assert "aigc_detect" in proc.stdout
