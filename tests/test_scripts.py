"""`scripts/` is not covered by any other test, and it rots silently.

`tests/test_cli_contract.test_every_package_module_imports` walks
`aigc_detect.__path__`, so it never sees `scripts/`. That gap had already cost
something by the time this file was written: `scripts/worker.py` shelled out to
`main.py build-ood` and `main.py build-heldout` -- two commands retired when
manifest recipes replaced the hand-written builders -- and later grew an import
of a `download_*.py` module that the fetcher registry deleted. Neither break
failed a test, and neither was visible from the outside: the import that would
have raised sat INSIDE a function, so even `import scripts.worker` succeeded.
The file was dead for two tiers before anyone noticed.

The surviving scripts are each the only record of something -- how the backbone
race was driven, how a corpus was audited before the gate existed, how a run
directory is plotted. That is why they are kept rather than deleted, and it is
exactly why they need a test: a script nobody runs weekly is a script whose
breakage nobody sees.

WHY IMPORT AND NOT EXECUTE. Every one of these does real work when run -- pulls,
GPU passes, a matplotlib window. Importing proves the module still resolves
against the package as it stands today, which is the failure mode that actually
happens during a restructure. It does not prove the script still WORKS; nothing
cheap does, and claiming otherwise would be worse than this test's honest,
narrower promise.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# Modules whose imports live in an optional dependency group, so a bare
# `uv sync` legitimately cannot import them. Anything NOT listed here must
# import against the base dependency set.
OPTIONAL_DEPS = {
    "plot_run": ("matplotlib", "the `viz` extra -- uv sync --extra viz"),
}


def _script_names() -> list[str]:
    return sorted(
        m.name for m in pkgutil.iter_modules([str(SCRIPTS_DIR)]) if not m.name.startswith("_")
    )


def test_there_are_scripts_to_check():
    """A guard on the guard: an empty parametrize list passes vacuously, and a
    renamed or moved `scripts/` directory would make every test below silently
    test nothing."""
    assert _script_names(), f"no importable modules found under {SCRIPTS_DIR}"


@pytest.mark.parametrize("name", _script_names())
def test_script_still_imports(name: str):
    try:
        importlib.import_module(f"scripts.{name}")
    except ImportError as exc:
        missing, hint = OPTIONAL_DEPS.get(name, (None, None))
        if missing and missing in str(exc):
            pytest.skip(f"scripts/{name}.py needs {missing}, which lives in {hint}")
        raise
