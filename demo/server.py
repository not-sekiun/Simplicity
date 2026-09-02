#!/usr/bin/env python
"""Thin shim.

The real implementation lives in `apps/server/app.py` (tier 8b packaged the
demo server the same way `main.py`/`predict.py` package the CLI and the
graded inference entry point -- a real module under `apps/`/`src/`, plus a
root-relative shim so a documented invocation keeps working). This file
exists only so `uv run python demo/server.py ...` -- what `demo/README.md`
has always told people to type -- keeps working verbatim. Prefer
`uv run aigc-serve ...` (the console script `pyproject.toml` registers) for
anything new; both run the identical `apps.server.app.main`.
"""

from __future__ import annotations

from apps.server.app import main

if __name__ == "__main__":
    main()
