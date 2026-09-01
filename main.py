#!/usr/bin/env python
"""Thin shim.

The real CLI implementation lives in aigc_detect.cli. This file exists only
so the documented invocation `uv run main.py <command>` keeps working.
"""

from __future__ import annotations

from aigc_detect.cli import main

if __name__ == "__main__":
    main()
