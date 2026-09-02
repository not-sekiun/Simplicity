"""Module-scoped diagnostic logging for aigc_detect.

Two kinds of console output exist in this project, and they must never mix:

  (1) PROGRAM OUTPUT -- the reports and tables a user reads and pipes: the
      eval-grid per-view table, check-env's status report, audit summaries,
      predict.py's JSON. This still goes through plain ``print()`` straight
      to stdout, unchanged by this module.
  (2) DIAGNOSTICS -- "stale cache, recomputing", "skipped N unreadable
      files", "checkpoint fallback engaged" and the like: notes about what
      the program is doing, not what it was asked to produce. These go
      through the loggers this module configures.

Diagnostics are written to **stderr**, never stdout. If they shared stdout,
``uv run aigc eval-grid ... > report.csv`` or `` | jq`` would capture
interleaved log lines along with the actual report, silently corrupting
every downstream consumer of that output. Keeping them on stderr means the
two streams can be split with an ordinary redirect (``2>/dev/null``) without
touching a single call site, and a user watching the terminal still sees
both at once.

Call :func:`configure` exactly once, near process start (the CLI entry point
does this in ``main()``); call :func:`get_logger` everywhere else, the same
way ``logging.getLogger`` is normally used.
"""

from __future__ import annotations

import logging
import os

_PACKAGE_LOGGER_NAME = "aigc_detect"
_DEFAULT_LEVEL = "INFO"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Thin wrapper over :func:`logging.getLogger`.

    Exists so call sites read ``from aigc_detect.log import get_logger``
    rather than reaching for the stdlib module directly -- one import to
    grep for, and one place to change if the wrapping ever needs to do more.
    """
    return logging.getLogger(name)


def configure(level: str | None = None) -> None:
    """Install the ``aigc_detect`` logger's one handler.

    Idempotent: calling this more than once (e.g. once from the CLI entry
    point and again from a script that is also runnable standalone) does not
    stack up duplicate handlers or duplicate log lines.

    Level resolution order: the ``level`` argument, else the
    ``AIGC_LOG_LEVEL`` environment variable, else ``INFO``. An unrecognised
    value falls back to ``INFO`` with a warning rather than raising --
    a typoed env var should degrade, not crash an unrelated command.
    """
    global _configured

    raw_level = level or os.environ.get("AIGC_LOG_LEVEL") or _DEFAULT_LEVEL
    resolved = logging.getLevelName(raw_level.strip().upper())
    if not isinstance(resolved, int):
        logging.getLogger(_PACKAGE_LOGGER_NAME).warning(
            "unrecognised log level %r, falling back to %s", raw_level, _DEFAULT_LEVEL
        )
        resolved = logging.getLevelName(_DEFAULT_LEVEL)

    package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    package_logger.setLevel(resolved)
    # Records must not also reach the root logger's handler (if any is ever
    # installed) -- this package owns its own single handler below, and
    # propagating would print every line twice.
    package_logger.propagate = False

    if _configured:
        return

    handler = logging.StreamHandler()  # stderr by default -- see module docstring
    handler.addFilter(_ShortNameFilter())
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    package_logger.addHandler(handler)
    _configured = True


class _ShortNameFilter(logging.Filter):
    """Strips the redundant leading ``aigc_detect.`` off a logger's name.

    Every logger in this project is named after its module
    (``aigc_detect.embed.views``), but the package prefix is implied by
    every line ever emitted here, so it is dropped for readability -- the
    formatted record reads ``embed.views: ...`` rather than
    ``aigc_detect.embed.views: ...``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == _PACKAGE_LOGGER_NAME:
            record.name = "aigc_detect"
        elif record.name.startswith(_PACKAGE_LOGGER_NAME + "."):
            record.name = record.name[len(_PACKAGE_LOGGER_NAME) + 1 :]
        return True
