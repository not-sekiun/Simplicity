# CLAUDE.md

Project context, structure, current build status, and key decisions live in
[AGENTS.md](AGENTS.md) — read that first before making changes.

A few Claude-Code-specific notes on top of it:

- This is a **uv-managed** project. Always run code as `uv run main.py ...`
  or `uv run python ...` — never a bare `python`/`pip` call.
- `data/` is gitignored and mostly not present after a fresh clone; don't
  assume downloaded datasets exist without checking (`uv run main.py
  check-env` reports what's there).
- Solo hackathon submission repo with a real deadline — keep commits small
  and scoped to what was asked; don't rewrite history unprompted.
