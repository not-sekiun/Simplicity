"""`aigc pull` -- fetch a declared source into `data/corpora/<id>/`.

RESUME IS THE DEFAULT, NOT A FLAG. Every fetcher backend is built on
`.pull_state.json` (see `aigc_detect.data.fetchers.base`'s module docstring
for the incident that motivated it), so there is no meaningful "start over"
behaviour to opt out of by default -- a bare `aigc pull run <id>` always
picks up where the last run left off, or starts fresh if there is nothing to
resume. `--force` is the opt-IN: it discards a partial pull deliberately,
the same asymmetry `open_state`'s config-hash check enforces (a changed
config demands `--force` rather than silently mixing two pulls under one
corpus id).

THE AUDIT RUNS BY DEFAULT TOO. `aigc_detect.data.audit.audit_corpus` is cheap
next to the pull itself (a few hundred images, a 256-dim logistic regression)
and is the check that would have caught the SD3 and depth-map-pexels
incidents before either reached training (see `data/audit/__init__.py`).
`--no-audit` exists for iterating on a fetcher itself, not for skipping the
check on a corpus anyone intends to train on.
"""

from __future__ import annotations

import dataclasses

from aigc_detect.data.corpus import all_corpora
from aigc_detect.data.fetchers import CorpusPaths, get_fetcher, open_state, verify_and_repair
from aigc_detect.data.sources import UNREGISTERED, all_sources, get_source


def _on_disk_rows(source_id: str) -> str:
    corpora = all_corpora()
    if source_id not in corpora:
        return "(no corpus.yaml entry)"
    try:
        return f"{len(corpora[source_id].rows()):,}"
    except SystemExit:
        return "-"


def cmd_pull_list(_args):
    sources = all_sources()
    print(f"{'source':<28} {'fetcher':<13} {'rows on disk':>13}  license")
    print("-" * 90)
    for sid, source in sorted(sources.items()):
        print(f"{sid:<28} {source.fetcher:<13} {_on_disk_rows(sid):>13}  {source.license}")
    if UNREGISTERED:
        print(f"\n[pull] deliberately unregistered (see data/sources.py): {sorted(UNREGISTERED)}")


def cmd_pull_show(args):
    source = get_source(args.id)
    print(f"[pull] {source.id}")
    print(f"       fetcher: {source.fetcher}")
    print(f"       license: {source.license}")
    if source.notes:
        print(f"       notes:   {source.notes}")
    print("       config:")
    for key, value in source.config.items():
        print(f"         {key}: {value}")
    dest = CorpusPaths.for_source(source.id)
    print(f"       pulls to: {dest.root}")
    print(f"       on disk:  {_on_disk_rows(source.id)} rows")


def cmd_pull_run(args):
    source = get_source(args.id)
    dest = CorpusPaths.for_source(args.id)

    if args.limit is not None:
        if "cap" in source.config or source.fetcher in ("hf_streaming", "hf_parquet"):
            source = dataclasses.replace(source, config={**source.config, "cap": args.limit})
        else:
            print(f"[pull] --limit has no effect on fetcher '{source.fetcher}' -- ignored")

    fetcher = get_fetcher(source.fetcher)
    state, resumed = open_state(source, dest, force=args.force)
    if resumed:
        print(f"[pull] '{args.id}': resuming from {state.rows_written:,} rows already committed "
              f"({state.rows_scanned:,} scanned)")

    result = fetcher.pull(source, dest, state)
    print(f"[pull] '{args.id}': {result.rows_written:,} rows written, "
          f"{result.rows_scanned:,} scanned, resumed={result.resumed}, completed={result.completed}")
    if result.note:
        print(f"[pull]   {result.note}")

    if not args.no_audit:
        from aigc_detect.data.audit import audit_corpus

        probe = audit_corpus(args.id)
        if probe.skipped:
            print(f"[pull] audit: n={probe.n} -- skipped ({'no rows' if probe.n == 0 else 'insufficient data'})")
        else:
            print(f"[pull] audit: n={probe.n} balanced_acc={probe.balanced_acc:.4f} "
                  f"roc_auc={probe.roc_auc:.4f} -> {probe.verdict}")
            if probe.suspect:
                print(f"[pull]   SUSPECT: '{args.id}' will refuse to resolve into a training manifest "
                      f"until this is looked at (see data/corpora/{args.id}/corpus.yaml).")


def cmd_pull_verify(args):
    dest = CorpusPaths.for_source(args.id)
    report = verify_and_repair(dest)
    print(f"[pull] '{args.id}': {report['rows']:,} rows valid, "
          f"{report['dropped_missing']:,} dropped (file missing), "
          f"{report['orphan_files']:,} orphan file(s) on disk")


def register_pull(sub):
    p = sub.add_parser("pull", help="Fetch a declared source into data/corpora/<id>/.")
    psub = p.add_subparsers(dest="pull_command", required=True)

    psub.add_parser("list", help="Every registered source: fetcher, license, whether it is on disk.")\
        .set_defaults(func=cmd_pull_list)

    p_show = psub.add_parser("show", help="One source's resolved config.")
    p_show.add_argument("id")
    p_show.set_defaults(func=cmd_pull_show)

    p_run = psub.add_parser(
        "run", help="Fetch (or resume fetching) a source into data/corpora/<id>/."
    )
    p_run.add_argument("id")
    p_run.add_argument("--force", action="store_true",
                        help="Discard any partial pull and start clean, instead of resuming.")
    p_run.add_argument("--limit", type=int, default=None,
                        help="Cap rows kept this run (only meaningful for hf_streaming/hf_parquet).")
    p_run.add_argument("--no-audit", action="store_true",
                        help="Skip the blind-probe audit at the end of the pull.")
    p_run.set_defaults(func=cmd_pull_run)

    p_verify = psub.add_parser(
        "verify", help="Reconcile index.csv against what's actually on disk; repair, don't guess."
    )
    p_verify.add_argument("id")
    p_verify.set_defaults(func=cmd_pull_verify)
