"""`aigc experiment` -- run a training experiment from a declared config.

Thin CLI wrapper: everything that decides what a config MEANS lives in
`aigc_detect.train.experiment` (load/validate/hash/run), not here. Three
subcommands, all read-then-act on the same `experiments/*.yaml` files:

    list   every config under experiments/
    show   one config's RESOLVED form (presets expanded, defaults filled) and
           its hash -- what `run` would actually train on, before spending
           any time doing it
    run    train it, write `data/runs/<run_id>/`
"""

from __future__ import annotations

import json


def cmd_experiment_list(_args):
    from aigc_detect.train.experiment import list_experiments, load_experiment

    names = list_experiments()
    if not names:
        raise SystemExit("[experiment] no configs found under experiments/")
    print(f"{'name':<20} {'trainer':<9} {'backbone':<14} {'views':<16} description")
    print("-" * 100)
    for name in names:
        cfg = load_experiment(name)
        r = cfg["resolved"]
        views = r["views"] if isinstance(r["views"], str) else f"{len(r['views'])} view(s)"
        desc = str(cfg["raw"].get("description", "")).strip().replace("\n", " ")
        print(f"{name:<20} {r['trainer']['kind']:<9} {r['backbone']:<14} {views!s:<16} {desc}")


def cmd_experiment_show(args):
    from aigc_detect.train.experiment import load_experiment

    cfg = load_experiment(args.name)
    print(f"[experiment] {cfg['name']}  ({cfg['path']})")
    print(f"[experiment] config_hash={cfg['config_hash']}")
    print(json.dumps(cfg["resolved"], indent=2))


def cmd_experiment_run(args):
    from aigc_detect.train.experiment import run_experiment

    run_experiment(args.name, out=args.out, log_dir=args.log_dir)


def register_experiment(sub):
    p = sub.add_parser("experiment", help="Declared training runs: config in, run directory out.")
    esub = p.add_subparsers(dest="experiment_command", required=True)

    esub.add_parser("list", help="Every experiment config under experiments/.") \
        .set_defaults(func=cmd_experiment_list)

    p_show = esub.add_parser("show", help="One config's resolved form (presets expanded) and its hash.")
    p_show.add_argument("name", help="Config name under experiments/ (without .yaml), or a path to one.")
    p_show.set_defaults(func=cmd_experiment_show)

    p_run = esub.add_parser("run", help="Train the recipe a config declares; writes data/runs/<run_id>/.")
    p_run.add_argument("name", help="Config name under experiments/ (without .yaml), or a path to one.")
    p_run.add_argument("--out", default=None,
                       help="Also copy the trained bundle here (e.g. models/<name>.pt), on top of "
                            "the run directory's own copy.")
    p_run.add_argument("--log-dir", default=None,
                       help="Also write per-step training curves here (train_loss_steps.csv, "
                            "val_curve.csv) -- folds in scripts/train_instrumented.py. Omit for a "
                            "faster run with only per-epoch console output.")
    p_run.set_defaults(func=cmd_experiment_run)
