"""`aigc manifest ...` -- inspect and materialize manifest recipes.

Additive on purpose. `split`, `build-ood`, `build-heldout` and `build-demo-val`
still work and still write where they always wrote; this group reads the same
data through the recipe engine so the two can be compared before the old path
is retired. `manifest check` is that comparison, and it is the command that says
whether the Tier 5 migration is safe to finish.
"""

from __future__ import annotations

import pandas as pd

from aigc_detect.config import DATA_DIR
from aigc_detect.data.corpus import all_corpora
from aigc_detect.data.manifest import list_recipes, load_recipe, resolve, resolved_path, write_resolved

#: Where each manifest lived before Tier 5. Mirrors tests/test_manifests.py.
LEGACY = {
    "train": "processed/train.csv", "val": "processed/val.csv",
    "train_ext": "processed/train_ext.csv", "heldout": "heldout/heldout.csv",
    "ood": "ood/ood.csv", "demo_val": "demo_val/demo_val.csv",
    "wildrf_test": "wildrf/wildrf_test.csv", "wildrf_real": "processed/wildrf_real.csv",
    "sid_real": "processed/sid_real.csv", "unsplash_real": "processed/unsplash_real.csv",
    "nano_banana": "processed/nano_banana.csv", "midjourney_v6": "processed/midjourney_v6.csv",
    "dalle3_holdout": "processed/dalle3_holdout.csv",
}


def cmd_manifest_list(_args):
    names = list_recipes()
    if not names:
        raise SystemExit("[manifest] no recipes found under data/manifests")
    print(f"{'manifest':<18} {'rows':>8}  {'train?':<7} description")
    print("-" * 88)
    for name in names:
        recipe = load_recipe(name)
        out = resolved_path(name)
        rows = f"{len(pd.read_csv(out)):,}" if out.exists() else "-"
        flag = "NEVER" if recipe.never_train else "yes"
        print(f"{name:<18} {rows:>8}  {flag:<7} {recipe.spec.get('description', '')}")


def cmd_manifest_show(args):
    recipe = load_recipe(args.name)
    df = resolve(args.name)
    print(f"[manifest] {args.name}: {len(df):,} rows")
    print(f"[manifest] {recipe.spec.get('description', '')}")
    if recipe.never_train:
        print("[manifest] NEVER TRAIN -- this is an evaluation tier")
    print(f"[manifest] include: {recipe.includes}")
    print(f"\n[manifest] labels: {df['label'].value_counts().to_dict()}")
    print(f"[manifest] sources: {df['source'].value_counts().to_dict()}")
    gens = df["generator"].value_counts()
    print(f"[manifest] generators ({len(gens)}): {dict(sorted(gens.items()))}")


def cmd_manifest_resolve(args):
    names = list_recipes() if args.name == "all" else [args.name]
    for name in names:
        out = write_resolved(name)
        print(f"[manifest] {name:<18} {len(pd.read_csv(out)):>8,} rows -> {out}")


def cmd_manifest_check(_args):
    """Compare every recipe against the CSV it replaces, row for row.

    This is Tier 5's acceptance test in command form. A recipe that merely
    selects the same SET of images is not good enough: row order decides what
    `--limit` takes and what the manifest fingerprint hashes.
    """
    failures = 0
    print(f"{'manifest':<18} {'resolved':>9} {'legacy':>9}  verdict")
    print("-" * 62)
    for name in list_recipes():
        legacy = DATA_DIR / LEGACY[name] if name in LEGACY else None
        got = resolve(name)
        if legacy is None or not legacy.exists():
            print(f"{name:<18} {len(got):>9,} {'-':>9}  no legacy file to compare")
            continue
        ref = pd.read_csv(legacy)
        cols = [c for c in ("image_path", "label", "source", "generator") if c in ref.columns]
        same = (got[cols].reset_index(drop=True).astype(str)
                .equals(ref[cols].reset_index(drop=True).astype(str)))
        if same:
            verdict = "identical, row for row"
        elif set(got.image_path) == set(ref.image_path):
            verdict = "SAME IMAGES, DIFFERENT ORDER"
            failures += 1
        else:
            only_new = len(set(got.image_path) - set(ref.image_path))
            only_old = len(set(ref.image_path) - set(got.image_path))
            verdict = f"DIFFERENT: +{only_new} / -{only_old} images"
            failures += 1
        print(f"{name:<18} {len(got):>9,} {len(ref):>9,}  {verdict}")
    if failures:
        raise SystemExit(f"\n[manifest] {failures} recipe(s) do not reproduce their legacy manifest")
    print("\n[manifest] every recipe reproduces its legacy manifest row for row")


def cmd_corpus_list(_args):
    print(f"{'corpus':<26} {'role':<11} {'rows':>8}  images")
    print("-" * 92)
    for cid, corpus in sorted(all_corpora().items(), key=lambda kv: (kv[1].role, kv[0])):
        try:
            rows = f"{len(corpus.rows()):,}"
        except SystemExit:
            rows = "-"
        images = str(corpus.images.relative_to(DATA_DIR)) if corpus.images else "(none)"
        print(f"{cid:<26} {corpus.role:<11} {rows:>8}  {images}")


def register_manifest(sub):
    p = sub.add_parser("manifest", help="Inspect and materialize manifest recipes.")
    msub = p.add_subparsers(dest="manifest_command", required=True)

    msub.add_parser("list", help="Every recipe, its size and whether it may be trained on.")\
        .set_defaults(func=cmd_manifest_list)

    p_show = msub.add_parser("show", help="One recipe's composition and label/source/generator mix.")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_manifest_show)

    p_res = msub.add_parser("resolve", help="Write data/manifests/resolved/<name>.csv.")
    p_res.add_argument("name", help="Recipe name, or 'all'.")
    p_res.set_defaults(func=cmd_manifest_resolve)

    msub.add_parser(
        "check",
        help="Verify every recipe reproduces the manifest it replaces, row for row.",
    ).set_defaults(func=cmd_manifest_check)


def cmd_corpus_orphans(args):
    """Images on disk that no manifest references. REPORTS, never deletes.

    "Unreferenced" is evidence, not a verdict -- see aigc_detect.data.prune for
    why that difference matters here, and read the report before acting on it.
    """
    from aigc_detect.data.prune import report, sweep

    usages = sweep()
    report(usages)
    if args.list:
        for u in usages:
            if not u.orphans:
                continue
            print(f"\n[prune] {u.corpus.id} -- {u.orphan_count:,} unreferenced:")
            for path in u.orphans[: args.list]:
                print(f"[prune]   {path}")
            if u.orphan_count > args.list:
                print(f"[prune]   ... and {u.orphan_count - args.list:,} more")


def register_corpus(sub):
    p = sub.add_parser("corpus", help="The corpus registry and what of it is actually used.")
    csub = p.add_subparsers(dest="corpus_command", required=True)

    csub.add_parser("list", help="Every registered corpus: role, row count, image root.")\
        .set_defaults(func=cmd_corpus_list)

    p_orph = csub.add_parser(
        "orphans",
        help="Images on disk that no manifest references (reports only, deletes nothing).",
    )
    p_orph.add_argument("--list", type=int, default=0, metavar="N",
                        help="Also print the first N unreferenced paths per corpus.")
    p_orph.set_defaults(func=cmd_corpus_orphans)
