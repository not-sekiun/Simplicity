"""`aigc cache ...` -- inspect, migrate and verify the embedding store."""

from __future__ import annotations

from aigc_detect.config import EMBEDDINGS_DIR, get_settings


def _open():
    from aigc_detect.cache.hashing import HashCache
    from aigc_detect.cache.store import EmbeddingStore

    settings = get_settings()
    return (
        EmbeddingStore(settings.store_root),
        HashCache(settings.hash_db_path),
        settings.cache_root,
    )


def cmd_cache_status(_args):
    store, hashes, root = _open()
    try:
        s = store.stats()
        print(f"[cache] root {root}")
        print(f"[cache] {s['rows']:,} vectors, {s['bytes'] / 1e9:.2f} GB on disk")
        h = hashes.stats()
        print(f"[cache] {h['paths']:,} hashed paths -> {h['distinct_ids']:,} distinct images")
        if s["per_backbone_view"]:
            print(f"\n{'backbone':<16} {'view':<20} {'rows':>10}")
            print("-" * 48)
            for key, view, n in s["per_backbone_view"]:
                print(f"{key:<16} {view:<20} {n:>10,}")
    finally:
        store.close()
        hashes.close()


def cmd_cache_migrate(args):
    from aigc_detect.cache.migrate import migrate

    store, hashes, _ = _open()
    try:
        result = migrate(EMBEDDINGS_DIR, store, hashes,
                         dry_run=args.dry_run, limit_files=args.limit_files)
        print(f"\n[cache] migrated {result['migrated_rows']:,} rows from {result['files']} files")
        if not args.dry_run:
            print(f"[cache] {result['deferred_rows']:,} stochastic rows deferred "
                  f"(recomputed on first use -- they were path-seeded)")
    finally:
        store.close()
        hashes.close()


def cmd_cache_verify(args):
    from aigc_detect.cache.verify import verify_sample

    store, hashes, _ = _open()
    try:
        ok = verify_sample(store, hashes, n_images=args.sample, seed=args.seed)
    finally:
        store.close()
        hashes.close()
    raise SystemExit(0 if ok else 1)


def cmd_cache_export(args):
    from aigc_detect.cache.export import export

    store, hashes, _ = _open()
    try:
        result = export(store, args.out, vectors=args.vectors,
                        backbone=args.backbone, view=args.view)
    finally:
        store.close()
        hashes.close()
    print(f"[cache] exported {result['rows']:,} rows from {result['groups']} "
          f"(backbone, view) group(s) -> {result['out_dir']}")


def cmd_cache_compact(args):
    store, hashes, _ = _open()
    try:
        result = store.compact(dry_run=args.dry_run)
    finally:
        store.close()
        hashes.close()
    verb = "would reclaim" if args.dry_run else "reclaimed"
    print(f"[cache] {verb} {result['bytes_reclaimed'] / 1e6:.1f} MB "
          f"across {result['shards_rewritten']} shard(s)")


def cmd_cache_merge(args):
    store, hashes, _ = _open()
    try:
        n = store.merge(args.other)
        print(f"[cache] merged {n:,} rows from {args.other}")
    finally:
        store.close()
        hashes.close()


def register_cache(sub):
    p = sub.add_parser("cache", help="Inspect, migrate and verify the embedding store.")
    csub = p.add_subparsers(dest="cache_command", required=True)

    csub.add_parser("status", help="Row counts and disk usage.").set_defaults(func=cmd_cache_status)

    p_mig = csub.add_parser(
        "migrate",
        help="Import the legacy data/embeddings/*.npz caches into the content-addressed store.",
    )
    p_mig.add_argument("--dry-run", action="store_true", help="Report what would migrate, write nothing.")
    p_mig.add_argument("--limit-files", type=int, default=None, help="Migrate only the first N files.")
    p_mig.set_defaults(func=cmd_cache_migrate)

    p_ver = csub.add_parser(
        "verify",
        help="Re-embed a random sample and compare against the store (catches a bad migration).",
    )
    p_ver.add_argument("--sample", type=int, default=200, help="Images to re-embed (default 200).")
    p_ver.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p_ver.set_defaults(func=cmd_cache_verify)

    p_exp = csub.add_parser(
        "export",
        help="Dump the index (and optionally the vectors) to CSV/npy for inspection.",
    )
    p_exp.add_argument("--out", required=True, help="Directory to write into (created if absent).")
    p_exp.add_argument("--vectors", action="store_true",
                       help="Also write one .npy matrix + ids.txt per (backbone, view) group.")
    p_exp.add_argument("--backbone", default=None, help="Only this backbone key, e.g. pe-core-l.")
    p_exp.add_argument("--view", default=None, help="Only this view name, e.g. clean.")
    p_exp.set_defaults(func=cmd_cache_export)

    p_cmp = csub.add_parser(
        "compact",
        help="Reclaim shard bytes left behind by an interrupted write.",
    )
    p_cmp.add_argument("--dry-run", action="store_true", help="Report the reclaimable bytes only.")
    p_cmp.set_defaults(func=cmd_cache_compact)

    p_mrg = csub.add_parser("merge", help="Fold another store (e.g. from a second machine) into this one.")
    p_mrg.add_argument("other", help="Path to the other store's root directory.")
    p_mrg.set_defaults(func=cmd_cache_merge)
