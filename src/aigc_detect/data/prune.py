"""The orphan sweep: which images on disk no manifest references.

Tier 1 reclaimed 35 GB by deleting whole things -- raced-and-lost backbone
towers, a duplicate dataset cache, a rejected corpus. What it could not touch
was the inside of a corpus: files that are simply not referenced by any of the
thirteen manifests. Finding those needs manifests that resolve deterministically,
which is why this is Tier 5's step and not Tier 1's.

READ-ONLY BY CONSTRUCTION. Nothing here deletes. It reports, and the report is
the input to a decision a person makes -- because "no manifest references it" is
evidence, not proof: a corpus can hold images a future recipe would want, and
two of the corpora here cannot be re-fetched at all (`wildfake_dalle_advanced`
was a manual browser pull from ModelScope, which is not reachable
programmatically from this machine).

WHY THE SWEEP MUST FOLLOW THE CACHE MIGRATION. Migration re-derives every cached
row's identity by hashing the file it came from, so it needs those files on
disk. Delete first and 1.4 GB of GPU-hours becomes unrecoverable. Tier 4 has
run, so that ordering constraint is discharged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aigc_detect.data.corpus import Corpus, all_corpora
from aigc_detect.data.manifest import list_recipes, resolve
from aigc_detect.log import get_logger

logger = get_logger(__name__)

#: What counts as an image when walking a corpus directory. Anything else found
#: under an images root is reported separately rather than counted as an orphan:
#: a stray .rar or .txt is a different kind of finding from an unused photo.
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp"})


@dataclass
class CorpusUsage:
    corpus: Corpus
    on_disk: int
    referenced: int
    orphan_bytes: int
    orphans: list[Path]
    non_images: list[Path]
    missing: int  # referenced by a manifest but absent from disk

    @property
    def orphan_count(self) -> int:
        return len(self.orphans)


def referenced_paths(manifests: Iterable[str] | None = None) -> set[str]:
    """Every image path any manifest resolves to, normalized for comparison.

    Resolution, not the committed CSVs: a recipe is the live definition, and a
    stale resolved CSV would make a still-wanted image look orphaned.
    """
    names = list(manifests) if manifests is not None else list_recipes()
    out: set[str] = set()
    for name in names:
        df = resolve(name)
        out.update(_normalize(p) for p in df["image_path"].astype(str))
    return out


def _normalize(path: str | Path) -> str:
    """Case-folded resolved path.

    Windows paths differ in case between a manifest written by one script and a
    directory walk done by another; treating those as different files would
    report every image in the project as an orphan.
    """
    return str(Path(str(path)).resolve()).casefold()


def sweep(manifests: Iterable[str] | None = None) -> list[CorpusUsage]:
    """Measure every registered corpus against what the manifests actually use."""
    wanted = referenced_paths(manifests)
    results: list[CorpusUsage] = []

    for _cid, corpus in sorted(all_corpora().items()):
        if corpus.images is None or not corpus.images.is_dir():
            continue
        orphans: list[Path] = []
        non_images: list[Path] = []
        on_disk = 0
        referenced = 0
        orphan_bytes = 0

        for path in corpus.images.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                non_images.append(path)
                continue
            on_disk += 1
            if _normalize(path) in wanted:
                referenced += 1
            else:
                orphans.append(path)
                orphan_bytes += path.stat().st_size

        try:
            indexed = {_normalize(p) for p in corpus.rows()["image_path"].astype(str)}
            missing = sum(1 for p in indexed if p in wanted and not Path(p).exists())
        except SystemExit:
            missing = 0

        results.append(CorpusUsage(corpus, on_disk, referenced, orphan_bytes,
                                   orphans, non_images, missing))
    return results


def report(usages: list[CorpusUsage]) -> None:
    print(f"{'corpus':<26} {'role':<11} {'on disk':>9} {'used':>9} {'orphans':>9} {'MB':>9}")
    print("-" * 80)
    total_orphans = total_bytes = 0
    for u in usages:
        print(f"{u.corpus.id:<26} {u.corpus.role:<11} {u.on_disk:>9,} {u.referenced:>9,} "
              f"{u.orphan_count:>9,} {u.orphan_bytes / 1e6:>9.1f}")
        total_orphans += u.orphan_count
        total_bytes += u.orphan_bytes
    print("-" * 80)
    print(f"{'TOTAL':<26} {'':<11} {'':>9} {'':>9} {total_orphans:>9,} {total_bytes / 1e6:>9.1f}")

    stray = [(u.corpus.id, p) for u in usages for p in u.non_images]
    if stray:
        print(f"\n[prune] {len(stray)} non-image file(s) under corpus image roots:")
        by_size = sorted(stray, key=lambda t: t[1].stat().st_size, reverse=True)[:10]
        for cid, path in by_size:
            print(f"[prune]   {path.stat().st_size / 1e6:>9.1f} MB  {cid}: {path.name}")

    broken = [u for u in usages if u.missing]
    for u in broken:
        print(f"\n[prune] WARNING: {u.corpus.id} -- {u.missing:,} manifest row(s) point at "
              f"files that are not on disk")
