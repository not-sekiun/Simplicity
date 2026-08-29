"""Error analysis (deliverable 5.5.5): concrete false positives/negatives,
a per-generator collapse ranking, and the clean-vs-worst-view trade-off, for
one trained head -- built entirely from the robustness-view embedding caches
`embed_views.py` already wrote. No forward pass runs here.

`eval_grid.py` (deliverable 5.5.4) answers "how much does accuracy degrade."
This module answers the question a judge asks next: "degrade on WHAT,
specifically" -- named images, named generators, at what confidence, and
what does that cost in false accepts vs false rejects. It reuses eval_grid's
`best_balanced_threshold` so every number here is read at the exact same
fixed operating point the robustness grid reports, not a re-tuned one.

Two views anchor the report: `clean` (best case) and whichever cached view
has the lowest AUC (empirically `chain_heavy`, HANDOFF section 3 / NARRATIVE
Run 7 -- but this is computed, never hardcoded, so it stays correct if a
future head or manifest shifts the worst view). Per-generator AUC pools every
cached degraded view, matching eval_grid's `--by-generator` "degraded" column,
so these numbers reconcile with `reports/race/<backbone>/run.log` rather than
introducing a second definition.

Usage:
    uv run main.py error-analysis --backbone pe-core-l --manifest ood \\
        --sample-rows 4000 --head models/pe-core-l__linear__augchain.pt
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from aigc_detect.config import GENERATOR_FAMILY, RANDOM_SEED, ROOT_DIR, TRAIN_GENERATORS
from aigc_detect.embed import fingerprint_paths
from aigc_detect.embed_views import cache_stem, load_view_cache, select_rows, view_embeddings_path
from aigc_detect.eval_grid import best_balanced_threshold
from aigc_detect.heads import build_head
from aigc_detect.transforms import build_robustness_views, eval_view_names


def _resolve_image_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else ROOT_DIR / p


def run_error_analysis(
    backbone_key: str,
    manifest_path: str | Path,
    head_path: str | Path,
    limit: int | None = None,
    sample_rows: int | None = None,
    sample_seed: int = RANDOM_SEED,
    top_k: int = 8,
    extra_views: tuple[str, ...] = (),
    out_dir: str | Path | None = None,
    copy_images: bool = True,
) -> dict:
    manifest_path, head_path = Path(manifest_path), Path(head_path)
    stem = cache_stem(manifest_path, limit=limit, sample_rows=sample_rows)

    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
    if ckpt.get("backbone") != backbone_key:
        raise SystemExit(
            f"[error-analysis] head {head_path.name} was trained on backbone "
            f"'{ckpt.get('backbone')}', not '{backbone_key}'."
        )
    head = build_head(ckpt["head_kind"], ckpt["in_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)

    print(f"[error-analysis] head={head_path.name} backbone={backbone_key} manifest={manifest_path.name} "
          f"cache_stem={stem}")

    df = select_rows(manifest_path, limit=limit, sample_rows=sample_rows, sample_seed=sample_seed)
    expected_m_fp = fingerprint_paths(df["image_path"])

    _, all_specs = build_robustness_views()
    specs = {n: all_specs[n] for n in eval_view_names()}

    loaded: dict[str, dict] = {}
    missing = []
    for name, spec in specs.items():
        path = view_embeddings_path(backbone_key, stem, name)
        if not path.exists():
            missing.append(name)
            continue
        emb, labels, meta = load_view_cache(backbone_key, stem, name, spec, expected_manifest_fp=expected_m_fp)
        with torch.no_grad():
            x = torch.from_numpy((emb - scaler_mean) / scaler_std)
            probs = torch.sigmoid(head(x).squeeze(-1)).numpy()
        loaded[name] = {"labels": labels, "probs": probs, "meta": meta, "auc": float(roc_auc_score(labels, probs))}
    if missing:
        print(f"[error-analysis] {len(missing)} view(s) not cached, skipped: {', '.join(missing)}")
    if "clean" not in loaded:
        raise SystemExit(
            "[error-analysis] the 'clean' view is not cached -- it defines the threshold. Run "
            f"`uv run main.py embed-views --backbone {backbone_key} --manifest "
            f"{manifest_path.stem} --sample-rows {sample_rows or ''}`."
        )

    clean = loaded["clean"]
    threshold = best_balanced_threshold(clean["labels"], clean["probs"])
    n_real = int((clean["labels"] == 0).sum())
    n_aigc = int((clean["labels"] == 1).sum())
    print(f"[error-analysis] n={len(clean['labels'])} ({n_real} real / {n_aigc} aigc), "
          f"fixed threshold {threshold:.4f}")

    degraded_names = [n for n in loaded if n != "clean"]
    worst_view = min(degraded_names, key=lambda n: loaded[n]["auc"]) if degraded_names else None
    if worst_view:
        print(f"[error-analysis] worst cached view: {worst_view} (AUC {loaded[worst_view]['auc']:.4f}, "
              f"vs clean {clean['auc']:.4f})")

    focus_views = list(dict.fromkeys(["clean"] + ([worst_view] if worst_view else []) + list(extra_views)))
    focus_views = [v for v in focus_views if v in loaded]

    # --- trade-off summary: FPR/FNR at the ONE fixed threshold, per focus view ---
    tradeoffs = []
    for name in focus_views:
        v = loaded[name]
        preds = (v["probs"] >= threshold).astype(np.int64)
        real_mask, aigc_mask = v["labels"] == 0, v["labels"] == 1
        fpr = float(preds[real_mask].mean()) if real_mask.any() else float("nan")
        fnr = float(1.0 - preds[aigc_mask].mean()) if aigc_mask.any() else float("nan")
        tradeoffs.append({"view": name, "auc": v["auc"], "fpr": fpr, "fnr": fnr})

    # --- per-generator collapse ranking (pooled over every cached degraded view) ---
    generators = clean["meta"].get("generators")
    gen_table = []
    if generators is not None and any(g for g in generators):
        real_mask = clean["labels"] == 0
        for gen in sorted(set(generators)):
            if not gen or GENERATOR_FAMILY.get(gen) == "real":
                continue
            gmask = (generators == gen) & (clean["labels"] == 1)
            n = int(gmask.sum())
            if n < 10:
                continue
            sel = gmask | real_mask
            y = clean["labels"][sel]
            a_clean = float(roc_auc_score(y, clean["probs"][sel]))
            if degraded_names:
                deg_p = np.concatenate([loaded[n2]["probs"][sel] for n2 in degraded_names])
                deg_y = np.concatenate([y for _ in degraded_names])
                a_deg = float(roc_auc_score(deg_y, deg_p))
            else:
                a_deg = float("nan")
            gen_table.append({
                "generator": gen,
                "family": GENERATOR_FAMILY.get(gen, "unknown"),
                "seen": "trained" if gen in TRAIN_GENERATORS else "UNSEEN",
                "n": n,
                "clean_auc": a_clean,
                "degraded_auc": a_deg,
                "drop": a_clean - a_deg,
            })
        gen_table.sort(key=lambda r: r["drop"], reverse=True)

    # --- concrete false positive / false negative examples, per focus view ---
    examples = []
    for name in focus_views:
        v = loaded[name]
        paths = v["meta"].get("image_paths")
        gens = v["meta"].get("generators")
        labels, probs = v["labels"], v["probs"]
        preds = (probs >= threshold).astype(np.int64)

        fp_idx = np.flatnonzero((labels == 0) & (preds == 1))
        fp_idx = fp_idx[np.argsort(-probs[fp_idx])][:top_k]  # most-confidently-wrong real->AIGC first
        fn_idx = np.flatnonzero((labels == 1) & (preds == 0))
        fn_idx = fn_idx[np.argsort(probs[fn_idx])][:top_k]  # most-confidently-wrong AIGC->real first

        for kind, idxs in (("false_positive", fp_idx), ("false_negative", fn_idx)):
            for rank, i in enumerate(idxs, start=1):
                examples.append({
                    "view": name,
                    "kind": kind,
                    "rank": rank,
                    "image_path": str(paths[i]) if paths is not None else "",
                    "generator": str(gens[i]) if gens is not None else "",
                    "label": int(labels[i]),
                    "pred": float(probs[i]),
                })

    out_dir = Path(out_dir) if out_dir else ROOT_DIR / "reports" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{backbone_key}__{stem}__{head_path.stem}"

    examples_csv = out_dir / f"examples__{tag}.csv"
    with examples_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["view", "kind", "rank", "image_path", "generator", "label", "pred"])
        w.writeheader()
        for row in examples:
            w.writerow({**row, "pred": f"{row['pred']:.6f}"})
    print(f"[error-analysis] {len(examples)} example(s) -> {examples_csv}")

    by_generator_csv = None
    if gen_table:
        by_generator_csv = out_dir / f"by_generator__{tag}.csv"
        with by_generator_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["generator", "family", "seen", "n", "clean_auc", "degraded_auc", "drop"]
            )
            w.writeheader()
            for row in gen_table:
                w.writerow({
                    **row,
                    "clean_auc": f"{row['clean_auc']:.6f}",
                    "degraded_auc": f"{row['degraded_auc']:.6f}",
                    "drop": f"{row['drop']:.6f}",
                })
        print(f"[error-analysis] per-generator table -> {by_generator_csv}")

    copied = 0
    if copy_images:
        img_dir = out_dir / f"examples__{tag}"
        img_dir.mkdir(parents=True, exist_ok=True)
        for row in examples:
            if not row["image_path"]:
                continue
            src = _resolve_image_path(row["image_path"])
            dst = img_dir / f"{row['view']}__{row['kind']}__rank{row['rank']}__{src.name}"
            try:
                shutil.copyfile(src, dst)
                copied += 1
            except OSError as exc:
                print(f"[error-analysis] WARNING: could not copy {src}: {exc}")
        print(f"[error-analysis] copied {copied}/{len(examples)} example image(s) -> {img_dir}")

    report_md = out_dir / f"report__{tag}.md"
    _write_markdown_report(
        report_md, backbone_key=backbone_key, head_path=head_path, manifest_path=manifest_path, stem=stem,
        threshold=threshold, n_real=n_real, n_aigc=n_aigc, focus_views=focus_views, tradeoffs=tradeoffs,
        gen_table=gen_table, examples=examples, top_k=top_k,
    )
    print(f"[error-analysis] report -> {report_md}")

    return {
        "threshold": threshold,
        "worst_view": worst_view,
        "tradeoffs": tradeoffs,
        "gen_table": gen_table,
        "examples": examples,
        "examples_csv": str(examples_csv),
        "by_generator_csv": str(by_generator_csv) if by_generator_csv else None,
        "report_md": str(report_md),
        "n_images_copied": copied,
    }


def _write_markdown_report(
    path: Path, *, backbone_key, head_path, manifest_path, stem, threshold, n_real, n_aigc,
    focus_views, tradeoffs, gen_table, examples, top_k,
) -> None:
    lines = []
    lines.append(f"# Error analysis -- {backbone_key} / {head_path.name} / {manifest_path.stem} ({stem})")
    lines.append("")
    lines.append("Deliverable 5.5.5. Auto-generated by `uv run main.py error-analysis` -- "
                  "regenerate rather than hand-edit; add commentary in a separate section of the README.")
    lines.append("")
    lines.append(f"- n = {n_real + n_aigc} ({n_real} real / {n_aigc} AIGC)")
    lines.append(f"- fixed threshold (balanced-accuracy optimum on `clean`): **{threshold:.4f}**")
    lines.append(f"- focus views: {', '.join(focus_views)}")
    lines.append("")

    lines.append("## Clean-vs-worst-view trade-off")
    lines.append("")
    lines.append("FPR/FNR at the one fixed threshold above -- the deployed operating point, not a "
                  "re-tuned one per view.")
    lines.append("")
    lines.append("| view | AUC | FPR (real flagged AIGC) | FNR (AIGC missed) |")
    lines.append("|---|---:|---:|---:|")
    for r in tradeoffs:
        lines.append(f"| {r['view']} | {r['auc']:.4f} | {r['fpr']:.4f} | {r['fnr']:.4f} |")
    lines.append("")
    if len(tradeoffs) >= 2:
        clean_r, worst_r = tradeoffs[0], tradeoffs[-1]
        d_fpr, d_fnr = worst_r["fpr"] - clean_r["fpr"], worst_r["fnr"] - clean_r["fnr"]
        skew = "false negatives (missed AIGC)" if d_fnr > d_fpr else "false positives (real flagged as AIGC)"
        lines.append(f"Going from `{clean_r['view']}` to `{worst_r['view']}`, FPR moves "
                      f"{d_fpr:+.4f} and FNR moves {d_fnr:+.4f} -- under degradation, error mass skews "
                      f"toward **{skew}**.")
        lines.append("")

    if gen_table:
        lines.append("## Per-generator collapse (worst first, pooled over cached degraded views)")
        lines.append("")
        lines.append("| generator | family | seen? | n | clean AUC | degraded AUC | drop |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for r in gen_table:
            lines.append(f"| {r['generator']} | {r['family']} | {r['seen']} | {r['n']} | "
                          f"{r['clean_auc']:.4f} | {r['degraded_auc']:.4f} | {r['drop']:+.4f} |")
        lines.append("")
        worst = gen_table[0]
        lines.append(f"Largest collapse: **{worst['generator']}** ({worst['family']}, {worst['seen']}), "
                      f"clean {worst['clean_auc']:.4f} -> degraded {worst['degraded_auc']:.4f} "
                      f"({worst['drop']:+.4f}).")
        lines.append("")

    for kind, label in (("false_positive", "False positives (real flagged as AIGC)"),
                         ("false_negative", "False negatives (AIGC missed)")):
        lines.append(f"## {label} -- top {top_k} most confident, per focus view")
        lines.append("")
        for name in focus_views:
            rows = [e for e in examples if e["view"] == name and e["kind"] == kind]
            if not rows:
                continue
            lines.append(f"### view: `{name}`")
            lines.append("")
            lines.append("| rank | pred | generator | image_path |")
            lines.append("|---:|---:|---|---|")
            for r in rows:
                lines.append(f"| {r['rank']} | {r['pred']:.4f} | {r['generator'] or '-'} | `{r['image_path']}` |")
            lines.append("")

    lines.append("## Trade-offs (for the write-up)")
    lines.append("")
    lines.append("- The threshold above is the one fixed operating point behind every number in this "
                  "report (and in `eval-grid`'s robustness table) -- a production deployment has exactly "
                  "one threshold, so results here are never re-tuned per view or per generator.")
    lines.append("- Raising the threshold trades FNR for FPR (fewer real images flagged, more AIGC missed) "
                  "and vice versa; which direction is safer depends on the deployment context (e.g. content "
                  "moderation prefers low FNR, a creator-facing warning label prefers low FPR).")
    lines.append("- The examples above are the model's most CONFIDENT mistakes, not its most representative "
                  "ones -- they're chosen to show the failure mode clearly, not its typical severity.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
