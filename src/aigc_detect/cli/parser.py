"""Entry script for the AIGC-detection project (TikTok TechJam 2026, Track 5).

Every command below is registered by one module under `aigc_detect.cli`, and
this docstring is the root `--help` text -- so a command that is not described
here is a command nobody can discover. Four commands that USED to be listed
(`split`, `build-ood`, `build-heldout`, `build-demo-val`) were retired in the
data-hierarchy tier: each ran one hand-written script to produce one CSV, and
all four are now `manifest resolve <name>` against a declared recipe.

DATA -- corpora, recipes, and pulls
    pull list | show <id> | run <id> [--force] [--limit N] [--no-audit]
         | verify <id>
                                      Fetch a source declared in
                                      registry/sources.yaml into
                                      data/corpora/<id>/. Resume is the DEFAULT,
                                      not a flag; --force discards a partial
                                      pull deliberately. Every pull ends with
                                      the blind-probe shortcut audit, whose
                                      verdict is written into the corpus's own
                                      corpus.yaml and gates training on it.
    corpus list | orphans [--list N]
                                      The corpus registry, and which images on
                                      disk no manifest references. The sweep
                                      REPORTS; it never deletes.
    manifest list | show <name> | resolve <name> | check
                                      Manifest recipes: a declarative selection
                                      over corpora, resolved into
                                      data/manifests/resolved/<name>.csv.
                                      `resolve all` rebuilds every one.
    download cifake | sid-set | tiny-genimage
    download-demo coco-val2017 | wildfake-dalle-advanced
    download-ood [--per-generator N] [--max-scan N] [--seed S] [--force]
                                      Named shorthands for the pulls above, kept
                                      because every document and every muscle
                                      memory uses this form.
    audit-data [--sample N] [--transform]
                                      Run the blind probe across the whole
                                      registry without pulling anything: a
                                      16x16 greyscale logistic regression that
                                      MUST fail to separate real from AIGC.
                                      Balanced accuracy >= 0.70 means a label
                                      shortcut survives. See docs/findings.md.
    preview-augment [--n N] [--out PATH]
                                      Save a grid image sanity-checking the
                                      augmentation pipeline.

EMBEDDINGS -- the content-addressed store and its projections
    cache status | migrate | verify | export | compact | merge
                                      The store. An embedding is identified by
                                      (image bytes, backbone id, view spec) and
                                      nothing else -- not the path, not the row
                                      order, not which manifest asked for it. A
                                      killed run resumes from its last committed
                                      batch; a moved repo costs no forward
                                      passes; two machines combine with `merge`.
    embed --backbone KEY --manifest NAME [--force] [--limit N]
                                      One pooled embedding per image.
    embed-views --backbone KEY --manifest NAME [--views V ...] [--force]
                [--limit N] [--sample-rows N] [--train-chains]
                                      All 18 robustness views -- clean, the 14
                                      single-transform rows of the brief's 5.2
                                      table, and 3 chained rows -- from a single
                                      decode. Writes to the store and projects
                                      data/embeddings/*.npz out of it, so a
                                      re-run costs only the gaps. This is the
                                      instrument for the AUC_robust half of the
                                      score. --train-chains adds 4 trainchain_*
                                      views (training material, never scored).

TRAINING AND EVALUATION
    experiment list | show <name> | run <name> [--log-dir DIR] [--out PATH]
                                      A declared run: experiments/<name>.yaml
                                      names the manifest, backbone, views,
                                      feature pipeline, head and schedule, and
                                      the runner writes data/runs/<run_id>/ with
                                      the resolved config, the eval grid, the
                                      derived threshold and a model bundle that
                                      carries its own operating point. This is
                                      the path that replaces a seven-flag
                                      command line nobody could reproduce.
    train-head --backbone KEY [--head linear|mlp] [--epochs E] [--lr LR]
               [--batch-size B]
    train-head-views --backbone KEY [--train-views V ...] [--with-chains]
                     [--clean-only] [--head linear|mlp] [--epochs E] [...]
                                      The pre-experiment training paths, kept
                                      because 25 archived heads and every number
                                      docs/findings.md cites came from them.
    eval-grid --backbone KEY --manifest NAME [--head PATH] [--sample-rows N]
              [--by-generator] [--out PATH]
                                      Per-view AUC and balanced accuracy at one
                                      fixed threshold, AUC_robust three ways,
                                      the robustness gap, and single-vs-chained
                                      means. Deliverable 5.5.4. No GPU work.
    error-analysis --backbone KEY --manifest NAME [--head PATH] [--top-k N]
                                      The most confident false positives and
                                      false negatives, plus a per-generator
                                      collapse ranking. Deliverable 5.5.5.
    list-backbones                    Registered frozen-backbone keys. Every one
                                      asserts <2e9 parameters; the vision tower
                                      ships, never the full checkpoint.

INFERENCE
    predict --input_dir DIR --output preds.json [--head PATH]
                                      JSON [{"image_path": str, "pred": float}]
                                      where pred = P(AIGC). Deliverable 5.5.2.
                                      The threshold comes from the model bundle,
                                      not from a constant in this repo. Also
                                      runnable as `uv run python predict.py`.

ENVIRONMENT
    check-env                         Verify the PyTorch/CUDA setup and report
                                      which corpora are actually on disk.

Examples:
    uv run main.py check-env
    uv run main.py pull run nano_banana
    uv run main.py manifest resolve train
    uv run main.py embed-views --backbone pe-core-l --manifest val
    uv run main.py experiment run allsev_e1
    uv run main.py eval-grid --backbone pe-core-l --manifest ood --sample-rows 4000
"""

from __future__ import annotations

import argparse

from aigc_detect.cli import (
    audit,
    cache,
    datasets,
    embed,
    env,
    evaluate,
    experiment,
    predict,
    preview,
    pull,
    recipes,
    train,
)
from aigc_detect.log import configure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    env.register_check_env(sub)
    datasets.register_download(sub)
    preview.register_preview_augment(sub)
    datasets.register_download_demo(sub)
    datasets.register_download_ood(sub)
    audit.register_audit_data(sub)
    env.register_list_backbones(sub)
    embed.register_embed(sub)
    embed.register_embed_views(sub)
    train.register_train_head_views(sub)
    evaluate.register_eval_grid(sub)
    evaluate.register_error_analysis(sub)
    train.register_train_head(sub)
    predict.register_predict(sub)
    cache.register_cache(sub)
    recipes.register_manifest(sub)
    recipes.register_corpus(sub)
    pull.register_pull(sub)
    experiment.register_experiment(sub)

    return parser


def main():
    configure()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
