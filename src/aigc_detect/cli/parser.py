"""Entry script for the AIGC-detection project (TikTok TechJam 2026, Track 5).

Subcommands:
    check-env                        Verify PyTorch/CUDA setup and report dataset status.
    download cifake                  Download CIFAKE in full (Kaggle).
    download sid-set [--limit-per-class N]
                                      Stream a capped subset of SID_Set (HuggingFace).
    download tiny-genimage [--limit-per-split N] [--force]
                                      Download Tiny-GenImage (HuggingFace): HF "train" ->
                                      data/raw/ (training pool), HF "validation" ->
                                      data/heldout/ (cross-generator test set, never
                                      trained on). Re-encodes every image as JPEG q95.
    split [--val-fraction F] [--seed S] [--exclude-source SOURCE ...]
          [--max-per-source N] [--holdout-generators G [G ...]]
                                      Build stratified data/processed/{train,val}.csv from
                                      data/raw/*_index.csv only (never data/heldout/ or
                                      data/demo_val/).
    preview-augment [--n N] [--out PATH]
                                      Save a grid image sanity-checking the augmentation
                                      pipeline (requires a train split to exist).
    download-demo coco-val2017       Download the self-reported demo-val "real" half.
    download-demo wildfake-dalle-advanced
                                      Index the demo-val "AIGC" half (manual fetch required
                                      first - see scripts/download_demo_val.py docstring).
    build-demo-val                   Merge demo-val indexes into data/demo_val/demo_val.csv.
                                      NEVER used for training - see 5.4 in the brief.
    download-ood [--per-generator N] [--max-scan N] [--min-scan N] [--seed S]
                 [--force]
                                      Stream a capped, generator-balanced slice of
                                      TheKernel01/AIGC-Detection-Benchmark into data/ood/ --
                                      a deliberately HARD out-of-distribution tier
                                      (10 of 18 generators unseen in training). Evaluation
                                      only, never trained on.
    build-ood                        Merge data/ood/*_index.csv into data/ood/ood.csv.
                                      Evaluation only, never trained on.
    build-heldout                    Merge data/heldout/*_index.csv into
                                      data/heldout/heldout.csv. Cross-generator test set,
                                      NEVER used for training.
    audit-data [--sample N] [--transform]
                                      Shortcut audit of data/raw/*_index.csv: per-source
                                      stats + a blind-probe canary for label shortcuts
                                      (e.g. aspect ratio). --transform runs the probe on
                                      build_eval_transform() tensors instead of raw images.
    list-backbones                   List registered frozen-backbone keys (see
                                      src/aigc_detect/registry/backbones.py).
    embed --backbone KEY --manifest {train,val,demo-val} [--force] [--limit N]
                                      Precompute + cache pooled embeddings for a manifest
                                      under data/embeddings/. Implements the "Simplicity
                                      Prevails" (arXiv:2602.01738) preprocessing recipe.
                                      demo-val is EVALUATION ONLY (see 5.4).
    embed-views --backbone KEY --manifest {train,val,heldout,demo-val}
                [--views V ...] [--force] [--limit N] [--sample-rows N]
                [--train-chains]
                                      Same, but for all 18 robustness views: clean,
                                      the 14 single-transform rows of the 5.2 table,
                                      and 3 chained rows. One decode per image feeds
                                      every view. Caches to
                                      <backbone>__<stem>__<view>.npz. This is the
                                      instrument for the AUC_robust half of the score.
                                      --sample-rows draws a label-balanced,
                                      source-proportional subsample (seeded, so every
                                      backbone faces the identical subset) and tags
                                      the cache stem with it -- the intended path for
                                      racing backbones cheaply. --train-chains also
                                      computes 4 extra trainchain_* views (train
                                      manifest only) -- augmentation material for
                                      train-head-views, never scored by eval-grid.
    train-head-views --backbone KEY [--train-sample-rows N] [--val-sample-rows N]
                      [--train-views V ...] [--with-chains] [--clean-only]
                      [--head linear|mlp] [--epochs E] [--lr LR]
                      [--batch-size B] [--weight-decay WD] [--out PATH]
                                      Train a head on cached CLEAN + DEGRADED
                                      embeddings (run `embed-views` for train and val
                                      first). The augmentation ablation: default trains
                                      on one severity per family and holds the rest,
                                      including all 3 scored chains, out for
                                      evaluation only. --with-chains additionally
                                      trains on the 4 trainchain_* composition views
                                      (--sample-rows must match what embed-views used).
    eval-grid --backbone KEY --manifest {train,val,heldout,demo-val} [--head PATH]
              [--sample-rows N] [--limit N]
                                      Score a trained head over every cached view:
                                      per-view AUC/balanced accuracy at one fixed
                                      threshold, AUC_robust three ways, the
                                      robustness gap, and single-vs-chained means.
                                      Deliverable 5.5.4. No GPU work.
    error-analysis --backbone KEY --manifest {train,val,heldout,demo-val,ood}
                    [--head PATH] [--sample-rows N] [--limit N] [--top-k N]
                                      Concrete false positives/negatives (most
                                      confident first) + a per-generator collapse
                                      ranking, at eval-grid's fixed threshold.
                                      Writes CSV + a markdown report + copied
                                      example images under reports/error_analysis/.
                                      Deliverable 5.5.5. No GPU work; needs
                                      embed-views cached for the same manifest first.
    train-head --backbone KEY [--head linear|mlp] [--epochs E] [--lr LR]
               [--batch-size B]
                                      Train a classifier head on cached embeddings for
                                      KEY (run `embed` for both train and val first).
    predict --input_dir DIR --output preds.json [--head PATH]
                                      Run inference on a directory of images, emit
                                      JSON [{"image_path": str, "pred": float}, ...]
                                      where pred = P(AIGC). Deliverable 5.5.2. Also
                                      runnable standalone as `uv run python predict.py`.

Examples:
    uv run main.py check-env
    uv run main.py download cifake
    uv run main.py download sid-set --limit-per-class 4000
    uv run main.py download tiny-genimage --limit-per-split 40
    uv run main.py split
    uv run main.py preview-augment --n 8
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
    manifests,
    predict,
    preview,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    env.register_check_env(sub)
    datasets.register_download(sub)
    manifests.register_split(sub)
    preview.register_preview_augment(sub)
    datasets.register_download_demo(sub)
    manifests.register_build_demo_val(sub)
    datasets.register_download_ood(sub)
    manifests.register_build_ood(sub)
    manifests.register_build_heldout(sub)
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

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
