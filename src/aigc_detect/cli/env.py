from __future__ import annotations

from aigc_detect.config import (
    DEMO_VAL_DIR,
    DEMO_VAL_MANIFEST,
    HELDOUT_DIR,
    HELDOUT_MANIFEST,
    RAW_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)


def cmd_check_env(_args):
    import torch

    print(f"torch:          {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device:         {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"vram:           {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no CUDA device visible to PyTorch. Training will run on CPU.")

    print()
    index_files = sorted(RAW_DIR.glob("*_index.csv"))
    if index_files:
        for f in index_files:
            print(f"raw index:      {f.name}")
    else:
        print("raw index:      none yet -- run `main.py download cifake` / `download sid-set`")

    print(f"train manifest: {'OK - ' + str(TRAIN_MANIFEST) if TRAIN_MANIFEST.exists() else 'missing -- run `main.py split`'}")
    print(f"val manifest:   {'OK - ' + str(VAL_MANIFEST) if VAL_MANIFEST.exists() else 'missing -- run `main.py split`'}")

    print()
    heldout_index_files = sorted(HELDOUT_DIR.glob("*_index.csv")) if HELDOUT_DIR.exists() else []
    if heldout_index_files:
        for f in heldout_index_files:
            print(f"heldout index:  {f.name}")
    else:
        print("heldout index:  none yet -- run `main.py download tiny-genimage`")
    print(f"heldout manifest (cross-generator test, never trained on): "
          f"{'OK - ' + str(HELDOUT_MANIFEST) if HELDOUT_MANIFEST.exists() else 'missing -- run `main.py build-heldout`'}")

    print()
    demo_index_files = sorted(DEMO_VAL_DIR.glob("*_index.csv")) if DEMO_VAL_DIR.exists() else []
    if demo_index_files:
        for f in demo_index_files:
            print(f"demo-val index: {f.name}")
    else:
        print("demo-val index: none yet -- run `main.py download-demo coco-val2017`")
    print(f"demo-val manifest (self-reported ONLY, never trained on): "
          f"{'OK - ' + str(DEMO_VAL_MANIFEST) if DEMO_VAL_MANIFEST.exists() else 'missing -- run `main.py build-demo-val`'}")


def cmd_list_backbones(_args):
    from aigc_detect.registry.backbones import BACKBONE_REGISTRY, list_backbones

    for key in list_backbones():
        entry = BACKBONE_REGISTRY[key]
        print(
            f"{key:12s} checkpoint={entry['checkpoint']:55s} loader={entry['loader']:12s} "
            f"pooled_dim={entry['pooled_dim']:5d} native_res={entry['native_res']}"
        )


def register_check_env(sub):
    sub.add_parser("check-env", help="Verify PyTorch/CUDA setup and dataset status.").set_defaults(func=cmd_check_env)


def register_list_backbones(sub):
    sub.add_parser(
        "list-backbones", help="List registered frozen-backbone keys (src/aigc_detect/registry/backbones.py)."
    ).set_defaults(func=cmd_list_backbones)
