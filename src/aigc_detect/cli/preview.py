from __future__ import annotations

import sys

from aigc_detect.config import TRAIN_MANIFEST


def cmd_preview_augment(args):
    import torchvision.transforms.v2.functional as F
    from PIL import Image

    from aigc_detect.data.dataset import ManifestImageDataset
    from aigc_detect.data.transforms import build_train_transform

    if not TRAIN_MANIFEST.exists():
        print(f"No train manifest at {TRAIN_MANIFEST}. Run `main.py download ...` then `main.py split` first.")
        sys.exit(1)

    ds = ManifestImageDataset(TRAIN_MANIFEST, transform=build_train_transform())
    n = min(args.n, len(ds))
    tiles = []
    for i in range(n):
        tensor, _label = ds[i]
        # Undo normalization for viewing.
        img = F.to_pil_image((tensor * 0.229 + 0.485).clamp(0, 1))
        tiles.append(img)

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    w, h = tiles[0].size
    grid = Image.new("RGB", (w * cols, h * rows), "white")
    for idx, tile in enumerate(tiles):
        grid.paste(tile, ((idx % cols) * w, (idx // cols) * h))
    grid.save(args.out)
    print(f"Saved augmentation preview grid ({n} samples) -> {args.out}")


def register_preview_augment(sub):
    p_preview = sub.add_parser("preview-augment", help="Save a grid image sanity-checking the aug pipeline.")
    p_preview.add_argument("--n", type=int, default=8)
    p_preview.add_argument("--out", default="augment_preview.png")
    p_preview.set_defaults(func=cmd_preview_augment)
