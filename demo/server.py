#!/usr/bin/env python
"""Local inference server for the Chrome-extension live demo.

OUT OF SCOPE of the competition deliverables -- `predict.py` (5.5.2) is the
graded inference entry point, and this file does not change it. This exists
only to back the demo: a Chrome extension that scores images on a page in
real time as you scroll (e.g. Reddit, Instagram) and overlays a
confidence-colored flag on the ones it thinks are AI-generated.

Reuses the same backbone-loading and preprocessing-parity conventions as
`aigc_detect.predict` (per-backbone native_res + norm stats, checkpoint's
OWN scaler, sigmoid on a linear/MLP head) so a prediction from this server
means the same thing as one from `predict.py`. See that module's docstring
for why each step matters -- it is not repeated here.

THE ABSTRACTION THE EXTENSION IS WRITTEN AGAINST: POST a URL, get back
{"pred": float}. The extension has zero knowledge of backbones, parameter
counts, or checkpoint paths -- everything about "which model" is resolved
here, from the checkpoint, exactly like predict.py does. Swap --head to a
different backbone entirely (once retrained) and the extension needs no
changes; the /health endpoint's backbone/head fields are for the popup UI
to *display*, not for the extension to branch on.

    uv sync --extra demo
    uv run python demo/server.py --head models/pe-core-l__linear__allsev_e1.pt

Endpoints:
    GET  /health                        {"ready": bool, "backbone": str, "head": str}
    POST /score       {"url": str}      {"url": str, "pred": float} | {"url": str, "error": str}
    POST /score_batch {"urls": [str]}   {"results": [ ...one of the above per url... ]}
    POST /score_frame {"frame": str}    {"pred": float} | {"error": str}

/score_frame exists for the extension's video path: a sampled <video> frame
(captured client-side via canvas, since browsers can't hand out a fetchable
URL for "the current frame of this playing video") has no URL for this
server to fetch the way /score and /score_batch do -- it arrives as a
base64-encoded JPEG data URL instead and is decoded straight to bytes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision.transforms import v2

from aigc_detect.config import ROOT_DIR
from aigc_detect.data.transforms import build_backbone_transform
from aigc_detect.log import configure, get_logger
from aigc_detect.registry.backbones import load_backbone
from aigc_detect.registry.heads import build_head

DEFAULT_HEAD = ROOT_DIR / "models" / "pe-core-l__linear__allsev_e1.pt"
FETCH_TIMEOUT = 6.0
FETCH_HEADERS = {
    # Reddit/Instagram reject or throttle requests with no browser-like UA.
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
_FETCH_POOL = ThreadPoolExecutor(max_workers=8)

configure()
logger = get_logger(__name__)

class Model:
    """Loaded once at startup. Holds exactly the state predict.py's
    run_inference derives from a checkpoint (see its module docstring,
    points 1-4) -- backbone + head + the backbone's OWN transform/norm
    stats + the checkpoint's OWN scaler. One source of truth for what a
    checkpoint means; nothing here is guessed or hardcoded per-backbone.
    """

    def __init__(self, head_path: Path):
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        self.backbone_key = ckpt["backbone"]
        self.head_kind = ckpt["head_kind"]
        in_dim = ckpt["in_dim"]
        self.scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_std = np.asarray(ckpt["scaler_std"], dtype=np.float32)

        self.module, pooled_dim, native_res = load_backbone(self.backbone_key)
        if pooled_dim != in_dim:
            raise SystemExit(
                f"[server] backbone '{self.backbone_key}' pooled_dim={pooled_dim} does not match "
                f"checkpoint in_dim={in_dim} -- checkpoint/backbone mismatch."
            )
        self.head = build_head(self.head_kind, in_dim)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self.device = next(self.module.parameters()).device
        self.head.to(self.device)

        # Identical to embed.py/predict.py: aspect-preserving resize + center
        # crop at the backbone's OWN native_res, then the backbone's OWN norm
        # stats. Never config.IMAGE_SIZE/NORM_MEAN -- see predict.py docstring.
        self.transform = v2.Compose(
            [
                *build_backbone_transform(native_res),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.module.norm_mean, std=self.module.norm_std),
            ]
        )
        self.head_path = head_path
        print(
            f"[server] loaded backbone={self.backbone_key} head={head_path.name} "
            f"native_res={native_res} device={self.device}"
        )

    @torch.no_grad()
    def score(self, images: list[Image.Image]) -> list[float]:
        if not images:
            return []
        batch = torch.stack([self.transform(img) for img in images]).to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_amp):
            feats = self.module(batch)
        feats = feats.float()
        # Checkpoint's own scaler, not batch statistics -- see predict.py
        # docstring point 4. Recomputing per-request would re-center every
        # request around whatever handful of images happen to be in it.
        mean_t = torch.from_numpy(self.scaler_mean).to(feats.device)
        std_t = torch.from_numpy(self.scaler_std).to(feats.device)
        x = (feats - mean_t) / std_t
        probs = torch.sigmoid(self.head(x).squeeze(-1)).cpu().numpy()
        return [float(p) for p in probs]


def _safe_fetch(url: str) -> Image.Image | Exception:
    try:
        resp = requests.get(url, headers=FETCH_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
        return exc


model: Model | None = None
app = FastAPI(title="aigc-detect live demo server")
app.add_middleware(
    CORSMiddleware,
    # Reached only from the extension's background service worker, which
    # runs as a chrome-extension:// origin, not from arbitrary web pages.
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    url: str


class ScoreBatchRequest(BaseModel):
    urls: list[str]


class ScoreFrameRequest(BaseModel):
    frame: str  # "data:image/jpeg;base64,...." or bare base64


def _decode_frame(data_url: str) -> Image.Image | Exception:
    try:
        b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (ValueError, OSError, UnidentifiedImageError, binascii.Error) as exc:
        return exc


@app.get("/health")
def health():
    if model is None:
        return {"ready": False}
    return {"ready": True, "backbone": model.backbone_key, "head": model.head_path.name}


@app.post("/score")
def score(req: ScoreRequest):
    if model is None:
        return {"url": req.url, "error": "model not loaded"}
    img = _safe_fetch(req.url)
    if isinstance(img, Exception):
        return {"url": req.url, "error": f"{type(img).__name__}: {img}"}
    pred = model.score([img])[0]
    logger.info(f"{pred=}")
    return {"url": req.url, "pred": pred}


@app.post("/score_batch")
def score_batch(req: ScoreBatchRequest):
    if model is None:
        return {"results": [{"url": u, "error": "model not loaded"} for u in req.urls]}

    fetched = list(_FETCH_POOL.map(_safe_fetch, req.urls))
    images, ok_urls, results = [], [], []
    for u, item in zip(req.urls, fetched, strict=True):
        if isinstance(item, Exception):
            results.append({"url": u, "error": f"{type(item).__name__}: {item}"})
        else:
            images.append(item)
            ok_urls.append(u)

    preds = model.score(images)
    logger.info(f"{preds=}")
    for u, p in zip(ok_urls, preds, strict=True):
        results.append({"url": u, "pred": p})
    return {"results": results}


@app.post("/score_frame")
def score_frame(req: ScoreFrameRequest):
    if model is None:
        return {"error": "model not loaded"}
    img = _decode_frame(req.frame)
    if isinstance(img, Exception):
        return {"error": f"{type(img).__name__}: {img}"}
    pred = model.score([img])[0]
    logger.info(f"{pred=}")
    return {"pred": pred}


def main():
    global model
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default=str(DEFAULT_HEAD), help=f"Head checkpoint (default: {DEFAULT_HEAD.name}).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    head_path = Path(args.head)
    if not head_path.exists():
        raise SystemExit(f"[server] head checkpoint not found: {head_path}")
    model = Model(head_path)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
