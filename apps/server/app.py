#!/usr/bin/env python
"""Local inference server for the Chrome-extension live demo.

OUT OF SCOPE of the competition deliverables -- `predict.py` (5.5.2) is the
graded inference entry point, and this file does not change it. This exists
only to back the demo: a Chrome extension that scores images on a page in
real time as you scroll (e.g. Reddit, Instagram) and overlays a
confidence-colored flag on the ones it thinks are AI-generated.

TIER 8b: "LOAD BUNDLE, HOLD A DETECTOR." This module used to carry its own
`Model` class, which opened a raw checkpoint dict and re-implemented the
scaler arithmetic (`(x - mean) / std`) and the backbone/transform assembly
by hand -- a second copy of exactly what `inference.predict.run_inference`
already did, and a third copy of what `train.features.FeaturePipeline` owns
(see `inference.detector`'s module docstring for the whole history). That is
gone: this server now does `bundle = load_bundle(path)` then
`detector = FrozenProbeDetector.load(bundle)` and never touches a backbone
registry, a norm-stat tuple, or a scaler array itself. `tests/test_parity.py`
is what proves this path and `predict.py`'s agree, instead of a docstring
asking the reader to trust it.

THE ABSTRACTION THE EXTENSION IS WRITTEN AGAINST: POST a URL, get back
{"pred": float}. The extension has zero knowledge of backbones, parameter
counts, or checkpoint paths -- everything about "which model" is resolved
here, from the bundle, exactly like predict.py does. Swap --head to a
different backbone entirely (once retrained) and the extension needs no
changes; the /health endpoint's backbone/head/threshold fields are for the
popup UI to *display*, not for the extension to branch on -- see
`Detector.describe`.

    uv sync --extra demo
    uv run aigc-serve --head models/pe-core-l__linear__allsev_e1.pt
    # or, unchanged: uv run python demo/server.py --head ...
    # (demo/server.py is a thin shim onto this module -- see its docstring)

Endpoints:
    GET  /health                        {"ready": bool, "backbone": str, "head": str, "threshold": float, ...}
    POST /score       {"url": str}      {"url": str, "pred": float} | {"url": str, "error": str}
    POST /score_batch {"urls": [str]}   {"results": [ ...one of the above per url... ]}
    POST /score_frame {"frame": str}    {"pred": float} | {"error": str}

/score_frame exists for the extension's video path: a sampled <video> frame
(captured client-side via canvas, since browsers can't hand out a fetchable
URL for "the current frame of this playing video") has no URL for this
server to fetch the way /score and /score_batch do -- it arrives as a
base64-encoded JPEG data URL instead and is decoded straight to bytes.

THE WIRE CONTRACT IS FROZEN. `demo/extension/src/detector-client.js` and
`background.js` are built and shipped against exactly the shapes above,
`threshold` included since tier 8b -- adding a field is safe, renaming or
removing one is not (see `demo/README.md` "Configuring" and
`FALLBACK_THRESHOLD`'s docstring for what a missing/renamed `threshold`
degrades to).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from aigc_detect.config import ROOT_DIR
from aigc_detect.inference.bundle import load_bundle
from aigc_detect.inference.detector import Detector, FrozenProbeDetector
from aigc_detect.log import configure, get_logger

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


def _safe_fetch(url: str) -> Image.Image | Exception:
    try:
        resp = requests.get(url, headers=FETCH_HEADERS, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
        return exc


detector: Detector | None = None
head_path: Path | None = None
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
    if detector is None:
        return {"ready": False}
    assert head_path is not None
    # `describe()` supplies backbone/backbone_revision/head_kind/threshold/
    # threshold_source/bundle_version; `head` (the checkpoint FILENAME, as
    # opposed to `backbone`, the model architecture) is server-level state no
    # Detector needs to know about, so it's added here rather than folded
    # into the protocol. See detector-client.js: `threshold` is the one field
    # the extension actually branches on, everything else is display-only.
    return {"ready": True, "head": head_path.name, **detector.describe()}


@app.post("/score")
def score(req: ScoreRequest):
    if detector is None:
        return {"url": req.url, "error": "model not loaded"}
    img = _safe_fetch(req.url)
    if isinstance(img, Exception):
        return {"url": req.url, "error": f"{type(img).__name__}: {img}"}
    pred = detector.score([img])[0]
    logger.info(f"{pred=}")
    return {"url": req.url, "pred": pred}


@app.post("/score_batch")
def score_batch(req: ScoreBatchRequest):
    if detector is None:
        return {"results": [{"url": u, "error": "model not loaded"} for u in req.urls]}

    fetched = list(_FETCH_POOL.map(_safe_fetch, req.urls))
    images, ok_urls, results = [], [], []
    for u, item in zip(req.urls, fetched, strict=True):
        if isinstance(item, Exception):
            results.append({"url": u, "error": f"{type(item).__name__}: {item}"})
        else:
            images.append(item)
            ok_urls.append(u)

    preds = detector.score(images)
    logger.info(f"{preds=}")
    for u, p in zip(ok_urls, preds, strict=True):
        results.append({"url": u, "pred": p})
    return {"results": results}


@app.post("/score_frame")
def score_frame(req: ScoreFrameRequest):
    if detector is None:
        return {"error": "model not loaded"}
    img = _decode_frame(req.frame)
    if isinstance(img, Exception):
        return {"error": f"{type(img).__name__}: {img}"}
    pred = detector.score([img])[0]
    logger.info(f"{pred=}")
    return {"pred": pred}


def main():
    global detector, head_path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default=str(DEFAULT_HEAD), help=f"Head checkpoint (default: {DEFAULT_HEAD.name}).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    path = Path(args.head)
    if not path.exists():
        raise SystemExit(f"[server] head checkpoint not found: {path}")
    bundle = load_bundle(path)
    detector = FrozenProbeDetector.load(bundle)
    head_path = path
    print(
        f"[server] loaded backbone={bundle.backbone.key} head={path.name} "
        f"threshold={bundle.threshold:g} (source: {bundle.threshold_source})"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
