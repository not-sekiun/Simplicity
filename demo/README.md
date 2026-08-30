# Live demo: Chrome extension

**Out of scope of the competition deliverables** — a bonus demo for the
video, not part of the graded solution (`predict.py` / `main.py` under
the repo root are). Flags likely AI-generated images in real time as you
scroll a page (Reddit, Instagram, anywhere with `<img>` tags), with a
yellow→red outline and an "AI NN%" tag scaled to confidence.

```
demo/
  server.py          Local inference server (FastAPI) — the only thing that
                      knows about backbones/checkpoints
  extension/          Unpacked Chrome (MV3) extension — knows nothing about
                      the model, only "POST a URL, get back {pred}"
```

## Why it's built this way

The extension is written **against the HTTP abstraction only**: it sends
`{url}` to a local server and gets back `{pred}` — a probability in
`[0, 1]`. It never sees a backbone name, a parameter count, or a
checkpoint path. `demo/server.py` resolves all of that from whatever
checkpoint it's pointed at, the same way `predict.py` does (see that
module's docstring for the preprocessing-parity details this preserves).

**Consequence: swapping the backbone or retraining the head needs zero
extension changes** — just restart the server with a different `--head`
(e.g. once the full-pool retrain mentioned in the main README finishes).

## Setup

```bash
# 1. Install the demo-only deps (kept out of the base project on purpose —
#    see pyproject.toml's [project.optional-dependencies])
uv sync --extra demo

# 2. Start the inference server (loads the backbone once, then serves)
uv run python demo/server.py
# ... or point it at a specific checkpoint:
uv run python demo/server.py --head models/pe-core-l__linear__photoreal.pt
```

Wait for `Uvicorn running on http://127.0.0.1:8765` — the backbone download/
load takes a bit on first run. Verify it's up:

```bash
curl http://127.0.0.1:8765/health
# {"ready":true,"backbone":"pe-core-l","head":"pe-core-l__linear__photoreal.pt"}
```

### Load the extension

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top-right toggle).
3. **Load unpacked** → select `demo/extension/`.
4. Click the extension icon (pin it if Chrome hides it) — the popup shows
   server status, the loaded backbone/head, and a threshold slider.
5. Go scroll a page with lots of images (Reddit, Instagram, an image search).
   Images that score above the threshold get a colored outline and a
   confidence tag as they scroll into view.

## How it works

- `content.js` watches the page with an `IntersectionObserver` +
  `MutationObserver` — new images (infinite scroll, lazy-loaded feeds) get
  picked up automatically. Images smaller than ~150px (icons, avatars,
  emoji) are skipped.
- Visible images are batched (up to 6 at a time, 150ms debounce) and sent
  via `chrome.runtime.sendMessage` to `background.js`.
- `background.js` (the MV3 service worker) is the only piece that actually
  calls the local server. Routing through the background worker — rather
  than fetching directly from the content script — matters because some
  sites' Content-Security-Policy blocks a content script's own network
  requests to `http://127.0.0.1`; a background service worker isn't subject
  to the page's CSP.
- Results come back as `{url, pred}` / `{url, error}` and are applied per
  matching `<img>` element: `outline` (not `border`, so page layout never
  shifts) colored yellow→red by confidence, plus a small floating badge
  positioned via `getBoundingClientRect()` and kept in sync on scroll/resize.
  The badge is appended to a single container element attached directly to
  `<html>` — a sibling of the page's own app root, never a descendant of
  it — so a React-based site (Reddit, Instagram both are) never has to
  reconcile anything this extension adds.
- Results are cached by image URL for the life of the page load, so a
  repeated avatar/thumbnail is only scored once.

## Configuring

Click the extension icon:

- **flag threshold** — `pred >= threshold` gets flagged (default 0.5).
  Saving reloads the active tab so the new value takes effect.
- **server URL** — change this if you started `server.py` on a different
  `--host`/`--port`, or add another host in `manifest.json`'s
  `host_permissions` if you're not using `127.0.0.1`.

To score against a different manifest match set (default is `<all_urls>`),
edit `extension/manifest.json`'s `content_scripts[0].matches` — e.g.
`["*://*.reddit.com/*", "*://*.instagram.com/*"]` to scope it down.

## Known limitations (fine for a demo, not for shipping)

- `<img>` elements only — CSS `background-image` content isn't scanned.
- No auth/session cookies are sent when the server fetches an image URL, so
  images behind a login wall (private accounts, some CDNs with hotlink
  protection) will come back as a fetch error rather than a score. The
  extension silently skips those rather than showing a false "not AI".
- The server processes one batch at a time — fine for a single browsing
  tab during a demo, not built for concurrent multi-tab load.
- No AVIF support (Pillow doesn't decode it out of the box); WebP and the
  usual formats work.
