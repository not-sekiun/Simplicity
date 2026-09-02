# Live demo: browser extension

**Out of scope of the competition deliverables** — a bonus demo for the
video, not part of the graded solution (`predict.py` / `main.py` under
the repo root are). Flags likely AI-generated images in real time as you
scroll a page (Reddit, Instagram, anywhere with `<img>` tags), with a
yellow→red outline and an "AI NN%" tag scaled to confidence. Also samples
`<video>` elements every few seconds while they're playing on-screen —
verified working against TikTok, Instagram Reels, and YouTube Shorts.

```
demo/
  server.py          Thin shim -> apps/server/app.py (tier 8b), kept so the
                      documented `uv run python demo/server.py` invocation
                      below keeps working — same relationship main.py has to
                      aigc_detect.cli and predict.py has to
                      inference.predict.run_inference.
  extension/          Browser extension source — knows nothing about the
                      model, only "POST a URL, get back {pred}"
    src/               content.js entry + its modules (detector-client,
                         img-scanner, video-sampler, overlay, heuristics),
                         background.js, popup.js — one source tree, built
                         per target below
    manifest.base.json  Manifest fields shared by every target
    build.js            esbuild step -> dist/chrome/, dist/firefox/
    test/                node:test unit tests (`npm test`)
    dist/                Build output, gitignored — load THIS, not `src/`
apps/server/          The real inference server (FastAPI) — the only thing
                      that knows about backbones/checkpoints. Holds an
                      `aigc_detect.inference.detector.Detector`, doesn't
                      reimplement its preprocessing. Registered as the
                      `aigc-serve` console script (needs `uv sync --extra
                      demo`; not part of the base dependency set — see
                      pyproject.toml).
```

## Why it's built this way

The extension is written **against the HTTP abstraction only**: it sends
`{url}` to a local server and gets back `{pred}` — a probability in
`[0, 1]`. It never sees a backbone name, a parameter count, or a
checkpoint path. `apps/server/app.py` (what `demo/server.py` shims onto)
resolves all of that from whatever checkpoint it's pointed at, by loading a
`Bundle` (`aigc_detect.inference.bundle`) and holding a `Detector`
(`aigc_detect.inference.detector`) — the same bundle/preprocessing
`predict.py` uses, not a second copy of it. `tests/test_parity.py` asserts
the two agree on real images, instead of a docstring asking you to trust it.

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
uv run python demo/server.py --head models/pe-core-l__linear__trainext.pt
# ... equivalently, the console script (same apps/server/app.py:main):
uv run aigc-serve --head models/pe-core-l__linear__trainext.pt
```

Wait for `Uvicorn running on http://127.0.0.1:8765` — the backbone download/
load takes a bit on first run. Verify it's up:

```bash
curl http://127.0.0.1:8765/health
# {"ready":true,"backbone":"pe-core-l","head":"pe-core-l__linear__trainext.pt","threshold":0.98}
```

`threshold` is the field the extension itself reads (see "Configuring"
below) — the loaded checkpoint's own calibrated decision threshold, not a
client-side guess. It's optional in the response; the extension falls back
to a conservative constant if a server omits it.

### Build the extension

```bash
cd demo/extension
npm install   # esbuild + webextension-polyfill, see package.json
npm run build # -> dist/chrome/, dist/firefox/
npm test      # heuristics unit tests, see test/
```

One source tree (`src/`) is bundled per target because Chrome MV3 and
Firefox disagree on how a `background` is declared — see `build.js`'s
module docstring. `dist/` is gitignored; rerun `npm run build` after
pulling changes to `demo/extension/src/`.

### Load the extension

**Chrome / Edge / other Chromium browsers (MV3):**

1. Open `chrome://extensions`.
2. Enable **Developer mode** (top-right toggle).
3. **Load unpacked** → select `demo/extension/dist/chrome/` (not `src/` —
   that's unbundled ES modules a content script can't load directly, and
   there's no `manifest.json` there at all).
4. Click the extension icon (pin it if Chrome hides it) — the popup shows
   server status, the loaded backbone/head, and a threshold slider.
5. Go scroll a page with lots of images (Reddit, Instagram, an image search).
   Images that score above the threshold get a colored outline and a
   confidence tag as they scroll into view.

**Firefox:**

1. Open `about:debugging#/runtime/this-firefox`.
2. **Load Temporary Add-on…** → select any file inside
   `demo/extension/dist/firefox/` (e.g. `manifest.json`).
3. Same popup/behavior as Chrome from step 4 above. A temporary add-on is
   unloaded when Firefox closes; for a persistent install it needs signing
   via `web-ext sign` against `browser_specific_settings.gecko.id`
   (`manifest.base.json` / `build.js`) — not set up here, this demo only
   needed the temporary-load path.

**Safari:**

Safari doesn't load a `dist/` folder directly — it needs Apple's converter
to produce an Xcode project first, which is a macOS-only step this repo's
build can't automate (no macOS toolchain in CI or on the dev machine this
was built on). Manually, on a Mac with Xcode installed:

```bash
xcrun safari-web-extension-converter demo/extension/dist/chrome/
```

This scaffolds an Xcode project wrapping the Chrome build (Safari's
extension format is close enough to MV3 that the Chrome output, not a
separate `dist/safari/`, is the right input). Open the generated project in
Xcode, build, and enable the extension in Safari's
Settings → Extensions — then enable **Allow Unsigned Extensions** in
Safari's Develop menu for a local, unsigned build. Untested end-to-end
(no Mac available while building this) — documented rather than automated
per the constraint above.

## How it works

`content.js` (in `src/`) is a thin entry point that wires together five
modules — this used to be one 523-line file; splitting it is what made the
heuristics below unit-testable at all (`test/heuristics.test.js`):

- `heuristics.js` — pure `(img, rect) -> boolean` decorative/too-small
  checks (aria-hidden, CSS blur, extreme aspect ratio, size floor). No DOM
  APIs beyond what's passed in, which is the whole reason it can be tested
  with `node --test` and no browser.
- `img-scanner.js` — the `IntersectionObserver` + batching/caching-by-URL
  logic for `<img>` elements.
- `video-sampler.js` — the timer-driven, cache-by-*element* logic for
  `<video>` elements (see its module docstring for why that's a different
  tracking problem than `<img>`).
- `overlay.js` — the floating badge + outline color, shared by both.
- `detector-client.js` — the only module that knows the
  `{type, urls}`/`{url, pred}` messaging contract with `background.js`, and
  resolves the flag threshold (see "Configuring" below).

- `content.js` runs a `MutationObserver` on the page (infinite scroll,
  lazy-loaded feeds all get picked up automatically) and hands new elements
  to `img-scanner.js`/`video-sampler.js`. `img-scanner.js`'s own
  `IntersectionObserver` is what actually decides an image is worth
  scoring — too small (~150px floor) or decorative (`heuristics.js`) and it
  never gets sent at all.
- Visible images are batched (up to 6 at a time, 150ms debounce) and sent
  via `browser.runtime.sendMessage` (the `webextension-polyfill` promise-
  based API, wrapping `chrome.*` on Chrome and native on Firefox/Safari) to
  `background.js`.
- `background.js` (the MV3 service worker, or Firefox's equivalent
  background script — see `build.js`) is the only piece that actually
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
- `<video>` elements get a parallel path, since a video has no equivalent of
  a stable, cacheable "image URL": short-form feeds reuse a small pool of
  `<video>` elements and load a new clip into the same element as you
  scroll, and the element's `src`/`currentSrc` is typically a per-clip
  `blob:` URL from MediaSource rather than a fetchable address. Every
  ~3 seconds, each playing/on-screen `<video>` has a frame grabbed via
  `canvas.drawImage()` + `toDataURL()`, sent to the server as a base64 JPEG
  (`POST /score_frame`, no URL fetch involved), and the result is smoothed
  with a light exponential moving average before updating that video's
  outline/badge. State is keyed by the element, not a src, and is reset
  immediately (via the `loadstart` event, or a direct `src` swap) whenever
  a new clip loads into a reused element, so a score never rides over onto
  the wrong clip.

## Configuring

Click the extension icon:

- **flag threshold** — `pred >= threshold` gets flagged. There is no
  hardcoded default any more: on every page load, `content.js` asks the
  server's `GET /health` for the loaded checkpoint's own calibrated
  threshold and uses that, unless you've explicitly saved a value here
  (an explicit save always wins). If the server can't be reached, or an
  older server's `/health` doesn't report `threshold` at all, it falls back
  to a conservative constant (`0.98`, matching the shipped checkpoint's
  actual calibration — see `src/detector-client.js`'s `FALLBACK_THRESHOLD`)
  rather than a permissive `0.5`: at 0.5 the shipped head runs at 18.75%
  FPR on the same eval tier where 0.980 measures 2.15% — a wrong-but-
  permissive default would silently over-flag real images. The popup shows
  whichever of these three would actually be used, live, next to the
  slider. Saving reloads the active tab so the new value takes effect.
- **server URL** — change this if you started `server.py` on a different
  `--host`/`--port`, or add another host in `manifest.base.json`'s
  `host_permissions` if you're not using `127.0.0.1` (then `npm run build`
  again).
- **debug mode** — badges *every* scored image/video, not just ones over
  threshold: green (confidently real) → yellow (at the threshold) → red
  (confidently AI), one continuous scale. Useful for sanity-checking the
  model on things it isn't flagging, not just the ones it is.

To score against a different manifest match set (default is `<all_urls>`),
edit `demo/extension/manifest.base.json`'s `content_scripts[0].matches` —
e.g. `["*://*.reddit.com/*", "*://*.instagram.com/*"]` to scope it down,
then `npm run build` again (this edits the shared base, so it applies to
every target).

## Known limitations (fine for a demo, not for shipping)

- CSS `background-image` content isn't scanned (neither `<img>` nor
  `<video>` covers it).
- Video sampling relies on `canvas.drawImage()` from the live `<video>`
  element not tainting the canvas, which depends on the CDN's CORS posture
  and isn't guaranteed for every site — verified working on TikTok,
  Instagram Reels, and YouTube Shorts specifically. On a site where it
  doesn't, `captureFrame()` fails closed (no score, no badge, no console
  spam) rather than erroring.
- Video scoring is single-frame-every-~3s, not continuous — a fast cut
  partway through an interval can be missed until the next sample.
- No auth/session cookies are sent when the server fetches an image URL, so
  images behind a login wall (private accounts, some CDNs with hotlink
  protection) will come back as a fetch error rather than a score. The
  extension silently skips those rather than showing a false "not AI".
- The server processes one batch at a time — fine for a single browsing
  tab during a demo, not built for concurrent multi-tab load.
- No AVIF support (Pillow doesn't decode it out of the box); WebP and the
  usual formats work.
