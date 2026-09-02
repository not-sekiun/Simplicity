// video-sampler.js -- samples <video> elements on a timer instead of the
// <img> path's IntersectionObserver+cache-by-src (img-scanner.js).
//
// Images are cached by `src` because the same photo genuinely recurs (a
// repeated avatar, a reposted thumbnail). Video on a short-form feed has no
// equivalent: a <video>'s `src`/`currentSrc` is typically a per-clip
// `blob:` URL handed out by MediaSource, and the feed reuses a small pool
// of <video> elements, loading a brand new clip into the SAME element as
// you scroll past it. So state here is keyed by the element itself, not a
// src -- there's no "already scored" cache, only "currently sampling
// whatever clip is loaded into this element right now" -- and it's
// explicitly reset the instant that clip changes underneath us.

import { scoreFrame } from "./detector-client.js";
import { badgeColor, positionBadge, createBadge } from "./overlay.js";

const VIDEO_SAMPLE_INTERVAL_MS = 3000; // a couple of seconds is fine --
// this isn't meant to be frame-accurate, and it keeps request volume sane
// against a server that scores one batch at a time (see server.py).
const VIDEO_FRAME_MAX_SIDE = 480; // downscaled before sending -- the
// backbone re-resizes to its own native_res anyway (see server.py), so
// shipping a full 1080x1920 frame over sendMessage would be pure waste.
const VIDEO_EMA_ALPHA = 0.6; // weight on the newest sample when smoothing
// -- damps single-frame noise (motion blur, a mid-transition frame)
// without lagging far behind a real change in the model's read.

// config: { threshold, minSize, debugMode, borderWidth, badgeScale } --
// same shape img-scanner.js takes, resolved once by content.js.
export function createVideoSampler(config) {
  const { threshold, minSize, debugMode, borderWidth, badgeScale } = config;

  const videoState = new Map(); // <video> -> { badge, ema, inFlight }

  function clearVideoIndicator(video) {
    video.style.outline = "";
    const state = videoState.get(video);
    if (state) {
      state.ema = null;
      if (state.badge) {
        state.badge.remove();
        state.badge = null;
      }
    }
  }

  function applyVideoScore(video, pred) {
    const state = videoState.get(video);
    if (!state) return; // untracked (removed/clip-changed) while the request was in flight
    state.ema = state.ema == null ? pred : VIDEO_EMA_ALPHA * pred + (1 - VIDEO_EMA_ALPHA) * state.ema;
    const smoothed = state.ema;
    if (smoothed < threshold && !debugMode) { // see img-scanner.js applyResult()'s DEBUG_MODE comment
      clearVideoIndicator(video);
      return;
    }
    video.style.outline = `${borderWidth}px solid ${badgeColor(smoothed, threshold)}`;
    video.style.outlineOffset = `${-borderWidth / 2}px`;
    if (!state.badge) state.badge = createBadge(badgeScale);
    state.badge.textContent = `AI ${(smoothed * 100).toFixed(0)}%`;
    state.badge.style.background = badgeColor(smoothed, threshold);
    positionBadge(video, state.badge);
  }

  // Returns a base64 JPEG data URL, or null if the frame isn't capturable
  // right now (no decoded dimensions yet) or the site's CDN taints the
  // canvas on drawImage (verified NOT the case for TikTok/Instagram/YouTube
  // Shorts, but sites vary -- fail quiet rather than spam the console).
  function captureFrame(video) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, VIDEO_FRAME_MAX_SIDE / Math.max(vw, vh));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(vw * scale);
    canvas.height = Math.round(vh * scale);
    try {
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.85);
    } catch {
      return null;
    }
  }

  function sampleVideo(video, state) {
    const frame = captureFrame(video);
    if (!frame) return;
    state.inFlight = true;
    scoreFrame(frame).then((resp) => {
      state.inFlight = false;
      if (!videoState.has(video)) return; // removed/reset while this was in flight
      if (!resp || !resp.ok || resp.error) return; // server unreachable/loading -- just skip this tick
      applyVideoScore(video, resp.pred);
    });
  }

  function tick() {
    for (const [video, state] of videoState) {
      if (!video.isConnected) {
        clearVideoIndicator(video);
        videoState.delete(video);
        continue;
      }
      // NOT `|| video.paused`: BUG (found live on Instagram's explore grid --
      // https://www.instagram.com/explore/ -- vs. TikTok's equivalent, which
      // worked): that grid's preview clips are paused by design, only
      // playing on hover/tap, yet still have a real decoded frame sitting on
      // the element (readyState HAVE_ENOUGH_DATA) the instant they scroll
      // into view. Gating on `!paused` meant this loop skipped every one of
      // them forever -- zero requests ever went out, so nothing was ever
      // scored, with no error to show for it. readyState alone is the right
      // gate: it's "is there a frame to capture", not "is it currently
      // playing".
      if (state.inFlight || video.readyState < video.HAVE_CURRENT_DATA) continue;
      const r = video.getBoundingClientRect();
      if (r.width < minSize || r.height < minSize) continue;
      if (r.bottom <= 0 || r.top >= window.innerHeight || r.right <= 0 || r.left >= window.innerWidth) continue;
      sampleVideo(video, state);
    }
  }

  function considerVideo(video) {
    if (videoState.has(video)) return;
    videoState.set(video, { badge: null, ema: null, inFlight: false });
    // Fires when a reused element starts loading a NEW clip -- the previous
    // clip's score is now about a completely different video, so clear it
    // immediately rather than letting it ride over (visibly stale) until
    // the next sample tick relabels it a few seconds later.
    video.addEventListener("loadstart", () => clearVideoIndicator(video));
  }

  function scan(root) {
    if (root.tagName === "VIDEO") considerVideo(root);
    if (root.querySelectorAll) root.querySelectorAll("video").forEach(considerVideo);
  }

  function untrack(root) {
    const videos = root.tagName === "VIDEO" ? [root] : root.querySelectorAll ? [...root.querySelectorAll("video")] : [];
    for (const video of videos) {
      clearVideoIndicator(video);
      videoState.delete(video);
    }
  }

  function reposition() {
    for (const [video, state] of videoState) {
      if (!video.isConnected) {
        clearVideoIndicator(video);
        videoState.delete(video);
      } else if (state.badge) {
        positionBadge(video, state.badge);
      }
    }
  }

  function start() {
    setInterval(tick, VIDEO_SAMPLE_INTERVAL_MS);
  }

  // MutationObserver attribute-change hook (wired by content.js): a site
  // that swaps clips via the `src` attribute directly (rather than
  // MediaSource, which fires `loadstart` on its own -- see considerVideo)
  // needs the same "new clip, stale score" reset here. Keeps the element
  // tracked (unlike untrack()) -- only the stale indicator is cleared.
  const resetOnSrcChange = clearVideoIndicator;

  return { scan, untrack, reposition, start, resetOnSrcChange };
}
