// content.js -- finds <img> elements, scores the ones large enough to
// matter as they scroll into view, and overlays a confidence-colored
// outline + "AI NN%" tag on the ones flagged. Also samples <video> elements
// (TikTok/Instagram/YouTube Shorts-style feeds) every few seconds while
// they're playing on-screen, for the same treatment -- see "video scoring"
// below for why that's a materially different tracking problem than <img>.
//
// THE ABSTRACTION: this file talks to the model through exactly two calls,
// chrome.runtime.sendMessage({type: "SCORE_BATCH", urls}) for images and
// {type: "SCORE_FRAME", frame} (one base64 JPEG, captured from a <video> via
// canvas) for video, and gets back {url, pred}/{pred} shapes. It has zero
// knowledge of backbones, checkpoints, or parameter counts -- swap
// demo/server.py's --head to a retrained or entirely different backbone and
// nothing here changes.

(async () => {
  const DEFAULTS = { threshold: 0.5, minSize: 150, debugMode: false };
  const { threshold: AIGC_THRESHOLD, minSize: MIN_SIZE, debugMode: DEBUG_MODE } = await loadConfig();

  const BATCH_SIZE = 6;
  const BATCH_DELAY_MS = 150; // debounce so fast scrolling coalesces into one request

  // Skip images shaped like banners/dividers/full-bleed background art rather
  // than a single photo -- see isLikelyDecorative().
  const MIN_ASPECT = 0.3; // narrower than ~1:3.3
  const MAX_ASPECT = 3.0; // wider than ~3:1

  async function loadConfig() {
    const stored = await chrome.storage.local.get(["threshold", "minSize", "debugMode"]);
    return { ...DEFAULTS, ...stored };
  }

  // Filters out page chrome that happens to be a large <img> but isn't a
  // "picture" in any sense a viewer would recognize -- decorative banners,
  // full-bleed background art behind other UI, etc. Found live on r/pics:
  // a login-promo sidebar's backdrop illustration was 283x1255 displayed
  // (comfortably over any reasonable size threshold) but aria-hidden="true"
  // and alt="", object-fit:cover, position:absolute -- Reddit's own markup
  // already says "this isn't content," we just weren't reading it.
  //
  // Three independent signals, since none alone is both safe and sufficient:
  //  - aria-hidden="true" is authoritative (a real content photo is never
  //    marked this way) but not every site sets it.
  //  - alt="" alone is too common on legitimate photos across the web to
  //    use by itself (would create false negatives), so it only counts
  //    combined with an extreme aspect ratio -- real photos are rarely
  //    banner-shaped, decorative fill art usually is.
  //  - CSS filter: blur(...) on the <img> itself is also authoritative: no
  //    photo a viewer would recognize as "the picture" is rendered blurred.
  //    Reddit duplicates every post photo -- a sharp foreground copy plus a
  //    blurred, alt="" backdrop clone stretched behind it as object-fit:cover
  //    filler (confirmed live: same src, both ~840x648/blur(24px) vs the
  //    700x540 sharp copy, on every post checked) -- so this also kills the
  //    resulting double-badge, without touching real photos: a deliberately
  //    blurred photo (e.g. an NSFW spoiler cover) is rare enough that
  //    leaving it flagged beats silently skipping it.
  function isLikelyDecorative(img, rect) {
    if (img.getAttribute("aria-hidden") === "true") return true;
    if (getComputedStyle(img).filter.includes("blur(")) return true;
    const aspect = rect.height > 0 ? rect.width / rect.height : 1;
    if (img.alt === "" && (aspect < MIN_ASPECT || aspect > MAX_ASPECT)) return true;
    return false;
  }

  // src -> "pending" | {pred} | {error}. Prevents rescoring the same image
  // (e.g. a repeated avatar) and lets a late result find every element that
  // currently shares that src, even across virtualized-list re-renders.
  const scored = new Map();
  const trackedEls = new Map(); // src -> Set<HTMLImageElement>
  const badges = new Map(); // src -> badge element

  let pending = [];
  let flushTimer = null;

  // One continuous scale, green -> yellow -> red, so debug mode (which
  // shows a badge below the threshold too) doesn't need a second unrelated
  // color scheme: green (confidently real, pred near 0) rises to yellow
  // exactly AT the threshold from both sides, then yellow -> red as
  // confidence in "AI-generated" rises to 1. Only the >= threshold half
  // is ever seen outside debug mode.
  function badgeColor(pred) {
    if (pred >= AIGC_THRESHOLD) {
      const t = Math.max(0, Math.min(1, (pred - AIGC_THRESHOLD) / (1 - AIGC_THRESHOLD)));
      const g = Math.round(200 * (1 - t));
      return `rgb(255, ${g}, 0)`;
    }
    const t = AIGC_THRESHOLD > 0 ? Math.max(0, Math.min(1, pred / AIGC_THRESHOLD)) : 1;
    const r = Math.round(255 * t);
    return `rgb(${r}, 200, 0)`;
  }

  function ensureOverlayRoot() {
    let root = document.getElementById("__aigc_detect_overlay_root__");
    if (!root) {
      root = document.createElement("div");
      root.id = "__aigc_detect_overlay_root__";
      document.documentElement.appendChild(root);
    }
    return root;
  }

  // The badge is a free-floating div, not part of the <img>'s own paint
  // layer -- unlike `outline`, nothing stops it from rendering in front of
  // page content that has popped up *over* the image (a modal, a lightbox,
  // another card sliding in). Sample the point the box actually occupies
  // and hide the badge whenever something else is frontmost there, so the
  // label disappears in lockstep with the box it's annotating instead of
  // floating in front of whatever is now covering it.
  function isElementOccluded(el, r) {
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) {
      return false; // off-screen -- not "covered", just out of the viewport
    }
    const top = document.elementFromPoint(cx, cy);
    return !!top && top !== el && !el.contains(top) && !top.contains(el);
  }

  // Shared by both <img> and <video> badges.
  function positionBadge(el, badge) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0 || isElementOccluded(el, r)) {
      badge.style.display = "none";
      return;
    }
    badge.style.display = "block";
    badge.style.top = `${r.top + window.scrollY}px`;
    badge.style.left = `${r.left + window.scrollX}px`;
  }

  function upsertBadge(img, src, pred) {
    const root = ensureOverlayRoot();
    let badge = badges.get(src);
    if (!badge) {
      badge = document.createElement("div");
      badge.className = "__aigc_detect_badge__";
      root.appendChild(badge);
      badges.set(src, badge);
    }
    badge.textContent = `AI ${(pred * 100).toFixed(0)}%`;
    badge.style.background = badgeColor(pred);
    positionBadge(img, badge);
  }

  function removeBadge(src) {
    const badge = badges.get(src);
    if (badge) {
      badge.remove();
      badges.delete(src);
    }
  }

  function applyResult(img, src, result) {
    if ("error" in result) return;
    const pred = result.pred;
    // DEBUG_MODE: annotate every scored image regardless of threshold, so
    // you can see what the model is actually saying about images it
    // wouldn't otherwise flag -- badgeColor() goes green for a confidently
    // "real" pred instead of clearing the outline/badge outright.
    if (pred < AIGC_THRESHOLD && !DEBUG_MODE) {
      img.style.outline = "";
      removeBadge(src);
      return;
    }
    img.style.outline = `4px solid ${badgeColor(pred)}`;
    img.style.outlineOffset = "-2px";
    upsertBadge(img, src, pred);
  }

  function repositionAll() {
    // Snapshot with [...] -- removeBadge() mutates `badges` mid-iteration
    // for any src whose last connected element just disappeared.
    for (const [src, badge] of [...badges]) {
      const els = trackedEls.get(src);
      let placed = false;
      if (els) {
        for (const el of els) {
          if (el.isConnected) {
            positionBadge(el, badge);
            placed = true;
            break;
          }
        }
      }
      // No connected element left to anchor to (unmounted from a
      // virtualized list, feed re-render, etc.) -- remove the badge rather
      // than leaving it displayed at its last known position. The box
      // (an inline style on the element itself) already vanished along
      // with the element; the label must go with it, not linger.
      if (!placed) removeBadge(src);
    }

    // Video badges are keyed by element, not src (see "video scoring"
    // below), so there's no multi-element fallback to try -- an
    // element that's gone just loses its badge outright.
    for (const [video, state] of videoState) {
      if (!video.isConnected) {
        clearVideoIndicator(video);
        videoState.delete(video);
      } else if (state.badge) {
        positionBadge(video, state.badge);
      }
    }
  }

  let rafScheduled = false;
  function scheduleReposition() {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
      rafScheduled = false;
      repositionAll();
    });
  }
  window.addEventListener("scroll", scheduleReposition, { passive: true, capture: true });
  window.addEventListener("resize", scheduleReposition, { passive: true });

  function flushQueue() {
    flushTimer = null;
    if (pending.length === 0) return;
    const batch = pending.splice(0, BATCH_SIZE);
    for (const src of batch) scored.set(src, "pending");

    chrome.runtime.sendMessage({ type: "SCORE_BATCH", urls: batch }, (resp) => {
      if (!resp || !resp.ok) {
        // Server unreachable/loading -- drop the "pending" mark so these
        // get retried the next time they're (re-)observed, e.g. on scroll.
        for (const src of batch) scored.delete(src);
        return;
      }
      for (const r of resp.results) {
        scored.set(r.url, r);
        const els = trackedEls.get(r.url);
        if (!els) continue;
        for (const el of els) {
          if (el.isConnected) applyResult(el, r.url, r);
        }
      }
    });

    if (pending.length > 0) scheduleFlush();
  }

  function scheduleFlush() {
    if (flushTimer) return;
    flushTimer = setTimeout(flushQueue, BATCH_DELAY_MS);
  }

  function enqueue(src) {
    if (scored.has(src)) return; // already scored, pending, or errored
    pending.push(src);
    scheduleFlush();
  }

  function trackImage(img) {
    const src = img.currentSrc || img.src;
    if (!src || !src.startsWith("http")) return;
    if (!trackedEls.has(src)) trackedEls.set(src, new Set());
    trackedEls.get(src).add(img);

    const existing = scored.get(src);
    if (existing && existing !== "pending") {
      applyResult(img, src, existing); // already-known result, e.g. a repeated avatar
    }
  }

  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target;
        io.unobserve(img);

        // Rendered (on-screen) size, not decoded/natural pixel size: a 24px
        // avatar can be backed by a 256x256 source image (common for
        // retina/shared default-avatar assets), and naturalWidth/Height
        // would miss that entirely -- what matters is whether it's actually
        // big enough on screen to be worth flagging.
        const rect = entry.boundingClientRect;
        if (rect.width < MIN_SIZE || rect.height < MIN_SIZE) continue; // icons/avatars/emoji
        if (isLikelyDecorative(img, rect)) continue; // banners/backdrops, not photos

        const src = img.currentSrc || img.src;
        if (src) enqueue(src);
      }
    },
    { rootMargin: "200px" } // start scoring slightly before it's on-screen
  );

  function considerImage(img) {
    if (img.dataset.aigcObserved) return;
    img.dataset.aigcObserved = "1";
    if (img.complete && img.naturalWidth > 0) {
      trackImage(img);
      io.observe(img);
    } else {
      img.addEventListener(
        "load",
        () => {
          trackImage(img);
          io.observe(img);
        },
        { once: true }
      );
    }
  }

  // ---- video scoring ---------------------------------------------------
  //
  // Images are cached by `src` because the same photo genuinely recurs (a
  // repeated avatar, a reposted thumbnail). Video on a short-form feed has
  // no equivalent: a <video>'s `src`/`currentSrc` is typically a per-clip
  // `blob:` URL handed out by MediaSource, and the feed reuses a small pool
  // of <video> elements, loading a brand new clip into the SAME element as
  // you scroll past it. So state here is keyed by the element itself, not
  // a src -- there's no "already scored" cache, only "currently sampling
  // whatever clip is loaded into this element right now" -- and it's
  // explicitly reset the instant that clip changes underneath us.
  const VIDEO_SAMPLE_INTERVAL_MS = 3000; // a couple of seconds is fine --
  // this isn't meant to be frame-accurate, and it keeps request volume sane
  // against a server that scores one batch at a time (see server.py).
  const VIDEO_FRAME_MAX_SIDE = 480; // downscaled before sending -- the
  // backbone re-resizes to its own native_res anyway (see server.py), so
  // shipping a full 1080x1920 frame over sendMessage would be pure waste.
  const VIDEO_EMA_ALPHA = 0.6; // weight on the newest sample when smoothing
  // -- damps single-frame noise (motion blur, a mid-transition frame)
  // without lagging far behind a real change in the model's read.

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
    if (smoothed < AIGC_THRESHOLD && !DEBUG_MODE) { // see applyResult()'s DEBUG_MODE comment
      clearVideoIndicator(video);
      return;
    }
    video.style.outline = `4px solid ${badgeColor(smoothed)}`;
    video.style.outlineOffset = "-2px";
    if (!state.badge) {
      state.badge = document.createElement("div");
      state.badge.className = "__aigc_detect_badge__";
      ensureOverlayRoot().appendChild(state.badge);
    }
    state.badge.textContent = `AI ${(smoothed * 100).toFixed(0)}%`;
    state.badge.style.background = badgeColor(smoothed);
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
    chrome.runtime.sendMessage({ type: "SCORE_FRAME", frame }, (resp) => {
      state.inFlight = false;
      if (!videoState.has(video)) return; // removed/reset while this was in flight
      if (!resp || !resp.ok || resp.error) return; // server unreachable/loading -- just skip this tick
      applyVideoScore(video, resp.pred);
    });
  }

  function videoTick() {
    for (const [video, state] of videoState) {
      if (!video.isConnected) {
        clearVideoIndicator(video);
        videoState.delete(video);
        continue;
      }
      if (state.inFlight || video.paused || video.readyState < video.HAVE_CURRENT_DATA) continue;
      const r = video.getBoundingClientRect();
      if (r.width < MIN_SIZE || r.height < MIN_SIZE) continue;
      if (r.bottom <= 0 || r.top >= window.innerHeight || r.right <= 0 || r.left >= window.innerWidth) continue;
      sampleVideo(video, state);
    }
  }
  setInterval(videoTick, VIDEO_SAMPLE_INTERVAL_MS);

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
    if (root.tagName === "IMG") considerImage(root);
    if (root.tagName === "VIDEO") considerVideo(root);
    if (root.querySelectorAll) {
      root.querySelectorAll("img").forEach(considerImage);
      root.querySelectorAll("video").forEach(considerVideo);
    }
  }

  // Mirror of scan(): drop a removed <img>/<video> from tracking immediately
  // (rather than waiting for the next scroll/resize to notice via
  // repositionAll) so its badge disappears the moment the box it's
  // annotating does, instead of lingering until the user happens to scroll.
  function untrack(root) {
    const imgs = root.tagName === "IMG" ? [root] : root.querySelectorAll ? [...root.querySelectorAll("img")] : [];
    for (const img of imgs) {
      const src = img.currentSrc || img.src;
      const els = trackedEls.get(src);
      if (!els) continue;
      els.delete(img);
      if (els.size === 0) removeBadge(src);
    }

    const videos = root.tagName === "VIDEO" ? [root] : root.querySelectorAll ? [...root.querySelectorAll("video")] : [];
    for (const video of videos) {
      clearVideoIndicator(video);
      videoState.delete(video);
    }
  }

  scan(document.body);

  const mo = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "childList") {
        for (const node of m.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) scan(node);
        }
        for (const node of m.removedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) untrack(node);
        }
      } else if (m.type === "attributes" && m.target.tagName === "IMG") {
        // Lazy-loaded feeds often swap a placeholder src for the real image
        // after the initial load -- re-consider so the upgrade gets scored.
        delete m.target.dataset.aigcObserved;
        considerImage(m.target);
      } else if (m.type === "attributes" && m.target.tagName === "VIDEO") {
        // A site that swaps clips via the `src` attribute directly (rather
        // than MediaSource, which fires `loadstart` on its own -- see
        // considerVideo) needs the same "new clip, stale score" reset here.
        clearVideoIndicator(m.target);
      }
    }
  });
  mo.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset"],
  });
})();
