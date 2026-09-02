// img-scanner.js -- finds <img> elements, decides which are worth scoring
// (heuristics.js), batches them to the server via detector-client.js, and
// applies the result as an outline + overlay badge (overlay.js). Split out
// of the original monolithic content.js. video-sampler.js is a deliberately
// separate module rather than a shared one: a <video>'s identity problem
// (no stable, cacheable equivalent of an <img>'s `src` -- see that file's
// docstring) is different enough that sharing one tracking map between the
// two would just be confusing.

import { shouldSkipImage } from "./heuristics.js";
import { scoreBatch } from "./detector-client.js";
import { badgeColor, positionBadge, createBadge } from "./overlay.js";

const BATCH_SIZE = 6;
const BATCH_DELAY_MS = 150; // debounce so fast scrolling coalesces into one request

// config: { threshold, minSize, debugMode, borderWidth, badgeScale }, all
// resolved once by content.js before any scanner is created -- see that
// file for why `threshold` in particular is resolved async (it may come
// from the server) before this factory ever runs.
export function createImgScanner(config) {
  const { threshold, minSize, debugMode, borderWidth, badgeScale } = config;

  // src -> "pending" | {pred} | {error}. Prevents rescoring the same image
  // (e.g. a repeated avatar) and lets a late result find every element that
  // currently shares that src, even across virtualized-list re-renders.
  const scored = new Map();
  const trackedEls = new Map(); // src -> Set<HTMLImageElement>
  const badges = new Map(); // src -> badge element

  let pending = [];
  let flushTimer = null;

  function removeBadge(src) {
    const badge = badges.get(src);
    if (badge) {
      badge.remove();
      badges.delete(src);
    }
  }

  function upsertBadge(img, src, pred) {
    let badge = badges.get(src);
    if (!badge) {
      badge = createBadge(badgeScale);
      badges.set(src, badge);
    }
    badge.textContent = `AI ${(pred * 100).toFixed(0)}%`;
    badge.style.background = badgeColor(pred, threshold);
    positionBadge(img, badge);
  }

  function applyResult(img, src, result) {
    if ("error" in result) return;
    const pred = result.pred;
    // DEBUG_MODE: annotate every scored image regardless of threshold, so
    // you can see what the model is actually saying about images it
    // wouldn't otherwise flag -- badgeColor() goes green for a confidently
    // "real" pred instead of clearing the outline/badge outright.
    if (pred < threshold && !debugMode) {
      img.style.outline = "";
      removeBadge(src);
      return;
    }
    img.style.outline = `${borderWidth}px solid ${badgeColor(pred, threshold)}`;
    img.style.outlineOffset = `${-borderWidth / 2}px`;
    upsertBadge(img, src, pred);
  }

  function flushQueue() {
    flushTimer = null;
    if (pending.length === 0) return;
    const batch = pending.splice(0, BATCH_SIZE);
    for (const src of batch) scored.set(src, "pending");

    scoreBatch(batch).then((resp) => {
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

        // Rendered (on-screen) size, not decoded/natural pixel size -- see
        // heuristics.js's isTooSmall() docstring for why.
        const rect = entry.boundingClientRect;
        if (shouldSkipImage(img, rect, minSize)) continue;

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

  function scan(root) {
    if (root.tagName === "IMG") considerImage(root);
    if (root.querySelectorAll) root.querySelectorAll("img").forEach(considerImage);
  }

  // Mirror of scan(): drop a removed <img> from tracking immediately
  // (rather than waiting for the next scroll/resize to notice via
  // reposition()) so its badge disappears the moment the box it's
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
  }

  function reposition() {
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
      // than leaving it displayed at its last known position.
      if (!placed) removeBadge(src);
    }
  }

  // MutationObserver attribute-change hook (wired by content.js): lazy-
  // loaded feeds often swap a placeholder src for the real image after the
  // initial load -- re-consider so the upgrade gets scored.
  function reconsiderSrcChange(img) {
    delete img.dataset.aigcObserved;
    considerImage(img);
  }

  return { scan, untrack, reposition, reconsiderSrcChange };
}
