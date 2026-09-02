// heuristics.js -- pure page-signal checks used to decide whether an <img>
// is worth scoring at all. Pulled out of content.js's IntersectionObserver
// callback (where all of this used to live inline) specifically so it can
// be unit tested without a live DOM/IntersectionObserver -- see
// test/heuristics.test.js. Nothing here talks to the network or to
// browser.* APIs; every function is a plain (img, rect) -> boolean.

// Skip images shaped like banners/dividers/full-bleed background art rather
// than a single photo -- see isLikelyDecorative().
export const MIN_ASPECT = 0.3; // narrower than ~1:3.3
export const MAX_ASPECT = 3.0; // wider than ~3:1

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
//
// getStyle is injectable (defaults to the real getComputedStyle) purely so
// tests can supply a fake computed style without a DOM; production callers
// never pass it.
export function isLikelyDecorative(img, rect, opts = {}) {
  const { minAspect = MIN_ASPECT, maxAspect = MAX_ASPECT, getStyle = (el) => getComputedStyle(el) } = opts;
  if (img.getAttribute("aria-hidden") === "true") return true;
  if (getStyle(img).filter.includes("blur(")) return true;
  const aspect = rect.height > 0 ? rect.width / rect.height : 1;
  if (img.alt === "" && (aspect < minAspect || aspect > maxAspect)) return true;
  return false;
}

// Rendered (on-screen) size gate, not decoded/natural pixel size: a 24px
// avatar can be backed by a 256x256 source image (common for retina/shared
// default-avatar assets), and naturalWidth/Height would miss that entirely
// -- what matters is whether it's actually big enough on screen to be worth
// flagging. Also catches 1x1 analytics/tracking pixels, which render at
// (or near) their natural size on essentially every site that uses them.
export function isTooSmall(rect, minSize) {
  return rect.width < minSize || rect.height < minSize;
}

// Combined "don't bother scoring this" decision -- what img-scanner.js's
// IntersectionObserver callback actually calls. Size is checked first
// since it's the cheaper test (no getComputedStyle call) and it's what
// throws out the overwhelming majority of skips (icons, avatars, tracking
// pixels) before paying for the decorative check at all. `opts` is
// forwarded to isLikelyDecorative() unchanged -- see its docstring for why
// getStyle is injectable.
export function shouldSkipImage(img, rect, minSize, opts = {}) {
  return isTooSmall(rect, minSize) || isLikelyDecorative(img, rect, opts);
}
