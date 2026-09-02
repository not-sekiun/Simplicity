// overlay.js -- the floating "AI NN%" badge and the outline color it and
// img-scanner.js/video-sampler.js's outlines share. Generic over WHAT is
// being annotated (<img> or <video>): those two modules keep their own
// tracking maps, keyed differently, because a <video>'s identity problem is
// different from an <img>'s (see video-sampler.js's module docstring) --
// but both badges are built, positioned, and colored identically, so that
// part lives once, here, instead of twice.

let overlayRoot = null;

// One overlay root, appended directly to <html> -- a sibling of the page's
// own app root (e.g. Reddit/Instagram's React #root), never a descendant of
// it, so the site's own framework never has to reconcile anything this
// extension adds and never touches it. badgeScale is read by overlay.css's
// .__aigc_detect_badge__ rule via this custom property, so font size and
// padding both scale together instead of the badge just growing taller.
export function ensureOverlayRoot(badgeScale) {
  if (!overlayRoot || !overlayRoot.isConnected) {
    overlayRoot = document.createElement("div");
    overlayRoot.id = "__aigc_detect_overlay_root__";
    document.documentElement.appendChild(overlayRoot);
  }
  overlayRoot.style.setProperty("--aigc-badge-scale", badgeScale);
  return overlayRoot;
}

export function createBadge(badgeScale) {
  const badge = document.createElement("div");
  badge.className = "__aigc_detect_badge__";
  ensureOverlayRoot(badgeScale).appendChild(badge);
  return badge;
}

// One continuous scale, green -> yellow -> red, so debug mode (which shows
// a badge below the threshold too) doesn't need a second unrelated color
// scheme: green (confidently real, pred near 0) rises to yellow exactly AT
// the threshold from both sides, then yellow -> red as confidence in
// "AI-generated" rises to 1. Only the >= threshold half is ever seen
// outside debug mode.
export function badgeColor(pred, threshold) {
  if (pred >= threshold) {
    const t = Math.max(0, Math.min(1, (pred - threshold) / (1 - threshold)));
    const g = Math.round(200 * (1 - t));
    return `rgb(255, ${g}, 0)`;
  }
  const t = threshold > 0 ? Math.max(0, Math.min(1, pred / threshold)) : 1;
  const r = Math.round(255 * t);
  return `rgb(${r}, 200, 0)`;
}

// The badge is a free-floating div, not part of the annotated element's own
// paint layer -- unlike `outline`, nothing stops it from rendering in front
// of page content that has popped up *over* the element (a modal, a
// lightbox, another card sliding in). Sample the point the box actually
// occupies and hide the badge whenever something else is frontmost there,
// so the label disappears in lockstep with the box it's annotating instead
// of floating in front of whatever is now covering it.
// BUG (found live on Instagram/TikTok feeds, both: outline shows, badge
// never does): elementFromPoint() only returns the SINGLE topmost hit at
// that point. Every video on both sites sits under one or more plain
// <div>s stacked above it purely for click/tap handling (mute toggle,
// tap-to-pause, a controls layer) -- fully transparent, no visible paint of
// their own. The old check treated any such div as "occluding" since it's
// neither the element nor a container/containee of it, so the badge was
// hidden on essentially every video on both sites even though nothing was
// actually covering it on screen.
//
// elementsFromPoint() (plural) returns the FULL paint-order stack at that
// point instead of just the top hit. Walk it looking for el, skipping past
// anything with no visible paint of its own (transparent background, no
// background image) -- those are hit-testing layers, not occlusion. Only a
// node with actual paint sitting in front of el (a modal, a lightbox, a
// card that's slid in over it) counts as truly occluding.
export function isElementOccluded(el, r) {
  const cx = r.left + r.width / 2;
  const cy = r.top + r.height / 2;
  if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) {
    return false; // off-screen -- not "covered", just out of the viewport
  }
  const stack = document.elementsFromPoint(cx, cy);
  for (const node of stack) {
    if (node === el || node.contains(el) || el.contains(node)) return false;
    const style = getComputedStyle(node);
    const hasVisiblePaint =
      style.backgroundImage !== "none" || !/^(rgba\(0,\s*0,\s*0,\s*0\)|transparent)$/.test(style.backgroundColor);
    if (hasVisiblePaint) return true; // something actually painted is in front of el
  }
  return true; // el never showed up in the stack at all
}

// Shared by both <img> and <video> badges.
export function positionBadge(el, badge) {
  const r = el.getBoundingClientRect();
  if (r.width === 0 || r.height === 0 || isElementOccluded(el, r)) {
    badge.style.display = "none";
    return;
  }
  badge.style.display = "block";
  badge.style.top = `${r.top + window.scrollY}px`;
  badge.style.left = `${r.left + window.scrollX}px`;
}
