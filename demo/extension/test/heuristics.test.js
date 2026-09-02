// test/heuristics.test.js -- unit tests for the decorative-image /
// too-small-to-bother heuristics pulled out of content.js's scroll handler
// (see src/heuristics.js's module docstring). No DOM: these functions take
// plain objects standing in for an <img>/rect, with getComputedStyle
// injected rather than read off the (nonexistent, under plain node) global
// -- that's the whole reason isLikelyDecorative() accepts a `getStyle`
// override, so this file can run under `node --test` with no jsdom.

import { test } from "node:test";
import assert from "node:assert/strict";
import { isLikelyDecorative, isTooSmall, shouldSkipImage, MIN_ASPECT, MAX_ASPECT } from "../src/heuristics.js";

// Minimal stand-in for an <img> element -- only the members
// isLikelyDecorative() actually touches.
function fakeImg({ ariaHidden = null, alt = "a photo" } = {}) {
  return {
    getAttribute: (name) => (name === "aria-hidden" ? ariaHidden : null),
    alt,
  };
}
const noFilter = () => ({ filter: "none" });
const blurFilter = () => ({ filter: "blur(24px)" });

test("aria-hidden image is decorative regardless of shape", () => {
  const img = fakeImg({ ariaHidden: "true" });
  const rect = { width: 800, height: 600 }; // a perfectly normal photo aspect
  assert.equal(isLikelyDecorative(img, rect, { getStyle: noFilter }), true);
});

test("blur-filtered image is decorative regardless of alt/shape", () => {
  const img = fakeImg({ alt: "a real caption" });
  const rect = { width: 800, height: 600 };
  assert.equal(isLikelyDecorative(img, rect, { getStyle: blurFilter }), true);
});

test("a 1x1 tracking pixel is skipped on size alone", () => {
  const img = fakeImg();
  const rect = { width: 1, height: 1 };
  assert.equal(isTooSmall(rect, 150), true);
  // Size alone is decisive here -- isTooSmall() short-circuits before
  // isLikelyDecorative() would need a getStyle, so no override is passed.
  assert.equal(shouldSkipImage(img, rect, 150), true);
});

test("a normal 800x600 photo is neither decorative nor skipped", () => {
  const img = fakeImg();
  const rect = { width: 800, height: 600 };
  assert.equal(isLikelyDecorative(img, rect, { getStyle: noFilter }), false);
  assert.equal(shouldSkipImage(img, rect, 150, { getStyle: noFilter }), false);
});

test('aspect-ratio rule requires alt="" AND fires only strictly past the boundary', () => {
  assert.equal(MIN_ASPECT, 0.3);
  assert.equal(MAX_ASPECT, 3.0);

  // Exactly MIN_ASPECT: not decorative -- the check is a strict `<`.
  assert.equal(
    isLikelyDecorative(fakeImg({ alt: "" }), { width: 90, height: 300 }, { getStyle: noFilter }),
    false
  );
  // Just narrower than MIN_ASPECT: decorative.
  assert.equal(
    isLikelyDecorative(fakeImg({ alt: "" }), { width: 89, height: 300 }, { getStyle: noFilter }),
    true
  );
  // Exactly MAX_ASPECT: not decorative -- the check is a strict `>`.
  assert.equal(
    isLikelyDecorative(fakeImg({ alt: "" }), { width: 300, height: 100 }, { getStyle: noFilter }),
    false
  );
  // Just wider than MAX_ASPECT: decorative.
  assert.equal(
    isLikelyDecorative(fakeImg({ alt: "" }), { width: 301, height: 100 }, { getStyle: noFilter }),
    true
  );
  // Same extreme aspect, but alt is non-empty: NOT decorative -- an extreme
  // aspect ratio alone is never enough, alt="" is a required second signal
  // (see heuristics.js's module docstring for why).
  assert.equal(
    isLikelyDecorative(fakeImg({ alt: "banner caption" }), { width: 301, height: 100 }, { getStyle: noFilter }),
    false
  );
});
