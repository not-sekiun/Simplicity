// detector-client.js -- the extension's only knowledge of the messaging
// contract with background.js, which is in turn the extension's only
// knowledge of the local inference server (see background.js's module
// docstring for why routing through it, rather than fetching directly, is
// load-bearing). img-scanner.js and video-sampler.js call these functions
// and never touch browser.runtime directly, so the {type, urls} in /
// {url, pred} out contract has exactly one place to change if it ever does.
//
// Promise-based throughout because webextension-polyfill normalizes
// browser.runtime.sendMessage to a Promise on every target (natively on
// Firefox, via a callback-to-Promise shim it installs over chrome.* on
// Chrome/Safari) -- no hand-rolled sendResponse callbacks needed here.

import browser from "webextension-polyfill";

// Wrong-but-conservative beats wrong-but-permissive. This is the decision
// threshold the shipped checkpoint (pe-core-l, linear probe) is actually
// calibrated to -- see AGENTS.md "Current state" / docs/findings.md. It's
// the fallback used ONLY when the server can't be reached or its /health
// response omits `threshold` (an older server build, or a request that
// failed outright): 0.5 would silently run the demo at the wrong operating
// point (18.75% FPR on the same eval tier, measured, vs. 2.15% at 0.980)
// with no visible symptom other than more flagged images than expected --
// a wrong-but-conservative default under-flags instead, which is the
// failure mode you'd rather have with no server to correct it.
export const FALLBACK_THRESHOLD = 0.98;

export async function scoreBatch(urls) {
  return browser.runtime.sendMessage({ type: "SCORE_BATCH", urls });
}

export async function scoreFrame(frame) {
  return browser.runtime.sendMessage({ type: "SCORE_FRAME", frame });
}

// Contract: GET /health via background.js resolves to
// {"ready": bool, "backbone": str, "head": str, "threshold": <float>, ...}.
// `threshold` is the ONE field this module actually depends on, and it's
// treated as optional throughout -- a server that hasn't picked up the
// calibrated-threshold change yet simply omits it, and callers degrade to
// FALLBACK_THRESHOLD rather than assume 0 or throw.
export async function getHealth() {
  return browser.runtime.sendMessage({ type: "HEALTH" });
}

// Resolves the flag threshold once, at startup. Priority order:
//   1. An explicit user override (the popup's slider, persisted to
//      browser.storage.local -- see popup.js) always wins: someone who
//      moved the slider gets what they asked for even if it disagrees with
//      the server.
//   2. Otherwise, the server's own calibrated threshold, asked for fresh
//      every page load rather than cached -- a restarted server may now be
//      serving a different checkpoint with a different calibration.
//   3. Otherwise (server unreachable, or its /health doesn't report one),
//      FALLBACK_THRESHOLD.
export async function resolveThreshold(storedThreshold) {
  if (typeof storedThreshold === "number") return storedThreshold;
  try {
    const health = await getHealth();
    if (health && health.ok && typeof health.threshold === "number") {
      return health.threshold;
    }
  } catch {
    // server unreachable -- fall through to the conservative default
  }
  return FALLBACK_THRESHOLD;
}
