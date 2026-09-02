// content.js -- thin entry point. Finds <img>/<video> elements as the page
// mutates (infinite scroll, lazy-loaded feeds) and delegates everything
// else to img-scanner.js and video-sampler.js, which score the ones that
// matter and overlay a confidence-colored outline + "AI NN%" tag on the
// ones flagged (overlay.js).
//
// THE ABSTRACTION: this module (via detector-client.js) talks to the model
// through exactly two calls -- scoreBatch(urls) for images and
// scoreFrame(frame) (one base64 JPEG, captured from a <video> via canvas)
// for video -- and gets back {url, pred}/{pred} shapes. It has zero
// knowledge of backbones, checkpoints, or parameter counts -- swap
// demo/server.py's --head to a retrained or entirely different backbone
// and nothing here changes. KEEP THIS CONTRACT EXACTLY if you touch
// anything in this file or the modules it wires together.

import browser from "webextension-polyfill";
import { resolveThreshold } from "./detector-client.js";
import { createImgScanner } from "./img-scanner.js";
import { createVideoSampler } from "./video-sampler.js";

(async () => {
  const DEFAULTS = { minSize: 150, debugMode: false, borderWidth: 4, badgeScale: 1 };
  const stored = await browser.storage.local.get(["threshold", "minSize", "debugMode", "borderWidth", "badgeScale"]);
  const config = { ...DEFAULTS, ...stored };
  // Resolved async (may be a round trip to the server) BEFORE either
  // scanner is constructed -- see detector-client.resolveThreshold's
  // docstring for the priority order (user override > server's calibrated
  // value > conservative fallback).
  config.threshold = await resolveThreshold(stored.threshold);

  const imgScanner = createImgScanner(config);
  const videoSampler = createVideoSampler(config);
  videoSampler.start();

  function scan(root) {
    imgScanner.scan(root);
    videoSampler.scan(root);
  }

  function untrack(root) {
    imgScanner.untrack(root);
    videoSampler.untrack(root);
  }

  let rafScheduled = false;
  function scheduleReposition() {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
      rafScheduled = false;
      imgScanner.reposition();
      videoSampler.reposition();
    });
  }
  window.addEventListener("scroll", scheduleReposition, { passive: true, capture: true });
  window.addEventListener("resize", scheduleReposition, { passive: true });

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
        imgScanner.reconsiderSrcChange(m.target);
      } else if (m.type === "attributes" && m.target.tagName === "VIDEO") {
        videoSampler.resetOnSrcChange(m.target);
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
