// background.js -- MV3 service worker (Chrome/Safari) / background script
// (Firefox, via manifest.base.json's per-target background key -- see
// build.js).
//
// Every model call is routed through here rather than fetched directly from
// content.js: content-script code runs inside the PAGE's context for
// networking purposes, so a site with a strict Content-Security-Policy
// (Instagram in particular restricts `connect-src`) can silently block a
// fetch() to http://127.0.0.1 from a content script. A background service
// worker is its own extension-origin context and is not subject to the
// host page's CSP, so routing here is what makes this work on sites that
// would otherwise block it.
//
// This file is the ONLY place that knows the server's URL. content.js's
// modules only ever send {type: "SCORE_BATCH", urls} / {type: "SCORE_FRAME",
// frame} / {type: "HEALTH"} (via detector-client.js) and receive
// {url, pred}/{url, error} or {pred}/{error} back -- they have no idea a
// local HTTP server is even involved, let alone which port it's on.

import browser from "webextension-polyfill";

const DEFAULT_SERVER = "http://127.0.0.1:8765";

async function getServerBase() {
  const { serverBase } = await browser.storage.local.get("serverBase");
  return serverBase || DEFAULT_SERVER;
}

async function handleScoreBatch(urls) {
  const base = await getServerBase();
  const res = await fetch(`${base}/score_batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls }),
  });
  if (!res.ok) throw new Error(`server responded ${res.status}`);
  const data = await res.json();
  return { ok: true, results: data.results };
}

async function handleScoreFrame(frame) {
  const base = await getServerBase();
  const res = await fetch(`${base}/score_frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame }),
  });
  if (!res.ok) throw new Error(`server responded ${res.status}`);
  const data = await res.json();
  return { ok: true, pred: data.pred, error: data.error };
}

async function handleHealth() {
  const base = await getServerBase();
  const res = await fetch(`${base}/health`, { method: "GET" });
  const data = await res.json();
  return { ok: true, ...data };
}

// webextension-polyfill lets a listener return a Promise directly instead
// of the callback-plus-`return true`-to-keep-the-channel-open dance MV3's
// native chrome.runtime.onMessage requires -- it handles that translation
// on Chrome/Safari itself, so this works identically on Firefox (which
// already supports returning a Promise natively).
browser.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SCORE_BATCH") {
    return handleScoreBatch(msg.urls).catch((err) => ({ ok: false, error: String(err) }));
  }
  if (msg.type === "SCORE_FRAME") {
    return handleScoreFrame(msg.frame).catch((err) => ({ ok: false, error: String(err) }));
  }
  if (msg.type === "HEALTH") {
    return handleHealth().catch((err) => ({ ok: false, error: String(err) }));
  }
  return undefined; // not a message this listener handles
});
