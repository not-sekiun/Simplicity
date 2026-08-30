// background.js -- MV3 service worker.
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
// This file is the ONLY place that knows the server's URL. content.js only
// ever sends {type: "SCORE_BATCH", urls} / {type: "HEALTH"} and receives
// {url, pred} / {url, error} back -- it has no idea a local HTTP server is
// even involved, let alone which port it's on.

const DEFAULT_SERVER = "http://127.0.0.1:8765";

async function getServerBase() {
  const { serverBase } = await chrome.storage.local.get("serverBase");
  return serverBase || DEFAULT_SERVER;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "SCORE_BATCH") {
    (async () => {
      try {
        const base = await getServerBase();
        const res = await fetch(`${base}/score_batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls: msg.urls }),
        });
        if (!res.ok) throw new Error(`server responded ${res.status}`);
        const data = await res.json();
        sendResponse({ ok: true, results: data.results });
      } catch (err) {
        sendResponse({ ok: false, error: String(err) });
      }
    })();
    return true; // keep the message channel open for the async sendResponse
  }

  if (msg.type === "HEALTH") {
    (async () => {
      try {
        const base = await getServerBase();
        const res = await fetch(`${base}/health`, { method: "GET" });
        const data = await res.json();
        sendResponse({ ok: true, ...data });
      } catch (err) {
        sendResponse({ ok: false, error: String(err) });
      }
    })();
    return true;
  }

  return false;
});
