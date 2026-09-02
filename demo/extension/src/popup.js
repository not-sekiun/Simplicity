// popup.js -- extension popup UI: shows server status and lets the user
// override the settings content.js reads once at page load (persisted via
// browser.storage.local; saving reloads the active tab so the new value
// takes effect immediately -- content.js has no live-reload path).

import browser from "webextension-polyfill";
import { getHealth, FALLBACK_THRESHOLD } from "./detector-client.js";

const DEFAULT_SERVER = "http://127.0.0.1:8765";
const DEFAULT_MIN_SIZE = 150;
const DEFAULT_DEBUG_MODE = false;
const DEFAULT_BORDER_WIDTH = 4;
const DEFAULT_BADGE_SCALE = 1;

async function refresh() {
  const stored = await browser.storage.local.get([
    "serverBase",
    "threshold",
    "minSize",
    "debugMode",
    "borderWidth",
    "badgeScale",
  ]);
  document.getElementById("serverUrl").value = stored.serverBase || DEFAULT_SERVER;
  const m = stored.minSize ?? DEFAULT_MIN_SIZE;
  document.getElementById("minSize").value = m;
  document.getElementById("minSizeVal").textContent = `${m}px`;
  const bw = stored.borderWidth ?? DEFAULT_BORDER_WIDTH;
  document.getElementById("borderWidth").value = bw;
  document.getElementById("borderWidthVal").textContent = `${bw}px`;
  const bs = stored.badgeScale ?? DEFAULT_BADGE_SCALE;
  document.getElementById("badgeScale").value = bs;
  document.getElementById("badgeScaleVal").textContent = `${Math.round(bs * 100)}%`;
  document.getElementById("debugMode").checked = stored.debugMode ?? DEFAULT_DEBUG_MODE;

  // One /health call serves two things below: the server-status dot, and
  // (when nothing's been explicitly saved) the threshold the slider should
  // show -- content.js resolves the exact same way, in the exact same
  // priority order (see detector-client.resolveThreshold): a saved override
  // wins, else the server's own calibrated value, else FALLBACK_THRESHOLD.
  // Showing anything else here (e.g. a hardcoded 0.5) would make the popup
  // lie about what's actually being used to flag images.
  const health = await getHealth().catch((err) => ({ ok: false, error: String(err) }));

  const t = typeof stored.threshold === "number"
    ? stored.threshold
    : health.ok && typeof health.threshold === "number"
      ? health.threshold
      : FALLBACK_THRESHOLD;
  document.getElementById("threshold").value = t;
  document.getElementById("thresholdVal").textContent = t.toFixed(2);

  const dot = document.getElementById("dot");
  const status = document.getElementById("status");
  if (health.ok && health.ready) {
    dot.className = "dot ok";
    status.textContent = "ready";
    document.getElementById("backbone").textContent = health.backbone || "-";
    document.getElementById("head").textContent = health.head || "-";
  } else {
    dot.className = "dot bad";
    status.textContent = health.ok ? "loading model..." : "unreachable";
    document.getElementById("backbone").textContent = "-";
    document.getElementById("head").textContent = "-";
  }
}

document.getElementById("threshold").addEventListener("input", (e) => {
  document.getElementById("thresholdVal").textContent = parseFloat(e.target.value).toFixed(2);
});
document.getElementById("minSize").addEventListener("input", (e) => {
  document.getElementById("minSizeVal").textContent = `${e.target.value}px`;
});
document.getElementById("borderWidth").addEventListener("input", (e) => {
  document.getElementById("borderWidthVal").textContent = `${e.target.value}px`;
});
document.getElementById("badgeScale").addEventListener("input", (e) => {
  document.getElementById("badgeScaleVal").textContent = `${Math.round(parseFloat(e.target.value) * 100)}%`;
});

document.getElementById("save").addEventListener("click", async () => {
  const url = document.getElementById("serverUrl").value.trim();
  const threshold = parseFloat(document.getElementById("threshold").value);
  const minSize = parseInt(document.getElementById("minSize").value, 10);
  const borderWidth = parseInt(document.getElementById("borderWidth").value, 10);
  const badgeScale = parseFloat(document.getElementById("badgeScale").value);
  const debugMode = document.getElementById("debugMode").checked;
  await browser.storage.local.set({
    serverBase: url || DEFAULT_SERVER,
    threshold,
    minSize,
    borderWidth,
    badgeScale,
    debugMode,
  });

  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.id) browser.tabs.reload(tab.id); // content.js reads config once at page load
  refresh();
});

refresh();
