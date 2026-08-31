const DEFAULT_SERVER = "http://127.0.0.1:8765";
const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_MIN_SIZE = 150;
const DEFAULT_DEBUG_MODE = false;
const DEFAULT_BORDER_WIDTH = 4;
const DEFAULT_BADGE_SCALE = 1;

async function refresh() {
  const { serverBase, threshold, minSize, debugMode, borderWidth, badgeScale } = await chrome.storage.local.get([
    "serverBase",
    "threshold",
    "minSize",
    "debugMode",
    "borderWidth",
    "badgeScale",
  ]);
  document.getElementById("serverUrl").value = serverBase || DEFAULT_SERVER;
  const t = threshold ?? DEFAULT_THRESHOLD;
  document.getElementById("threshold").value = t;
  document.getElementById("thresholdVal").textContent = t.toFixed(2);
  const m = minSize ?? DEFAULT_MIN_SIZE;
  document.getElementById("minSize").value = m;
  document.getElementById("minSizeVal").textContent = `${m}px`;
  const bw = borderWidth ?? DEFAULT_BORDER_WIDTH;
  document.getElementById("borderWidth").value = bw;
  document.getElementById("borderWidthVal").textContent = `${bw}px`;
  const bs = badgeScale ?? DEFAULT_BADGE_SCALE;
  document.getElementById("badgeScale").value = bs;
  document.getElementById("badgeScaleVal").textContent = `${Math.round(bs * 100)}%`;
  document.getElementById("debugMode").checked = debugMode ?? DEFAULT_DEBUG_MODE;

  chrome.runtime.sendMessage({ type: "HEALTH" }, (resp) => {
    const dot = document.getElementById("dot");
    const status = document.getElementById("status");
    if (resp && resp.ok && resp.ready) {
      dot.className = "dot ok";
      status.textContent = "ready";
      document.getElementById("backbone").textContent = resp.backbone || "-";
      document.getElementById("head").textContent = resp.head || "-";
    } else {
      dot.className = "dot bad";
      status.textContent = resp && resp.ok ? "loading model..." : "unreachable";
      document.getElementById("backbone").textContent = "-";
      document.getElementById("head").textContent = "-";
    }
  });
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
  await chrome.storage.local.set({
    serverBase: url || DEFAULT_SERVER,
    threshold,
    minSize,
    borderWidth,
    badgeScale,
    debugMode,
  });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.id) chrome.tabs.reload(tab.id); // content.js reads config once at page load
  refresh();
});

refresh();
