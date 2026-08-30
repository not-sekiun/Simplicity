const DEFAULT_SERVER = "http://127.0.0.1:8765";
const DEFAULT_THRESHOLD = 0.5;
const DEFAULT_MIN_SIZE = 150;

async function refresh() {
  const { serverBase, threshold, minSize } = await chrome.storage.local.get(["serverBase", "threshold", "minSize"]);
  document.getElementById("serverUrl").value = serverBase || DEFAULT_SERVER;
  const t = threshold ?? DEFAULT_THRESHOLD;
  document.getElementById("threshold").value = t;
  document.getElementById("thresholdVal").textContent = t.toFixed(2);
  const m = minSize ?? DEFAULT_MIN_SIZE;
  document.getElementById("minSize").value = m;
  document.getElementById("minSizeVal").textContent = `${m}px`;

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

document.getElementById("save").addEventListener("click", async () => {
  const url = document.getElementById("serverUrl").value.trim();
  const threshold = parseFloat(document.getElementById("threshold").value);
  const minSize = parseInt(document.getElementById("minSize").value, 10);
  await chrome.storage.local.set({ serverBase: url || DEFAULT_SERVER, threshold, minSize });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.id) chrome.tabs.reload(tab.id); // content.js reads config once at page load
  refresh();
});

refresh();
