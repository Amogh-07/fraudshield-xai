const $ = s => document.querySelector(s);
const DEFAULT_API = "http://localhost:8000";
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let tabUrl = null;

function render(r, ms) {
  $("#res").innerHTML = `<div class="host">${esc(r.host)}</div>
    <div style="margin:6px 0"><span class="pill ${r.level}">${esc(r.level.toUpperCase())} · ${Math.round(r.risk * 100)}%</span>
    <span class="mu"> ${ms} ms round trip</span></div>
    <ul>${r.reasons.map(x => `<li>${esc(x)}</li>`).join("") || "<li>No red flags found</li>"}</ul>
    <div class="mu">Rule-based merchant check plus user reports. It is not the XGBoost transaction model.</div>`;
  $("#actions").hidden = false;
}

function check() {
  if (!tabUrl || !/^https?:/.test(tabUrl)) { $("#res").textContent = "This page can't be checked."; return; }
  $("#res").textContent = "Checking…";
  const t0 = performance.now();
  chrome.runtime.sendMessage({ type: "merchant-risk", url: tabUrl }, r => {
    if (!r || r.error) { $("#res").textContent = "Can't reach the FraudShield server. Check the API URL below."; return; }
    render(r, Math.round(performance.now() - t0));
  });
}

function report(label) {
  chrome.runtime.sendMessage({ type: "report", url: tabUrl, label }, r => {
    $("#msg").textContent = r && !r.error ? "Thanks, recorded." : "Could not reach the server.";
    if (r && !r.error) check();
  });
}

(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabUrl = tab && tab.url;
  const { api } = await chrome.storage.sync.get({ api: DEFAULT_API });
  $("#api").value = api;
  $("#recheck").onclick = check;
  $("#bad").onclick = () => report(1);
  $("#ok").onclick = () => report(0);
  $("#save").onclick = async () => {
    await chrome.storage.sync.set({ api: $("#api").value.trim() || DEFAULT_API });
    $("#saved").textContent = " saved"; check();
  };
  $("#dash").onclick = async () => {
    const { api: a } = await chrome.storage.sync.get({ api: DEFAULT_API });
    chrome.tabs.create({ url: a });
  };
  check();
})();
