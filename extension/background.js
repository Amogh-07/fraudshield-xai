// The API URL is stored in chrome.storage (set it in the toolbar popup), so deploying never needs a code change.
const DEFAULT_API = "http://localhost:8000";

async function apiBase() {
  const { api } = await chrome.storage.sync.get({ api: DEFAULT_API });
  return api.replace(/\/+$/, "");
}

async function post(path, body) {
  const r = await fetch((await apiBase()) + path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

function setBadge(tabId, level) {
  if (tabId == null) return;
  const text = level === "high" ? "!" : level === "medium" ? "?" : "";
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color: level === "high" ? "#d32f2f" : "#f57c00" });
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  let job = null;
  if (msg.type === "merchant-risk") {
    job = post("/merchant-risk", { url: msg.url }).then(r => { setBadge(sender.tab && sender.tab.id, r.level); return r; });
  } else if (msg.type === "report") {
    job = post("/report-merchant", { url: msg.url, label: msg.label });
  }
  if (!job) return;
  job.then(reply).catch(e => reply({ error: String(e) }));
  return true; // keep the channel open for the async reply
});
