const SEL = 'input[autocomplete="cc-number"],input[autocomplete="cc-csc"],input[name*="card" i],input[id*="card" i],input[name*="cvv" i],input[placeholder*="card number" i]';
let shown = false;

const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function isPaymentPage() {
  return !!document.querySelector(SEL) || /checkout|payment|billing/i.test(location.pathname);
}

function banner(r) {
  const host = document.createElement("div");
  const root = host.attachShadow({ mode: "closed" });
  const color = r.level === "high" ? "#d32f2f" : "#f57c00";
  root.innerHTML = `<style>
    .b{all:initial;position:fixed;top:16px;right:16px;z-index:2147483647;width:330px;font:14px system-ui,sans-serif;
       background:#fff;color:#111;border-left:6px solid ${color};border-radius:8px;box-shadow:0 8px 30px #0005;padding:14px;display:block}
    h4{margin:0 0 6px;color:${color}} ul{margin:6px 0 10px;padding-left:18px} button{margin:0 6px 4px 0;padding:6px 10px;cursor:pointer}
    .m{font-size:12px;color:#555}
  </style><div class="b"><h4>FraudShield: ${esc(r.level.toUpperCase())} merchant risk (${Math.round(r.risk * 100)}%)</h4>
  <div>${esc(r.host)}</div><ul>${r.reasons.map(x => `<li>${esc(x)}</li>`).join("") || "<li>No red flags found</li>"}</ul>
  <button id="x">Dismiss</button><button id="bad">Report as fraud</button><button id="ok">Looks safe</button><div class="m" id="m"></div></div>`;
  const send = label => chrome.runtime.sendMessage({ type: "report", url: location.href, label }, resp => {
    root.getElementById("m").textContent = resp && !resp.error ? "Thanks, recorded." : "Could not reach the FraudShield server.";
  });
  root.getElementById("x").onclick = () => host.remove();
  root.getElementById("bad").onclick = () => send(1);
  root.getElementById("ok").onclick = () => send(0);
  document.documentElement.appendChild(host);
}

function check() {
  if (shown || !isPaymentPage()) return;
  shown = true;
  chrome.runtime.sendMessage({ type: "merchant-risk", url: location.href }, r => {
    if (r && !r.error && r.level !== "low") banner(r);   // warn before the user types card details
  });
}
check();
new MutationObserver(check).observe(document.documentElement, { childList: true, subtree: true });
