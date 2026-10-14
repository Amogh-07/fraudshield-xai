import json, random, sqlite3, threading, time, uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import joblib, numpy as np, pandas as pd, shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from river.drift import ADWIN

import train as T

ART = Path("artifacts")
FB, DBP = ART / "feedback.jsonl", ART / "fraudshield.db"
LOCK = threading.Lock()                      # guards the SQLite connection

app = FastAPI(title="FraudShield XAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = dict(feed=deque(maxlen=200), txns={}, adwin=ADWIN(delta=0.01), drift_events=[],
         lat=deque(maxlen=1000), tot=deque(maxlen=1000),
         retrain=dict(running=False, result=None))

# ------------------------------------------------------------------ persistence (SQLite)
SCHEMA = """
CREATE TABLE IF NOT EXISTS txns(id TEXT PRIMARY KEY, ts REAL, prob REAL, flagged INTEGER, latency_ms REAL,
  total_ms REAL, amount REAL, features TEXT, label INTEGER, explanation TEXT);
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY AUTOINCREMENT, host TEXT, label INTEGER, ts REAL);
"""
COLS = "id,ts,prob,flagged,latency_ms,total_ms,amount,features,label,explanation"


def row_to_rec(r):
    return dict(id=r[0], ts=r[1], prob=r[2], flagged=bool(r[3]), latency_ms=r[4], total_ms=r[5], amount=r[6],
                features=json.loads(r[7]), label=r[8], explanation=json.loads(r[9]) if r[9] else None)


def save_txn(r):
    with LOCK:
        S["db"].execute("INSERT OR REPLACE INTO txns VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (r["id"], r["ts"], r["prob"], int(r["flagged"]), r["latency_ms"], r["total_ms"],
                         r["amount"], json.dumps(r["features"]), r["label"],
                         json.dumps(r["explanation"]) if r.get("explanation") else None))
        S["db"].commit()


def get_txn(tid):
    r = S["txns"].get(tid)
    if r:
        return r
    with LOCK:
        row = S["db"].execute(f"SELECT {COLS} FROM txns WHERE id=?", (tid,)).fetchone()
    return row_to_rec(row) if row else None


def init_db():
    S["db"] = sqlite3.connect(DBP, check_same_thread=False)
    S["db"].executescript(SCHEMA)
    S["db"].execute("DELETE FROM txns WHERE id NOT IN (SELECT id FROM txns ORDER BY ts DESC LIMIT 5000)")
    S["db"].commit()
    rows = S["db"].execute(f"SELECT {COLS} FROM txns ORDER BY ts DESC LIMIT 200").fetchall()
    for row in reversed(rows):               # oldest first, so the newest ends up at the front of the feed
        rec = row_to_rec(row)
        S["txns"][rec["id"]] = rec
        S["feed"].appendleft(rec["id"])
        if rec["latency_ms"] is not None:
            S["lat"].append(rec["latency_ms"])
        if rec["total_ms"] is not None:
            S["tot"].append(rec["total_ms"])


def remember(rec):
    if len(S["feed"]) == S["feed"].maxlen:   # evict the oldest from memory; it stays in SQLite
        S["txns"].pop(S["feed"][-1], None)
    S["feed"].appendleft(rec["id"])
    S["txns"][rec["id"]] = rec
    S["lat"].append(rec["latency_ms"])
    S["tot"].append(rec["total_ms"])
    save_txn(rec)


def n_labels():
    return sum(1 for line in FB.open() if line.strip()) if FB.exists() else 0


def pct(a, q):
    return round(float(np.percentile(list(a), q)), 2) if len(a) else None


# ------------------------------------------------------------------ model
def load():
    b = joblib.load(ART / "model.joblib")
    explainer = shap.TreeExplainer(b["model"])
    sample = pd.read_csv(ART / "sample.csv")
    S.update(model=b["model"], feats=b["features"], thr=b["threshold"], explainer=explainer, sample=sample)


@app.on_event("startup")
def startup():
    ART.mkdir(exist_ok=True)
    if not (ART / "model.joblib").exists():
        print("WARNING: no trained model found; training on",
              "real data" if T.DATA.exists() else "SYNTHETIC data (metrics are meaningless)")
        T.train()
    load()
    init_db()


class Txn(BaseModel):
    features: dict


def explain(X, top=10):
    sv = S["explainer"].shap_values(X)[0]
    base = float(S["explainer"].expected_value)
    order = sorted(range(len(sv)), key=lambda i: -abs(sv[i]))
    rows = [dict(feature=S["feats"][i], value=float(X.iloc[0, i]), shap=float(sv[i])) for i in order[:top]]
    return dict(base=base, contributions=rows, other=float(sum(sv[i] for i in order[top:])))


def score(features: dict):
    t0 = time.perf_counter()
    X = pd.DataFrame([features]).reindex(columns=S["feats"]).fillna(0).astype(float)
    t1 = time.perf_counter()
    prob = float(S["model"].predict_proba(X)[0, 1])
    model_ms = (time.perf_counter() - t1) * 1000
    flagged = prob >= S["thr"]
    rec = dict(id=uuid.uuid4().hex[:8], ts=time.time(), prob=prob, flagged=bool(flagged),
               amount=features.get("Amount"), features=features, label=None, explanation=None)
    if flagged:                              # explanations for every flagged decision
        rec["explanation"] = explain(X)
    rec["latency_ms"] = round(model_ms, 2)                         # model call only
    rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)  # features + model + SHAP
    remember(rec)
    return rec


def public(rec):
    return {k: v for k, v in rec.items() if k not in ("features", "explanation")}


@app.post("/score")
def api_score(t: Txn):
    return public(score(t.features))


@app.get("/explain/{tid}")
def api_explain(tid: str):
    r = get_txn(tid)
    if not r:
        raise HTTPException(404, "unknown transaction")
    if not r.get("explanation"):
        X = pd.DataFrame([r["features"]]).reindex(columns=S["feats"]).fillna(0).astype(float)
        r["explanation"] = explain(X)
        save_txn(r)
    return r["explanation"]


@app.post("/simulate")
def simulate(fraud_boost: float = 0.2):
    """Demo traffic from the held-out live-stream pool; oversamples fraud so the feed isn't empty."""
    d = S["sample"]
    pool = d[d["Class"] == 1] if random.random() < fraud_boost else d[d["Class"] == 0]
    row = pool.sample(1).iloc[0]
    r = score({f: float(row[f]) for f in S["feats"]})
    out = public(r)
    out["true_class"] = int(row["Class"])
    return out


@app.get("/transactions")
def transactions(n: int = 30):
    return [public(S["txns"][i]) for i in list(S["feed"])[:n] if i in S["txns"]]


# ------------------------------------------------------------------ analyst feedback + drift
class Feedback(BaseModel):
    id: str
    label: int  # 1 = confirmed fraud, 0 = legitimate


@app.post("/feedback")
def feedback(f: Feedback):
    r = get_txn(f.id)
    if not r:
        raise HTTPException(404, "unknown transaction")
    r["label"] = f.label
    with FB.open("a") as fh:                 # queue for retraining
        fh.write(json.dumps(dict(**r["features"], Class=f.label)) + "\n")
    save_txn(r)
    S["adwin"].update(int(r["flagged"] != bool(f.label)))   # stream of model errors
    if S["adwin"].drift_detected:
        S["drift_events"].append(time.time())
    return dict(queued=n_labels(), drift=bool(S["adwin"].drift_detected))


@app.post("/simulate-drift")
def simulate_drift(n: int = 300, fraud_rate: float = 0.3):
    """Demo: n normal labelled txns, then n where the fraud pattern has drifted.
    Ground-truth labels stand in for analyst confirmations; the live feed is untouched."""
    d, feats = S["sample"], S["feats"]
    key = ["V14", "V17", "V12", "V10", "V4", "V3", "V16", "V11"]
    rows = []
    for phase in (0, 1):
        for _ in range(n):
            fraud = random.random() < fraud_rate
            row = d[d["Class"] == int(fraud)].sample(1).iloc[0][feats + ["Class"]].copy()
            if phase == 1 and fraud:
                row[key] = row[key] * 0.15   # fraud now looks normal on its strongest signals
            rows.append(row)
    X = pd.DataFrame(rows).reset_index(drop=True)
    y = X.pop("Class").astype(int).values
    flagged = S["model"].predict_proba(X[feats])[:, 1] >= S["thr"]
    err = (flagged != (y == 1)).astype(int)
    S["adwin"], hits = ADWIN(delta=0.01), []
    for i, e in enumerate(err):
        S["adwin"].update(int(e))
        if S["adwin"].drift_detected:
            hits.append(i)
    real = [i for i in hits if i >= n]
    S["drift_events"] = [time.time()] * len(real)
    return dict(error_rate_normal=float(err[:n].mean()), error_rate_drifted=float(err[n:].mean()),
                detected_after=(real[0] - n) if real else None,
                false_alarms_before_drift=len([i for i in hits if i < n]))


# ------------------------------------------------------------------ retraining (background thread)
def _retrain_job(tol):
    try:
        extra = pd.read_json(FB, lines=True) if FB.exists() else None
        m = T.train(extra, tol=tol)
        if m.get("accepted", True):
            load()
            S["adwin"], S["drift_events"] = ADWIN(delta=0.01), []
        S["retrain"].update(running=False, result=m)
    except Exception as e:                   # surfaced to the dashboard instead of hanging silently
        S["retrain"].update(running=False, result=dict(error=str(e)))


@app.post("/retrain")
def retrain(force_reject: bool = False):
    """Starts retraining in the background. force_reject=true sets tolerance to -1 to demo the guard."""
    if S["retrain"]["running"]:
        return dict(started=False, reason="a retrain is already running")
    if n_labels() == 0:
        return dict(started=False, reason="no analyst labels queued yet: confirm or reject a transaction first")
    S["retrain"].update(running=True, result=None)
    threading.Thread(target=_retrain_job, args=(-1.0 if force_reject else 0.005,), daemon=True).start()
    return dict(started=True)


@app.get("/retrain/status")
def retrain_status():
    return S["retrain"]


@app.get("/status")
def status():
    with LOCK:
        reports = S["db"].execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        history = S["db"].execute("SELECT COUNT(*) FROM txns").fetchone()[0]
    return dict(threshold=S["thr"], queued_labels=n_labels(), drift_events=len(S["drift_events"]),
                retrain_recommended=bool(S["drift_events"]), retrain_running=S["retrain"]["running"],
                merchant_reports=reports, history=history,
                latency=dict(model_p50=pct(S["lat"], 50), model_p95=pct(S["lat"], 95),
                             total_p50=pct(S["tot"], 50), total_p95=pct(S["tot"], 95), n=len(S["lat"])),
                metrics=json.loads((ART / "metrics.json").read_text()))


# ------------------------------------------------------------------ merchant risk (used by the Chrome extension)
SUSPICIOUS_TLDS = {"xyz", "top", "click", "icu", "tk", "ml", "ga", "cf", "gq", "buzz", "work"}
KEYWORDS = ("verify", "secure-", "login-", "update-", "wallet", "free", "gift", "prize", "support-")


class Url(BaseModel):
    url: str


@app.post("/merchant-risk")
def merchant_risk(u: Url):
    """Rule-based merchant check (NOT the XGBoost model) plus crowd reports from extension users."""
    p = urlparse(u.url)
    host = (p.hostname or "").lower()
    reasons, risk = [], 0.05

    def add(w, why):
        nonlocal risk
        risk += w
        reasons.append(why)

    if p.scheme != "https": add(0.45, "Page is not served over HTTPS")
    if host.split(".")[-1] in SUSPICIOUS_TLDS: add(0.25, "High-abuse top-level domain")
    if host.startswith("xn--") or ".xn--" in host: add(0.3, "Punycode (look-alike) domain")
    if host.replace(".", "").isdigit(): add(0.35, "Raw IP address instead of a domain")
    if host.count("-") >= 3: add(0.15, "Many hyphens in domain")
    if host.count(".") >= 4: add(0.15, "Unusually deep subdomain chain")
    if any(k in host for k in KEYWORDS): add(0.2, "Phishing-style keyword in domain")
    with LOCK:
        counts = dict(S["db"].execute("SELECT label, COUNT(*) FROM reports WHERE host=? GROUP BY label",
                                      (host,)).fetchall())
    fraud, safe = counts.get(1, 0), counts.get(0, 0)
    if fraud + safe:
        share = (fraud - safe) / (fraud + safe + 3)   # -1..1; the +3 stops one or two votes swinging it
        if share > 0:
            add(0.4 * share, f"Reported as fraud by {fraud} of {fraud + safe} user(s)")
        else:
            risk = max(0.05, risk + 0.1 * share)       # safe votes lower risk by at most 0.1, since votes can be faked
            reasons.append(f"User reports: {fraud} fraud, {safe} safe")
    risk = min(risk, 0.99)
    return dict(host=host, risk=round(risk, 2), level="high" if risk >= .5 else "medium" if risk >= .25 else "low",
                reasons=reasons)


class Report(BaseModel):
    url: str
    label: int  # 1 = fraud, 0 = safe


@app.post("/report-merchant")
def report_merchant(r: Report):
    host = (urlparse(r.url).hostname or "").lower()
    if not host:
        raise HTTPException(400, "bad url")
    with LOCK:
        S["db"].execute("INSERT INTO reports(host,label,ts) VALUES (?,?,?)", (host, int(r.label), time.time()))
        S["db"].commit()
    return dict(host=host, recorded=True)


@app.get("/health")
def health():
    return dict(ok=True)


@app.get("/")
def dash():
    return FileResponse("static/dashboard.html")
