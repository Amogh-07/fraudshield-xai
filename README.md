# FraudShield XAI

Real-time credit card fraud detection with explainable AI: XGBoost + SHAP, SMOTE, ADWIN drift detection,
a FastAPI backend, an analyst dashboard, and a Chrome extension that warns about risky merchants.

## Project layout
    train.py            train model (XGBoost + SMOTE), retrain guard, saves artifacts/
    app.py              FastAPI backend + SQLite history
    static/dashboard.html   analyst dashboard
    extension/          Chrome extension (Manifest V3)
    baseline.py         ablation: scale_pos_weight instead of SMOTE (one split)
    compare_seeds.py    ablation over 5 splits (mean ± std)
    render.yaml         Render deploy config

## Run locally (Windows / macOS / Linux)
    pip install -r requirements.txt
    # download "Credit Card Fraud Detection" (mlg-ulb) from Kaggle -> Data tab -> Download
    # unzip and place the file at  data/creditcard.csv
    python train.py              # prints real metrics, writes artifacts/
    uvicorn app:app --reload     # dashboard: http://127.0.0.1:8000

Without `data/creditcard.csv`, `train.py` falls back to synthetic data so the app still runs.
Metrics from synthetic data are meaningless. Never quote them.

## How the data is split
10% of rows form a live-stream pool (dashboard demo traffic and analyst labels), the other 90% is split
80/20 into train/test. Analyst labels are added to the training part only, so the test set is never
contaminated and retrain metrics stay honest. SMOTE is applied to the training split only.

## Dashboard features
- Live transaction feed (every 3 s), scrollable history of the last 200, pause and flagged-only filters
- SHAP waterfall per flagged transaction; Confirm fraud / Mark legit buttons queue labels for retraining
- Fraud Score column: uncalibrated model score (SMOTE pushes scores to 0 or 1, so it is not a probability)
- Latency panel: model-only and end-to-end (including SHAP) p50 / p95
- History is stored in SQLite (`artifacts/fraudshield.db`) and survives restarts
- Non-blocking retraining: runs in a background thread, the dashboard keeps updating
- Retrain guard: the new model replaces the old one only if test PR-AUC does not drop by more than 0.005.
  "Test retrain guard" forces a rejection so you can see it work
- Drift: ADWIN watches the stream of model errors against analyst labels. "Simulate drift" feeds 300 normal
  then 300 drifted labelled transactions (fraud features shrunk) and reports when ADWIN fires

## Chrome extension
1. chrome://extensions -> Developer mode -> Load unpacked -> select the `extension/` folder
2. Click the toolbar icon; set the API URL (default http://localhost:8000). No code edits needed after deploying
3. On pages with card fields the extension shows a warning banner (risk level, reasons, report buttons)

The extension calls `/merchant-risk`: a rule-based URL check (HTTPS, odd TLDs, look-alike domains, keywords)
plus crowd reports from extension users. It does NOT run the XGBoost model, because a browser never has the
transaction features (V1-V28, Amount, Time). The XGBoost model scores transactions through `/score`.

## API
POST /score · GET /explain/{id} · POST /simulate · GET /transactions · POST /feedback · POST /simulate-drift
POST /retrain[?force_reject=true] · GET /retrain/status · GET /status · POST /merchant-risk
POST /report-merchant · GET /health

## Ablation (reproduce with compare_seeds.py)
SMOTE vs `scale_pos_weight`, 5 random splits, mean ± std: PR-AUC 0.863±0.024 vs 0.863±0.025,
recall 0.822±0.027 vs 0.822±0.034, precision 0.927±0.007 vs 0.935±0.020. No measurable difference.

## Deploy on Render
1. Train locally on the real data. Commit `artifacts/model.joblib`, `artifacts/sample.csv`,
   `artifacts/metrics.json` (the CSV in `data/` is git-ignored and too big for GitHub)
2. Push to GitHub, create a Render Web Service from the repo (`render.yaml` is included)
3. Set the extension's API URL to your Render URL
4. Run `pip freeze` locally and pin xgboost, shap and scikit-learn to those versions so the saved model loads identically

Limits: the free tier sleeps when idle and has an ephemeral disk, so SQLite history and feedback labels
reset on redeploy. Anyone can POST to `/report-merchant`; there is no auth or rate limiting.
