"""Train XGBoost + SMOTE.
Uses data/creditcard.csv (Kaggle ULB) if present, else synthetic data (for smoke tests only).

Splits (fixed seeds):  10% live-stream pool (demo traffic + analyst labels, never used for testing)
                       -> remaining 90% split 80/20 into train / test.
Retraining adds analyst labels to the TRAIN part only, and a guard refuses to replace the
current model if test PR-AUC drops by more than `tol`.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA, ART = Path("data/creditcard.csv"), Path("artifacts")


def load_data():
    if DATA.exists():
        return pd.read_csv(DATA)
    X, y = make_classification(n_samples=60000, n_features=30, n_informative=12, weights=[0.99],
                               flip_y=0.01, class_sep=0.8, random_state=42)
    df = pd.DataFrame(X[:, :28], columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = np.abs(X[:, 28]) * 60
    df["Time"] = np.abs(X[:, 29]) * 40000
    df["Class"] = y
    return df


def train(extra: pd.DataFrame | None = None, tol: float = 0.005):
    ART.mkdir(exist_ok=True)
    df = load_data()
    feats = [c for c in df.columns if c != "Class"]
    rest, stream = train_test_split(df, test_size=0.1, stratify=df["Class"], random_state=7)
    tr, te = train_test_split(rest, test_size=0.2, stratify=rest["Class"], random_state=42)

    n_extra = 0
    if extra is not None and len(extra):                 # analyst labels join training only
        extra = extra.drop_duplicates(subset=feats, keep="last")[feats + ["Class"]]
        n_extra = len(extra)
        tr = pd.concat([tr, extra], ignore_index=True)

    fit, val = train_test_split(tr, test_size=0.2, stratify=tr["Class"], random_state=1)
    Xs, ys = SMOTE(random_state=42).fit_resample(fit[feats], fit["Class"])   # SMOTE on fit split only
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="aucpr", n_jobs=-1, random_state=42)
    model.fit(Xs, ys)

    # threshold chosen on untouched validation data (max F1), never on the test set
    p, r, t = precision_recall_curve(val["Class"], model.predict_proba(val[feats])[:, 1])
    f1 = 2 * p * r / (p + r + 1e-9)
    thr = float(t[np.argmax(f1[:-1])])

    prob = model.predict_proba(te[feats])[:, 1]
    pred = (prob >= thr).astype(int)
    m = dict(auc_roc=float(roc_auc_score(te["Class"], prob)),
             pr_auc=float(average_precision_score(te["Class"], prob)),
             accuracy=float(accuracy_score(te["Class"], pred)),
             fraud_recall=float(recall_score(te["Class"], pred)),
             fraud_precision=float(precision_score(te["Class"], pred, zero_division=0)),
             fraud_rate_test=float(te["Class"].mean()), threshold=thr,
             feedback_rows=n_extra, tolerance=tol)

    old = ART / "metrics.json"
    if extra is not None and old.exists():               # retrain guard
        prev = json.loads(old.read_text())
        if m["pr_auc"] < prev["pr_auc"] - tol:
            m["accepted"], m["previous_pr_auc"] = False, prev["pr_auc"]
            print(json.dumps(m, indent=2))
            return m
    m["accepted"] = True

    joblib.dump(dict(model=model, features=feats, threshold=thr), ART / "model.joblib")
    legit = stream[stream["Class"] == 0]                 # capped live-stream pool keeps the repo small
    pool = pd.concat([stream[stream["Class"] == 1], legit.sample(min(5000, len(legit)), random_state=1)])
    pool.to_csv(ART / "sample.csv", index=False)
    (ART / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    train()
