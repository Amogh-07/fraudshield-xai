"""Ablation: XGBoost with scale_pos_weight (no SMOTE), single split."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, recall_score, precision_score)
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")
feats = [c for c in df.columns if c != "Class"]
tr, te = train_test_split(df, test_size=0.2, stratify=df.Class, random_state=42)
fit, val = train_test_split(tr, test_size=0.2, stratify=tr.Class, random_state=1)

spw = (fit.Class == 0).sum() / (fit.Class == 1).sum()
m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.9, eval_metric="aucpr", scale_pos_weight=spw,
                  n_jobs=-1, random_state=42).fit(fit[feats], fit.Class)

p, r, t = precision_recall_curve(val.Class, m.predict_proba(val[feats])[:, 1])
f1 = 2 * p * r / (p + r + 1e-9)
thr = t[np.argmax(f1[:-1])]
prob = m.predict_proba(te[feats])[:, 1]
pred = (prob >= thr).astype(int)
print(dict(auc=roc_auc_score(te.Class, prob), pr_auc=average_precision_score(te.Class, prob),
           recall=recall_score(te.Class, pred), precision=precision_score(te.Class, pred), thr=float(thr)))
