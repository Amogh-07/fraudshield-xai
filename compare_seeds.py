"""Ablation over 5 random splits: SMOTE vs scale_pos_weight (mean ± std). Takes ~10 minutes."""
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score, precision_score
from xgboost import XGBClassifier

df = pd.read_csv("data/creditcard.csv")
feats = [c for c in df.columns if c != "Class"]
kw = dict(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9,
          colsample_bytree=0.9, eval_metric="aucpr", n_jobs=-1)

def evaluate(model, val, te):
    p, r, t = precision_recall_curve(val.Class, model.predict_proba(val[feats])[:, 1])
    thr = t[np.argmax((2 * p * r / (p + r + 1e-9))[:-1])]
    prob = model.predict_proba(te[feats])[:, 1]
    pred = prob >= thr
    return average_precision_score(te.Class, prob), recall_score(te.Class, pred), precision_score(te.Class, pred)

res = {"smote": [], "weight": []}
for seed in range(5):
    tr, te = train_test_split(df, test_size=0.2, stratify=df.Class, random_state=seed)
    fit, val = train_test_split(tr, test_size=0.2, stratify=tr.Class, random_state=seed)
    Xs, ys = SMOTE(random_state=seed).fit_resample(fit[feats], fit.Class)
    res["smote"].append(evaluate(XGBClassifier(**kw, random_state=seed).fit(Xs, ys), val, te))
    spw = (fit.Class == 0).sum() / (fit.Class == 1).sum()
    res["weight"].append(evaluate(XGBClassifier(**kw, scale_pos_weight=spw, random_state=seed).fit(fit[feats], fit.Class), val, te))

for k, v in res.items():
    a = np.array(v)
    print(k, "PR-AUC %.3f±%.3f  recall %.3f±%.3f  precision %.3f±%.3f" %
          (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(), a[:, 2].mean(), a[:, 2].std()))
