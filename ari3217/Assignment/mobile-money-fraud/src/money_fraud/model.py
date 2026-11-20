import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# the model's design stays in code; seed and n_jobs come from config
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def train_model(X, y, seed, n_jobs):
    return lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed, n_jobs=n_jobs).fit(X, y)


def precision_at_k(y_true, score, k):
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    yv = np.asarray(y_true)
    return yv[top].mean(), yv[top].sum() / yv.sum()


def fit_eval(sc, trs, te, cols, name, k, seed, n_jobs):
    t1 = time.time()
    m = train_model(sc.loc[trs, cols], sc.isFraud.values[trs], seed, n_jobs)
    p = m.predict_proba(sc.loc[te, cols])[:, 1]
    y_te = sc.isFraud.values[te]
    pk, rk = precision_at_k(y_te, p, k)
    return m, p, {"model": name, "AUPRC": average_precision_score(y_te, p), "ROC-AUC": roc_auc_score(y_te, p),
                  "precision@budget": pk, "recall@budget": rk, "fit_s": time.time() - t1}


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)