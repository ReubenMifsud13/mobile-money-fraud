import numpy as np
import lightgbm as lgb

# hardcoded for now; Stage 3 moves these into config
SEED = 3217
N_JOBS = 1

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, verbose=-1,
                  n_jobs=N_JOBS)


def train_model(X, y):
    return lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)


def precision_at_k(y_true, score, k):
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    yv = np.asarray(y_true)
    return yv[top].mean(), yv[top].sum() / yv.sum()