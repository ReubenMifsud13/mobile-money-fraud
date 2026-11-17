import numpy as np

SEED = 3217
LAST_STEP = 718
TRAIN_END = 408
VALID_END = 576
SCORED_TYPES = ["TRANSFER", "CASH_OUT"]
NEG_SAMPLE = 0.10

def make_split(tx):
    sc = tx[tx.type.isin(SCORED_TYPES) & (tx.step <= LAST_STEP)].copy()
    sc["row"] = sc.index.values
    sc["type"] = sc.type.cat.remove_unused_categories()
    sc = sc.reset_index(drop=True)

    tr = (sc.step <= TRAIN_END).values
    va = ((sc.step > TRAIN_END) & (sc.step <= VALID_END)).values
    te = (sc.step > VALID_END).values

    rng = np.random.default_rng(SEED)
    keep = rng.random(len(sc)) < NEG_SAMPLE
    trs = tr & ((sc.isFraud.values == 1) | keep)

    return sc, tr, va, te, keep, trs