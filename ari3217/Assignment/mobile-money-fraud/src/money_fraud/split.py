import numpy as np


def make_split(tx, last_step, train_end, valid_end, scored_types, neg_sample, seed):
    sc = tx[tx.type.isin(scored_types) & (tx.step <= last_step)].copy()
    sc["row"] = sc.index.values                      # remember position in the full log (the chain link points at it)
    sc["type"] = sc.type.cat.remove_unused_categories()
    sc = sc.reset_index(drop=True)
    tr = (sc.step <= train_end).values
    va = ((sc.step > train_end) & (sc.step <= valid_end)).values
    te = (sc.step > valid_end).values
    rng = np.random.default_rng(seed)
    keep = rng.random(len(sc)) < neg_sample
    trs = tr & ((sc.isFraud.values == 1) | keep)     # every fraud, a seeded 10 % of legitimate rows
    return sc, tr, va, te, keep, trs