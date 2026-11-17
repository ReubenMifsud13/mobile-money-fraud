import numpy as np
import pandas as pd

BASE = ["type", "amount", "hour"]
RECEIVER = ["dest_prior_in", "dest_prior_amt", "dest_age_steps"]
CHAIN = ["chain_back"]
GRAPH = BASE + RECEIVER + CHAIN


def add_time_columns(df):
    out = df.copy()
    assert out.step.is_monotonic_increasing          # rows are in time order: every as-of feature below relies on it
    out["hour"] = ((out.step - 1) % 24).astype(np.int8)
    out["day"] = ((out.step - 1) // 24 + 1).astype(np.int8)
    return out


def add_graph_features(df):
    tx = df.copy()
    # ---- as-of graph features: running counts over the receiver's earlier rows (rows are in time order) ----
    g = tx.groupby("dest", sort=False)
    tx["dest_prior_in"] = g.cumcount().astype(np.int32)                      # money received before this row
    tx["dest_prior_amt"] = g.amount.cumsum() - tx.amount                     # value received before this row
    tx["dest_age_steps"] = (tx.step - g.step.transform("min")).astype(np.int16)   # hours since the receiver first appeared
    tx["dest_total_in"] = g.dest.transform("size").astype(np.int32)          # ⚠ whole-month count: uses the future (Trap 2)
    # ---- the mule link: a CASH_OUT whose exact amount left by TRANSFER earlier in the same hour ----
    key = tx.step.astype(np.int64) * 10**12 + np.round(tx.amount * 100).astype(np.int64)
    is_T, is_C = (tx.type == "TRANSFER").values, (tx.type == "CASH_OUT").values
    first_T = pd.Series(np.flatnonzero(is_T), index=key[is_T].values).groupby(level=0).min()
    last_C = pd.Series(np.flatnonzero(is_C), index=key[is_C].values).groupby(level=0).max()
    t_row, c_row, row = key.map(first_T).values, key.map(last_C).values, np.arange(len(tx))
    tx["chain_back"] = (is_C & (t_row < row)).astype(np.int8)                # as-of: the transfer is already in the log
    tx["chain_fwd"] = (is_T & (c_row > row)).astype(np.int8)                 # ⚠ look-ahead: the cash-out has not happened yet (Trap 2)
    tx["t_partner"] = np.where(tx.chain_back == 1, t_row, -1).astype(np.int64)
    tx["traffic_so_far"] = tx.groupby("step").cumcount().astype(np.int32)    # ⚠ the simulator's clock (Trap 3)
    return tx