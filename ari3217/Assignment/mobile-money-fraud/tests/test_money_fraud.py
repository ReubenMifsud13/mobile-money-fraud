import numpy as np
import pandas as pd
import pytest

from money_fraud.features import GRAPH, add_graph_features, add_time_columns
from money_fraud.model import precision_at_k, train_model
from money_fraud.split import make_split

# test inputs, not config: unit tests must not depend on config.yaml
LAST_STEP, TRAIN_END, VALID_END = 718, 408, 576
SCORED_TYPES = ["TRANSFER", "CASH_OUT"]


# ---------- time columns ----------
def test_add_time_columns_values():
    df = pd.DataFrame({"step": [1, 24, 25, 48]})
    out = add_time_columns(df)
    assert out["hour"].tolist() == [0, 23, 0, 23]
    assert out["day"].tolist() == [1, 1, 2, 2]


def test_add_time_columns_does_not_modify_input():
    df = pd.DataFrame({"step": [1, 2, 3]})
    original = df.copy()
    add_time_columns(df)
    pd.testing.assert_frame_equal(df, original)


def test_add_time_columns_rejects_unsorted_rows():
    # every as-of feature assumes rows are in time order, so unsorted input must fail loudly
    with pytest.raises(AssertionError):
        add_time_columns(pd.DataFrame({"step": [3, 1, 2]}))


# ---------- graph features ----------
def test_receiver_history_values(toy_tx):
    out = add_graph_features(toy_tx)
    assert out["dest_prior_in"].tolist() == [0, 0, 1, 2, 0]
    assert out["dest_prior_amt"].tolist() == [0.0, 0.0, 500.0, 540.0, 0.0]
    assert out["dest_age_steps"].tolist() == [0, 0, 1, 4, 0]


def test_chain_link_values(toy_tx):
    out = add_graph_features(toy_tx)
    assert out["chain_back"].tolist() == [0, 1, 0, 0, 1]     # a cash-out whose transfer is already in the log
    assert out["chain_fwd"].tolist() == [1, 0, 0, 1, 0]      # the look-ahead twin (a trap, never a feature)
    assert out["t_partner"].tolist() == [-1, 0, -1, -1, 3]


def test_add_graph_features_does_not_modify_input(toy_tx):
    original = toy_tx.copy()
    add_graph_features(toy_tx)
    pd.testing.assert_frame_equal(toy_tx, original)


# ---------- quirk guards ----------
def test_graph_has_no_leaky_or_look_ahead_columns():
    leaky = {"oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"}   # Trap 1
    look_ahead = {"dest_total_in", "chain_fwd", "traffic_so_far"}                                       # Traps 2 and 3
    assert set(GRAPH).isdisjoint(leaky | look_ahead)


def test_graph_is_the_given_model():
    # the task says the model stays unchanged: its feature list is frozen here
    assert GRAPH == ["type", "amount", "hour", "dest_prior_in", "dest_prior_amt", "dest_age_steps", "chain_back"]


# ---------- scoring ----------
def test_precision_at_k_value():
    y = [1, 0, 1, 0]
    score = [0.9, 0.8, 0.7, 0.1]
    assert precision_at_k(y, score, 2) == pytest.approx((0.5, 0.5))


# ---------- split ----------
def test_split_masks_partition_the_scored_rows(split_df):
    sc, tr, va, te, keep, trs = make_split(split_df, LAST_STEP, TRAIN_END, VALID_END, SCORED_TYPES, 0.1, 3217)
    assert ((tr.astype(int) + va.astype(int) + te.astype(int)) == 1).all()   # every row in exactly one period


def test_split_is_forward_in_time(split_df):
    sc, tr, va, te, keep, trs = make_split(split_df, LAST_STEP, TRAIN_END, VALID_END, SCORED_TYPES, 0.1, 3217)
    assert sc.step[tr].max() < sc.step[va].min()
    assert sc.step[va].max() < sc.step[te].min()


def test_split_keeps_every_training_fraud_and_is_seeded(split_df):
    a = make_split(split_df, LAST_STEP, TRAIN_END, VALID_END, SCORED_TYPES, 0.1, 3217)
    b = make_split(split_df, LAST_STEP, TRAIN_END, VALID_END, SCORED_TYPES, 0.1, 3217)
    sc, tr, va, te, keep, trs = a
    assert trs[tr & (sc.isFraud.values == 1)].all()          # no training fraud is thrown away
    assert (trs & ~tr).sum() == 0                            # sampled rows come only from the training period
    assert np.array_equal(a[4], b[4])                        # same seed, same sample


def test_split_drops_unscored_types_and_the_tail(split_df):
    sc, tr, va, te, keep, trs = make_split(split_df, LAST_STEP, TRAIN_END, VALID_END, SCORED_TYPES, 0.1, 3217)
    assert set(sc.type.unique()) <= set(SCORED_TYPES)
    assert "PAYMENT" not in sc.type.cat.categories
    assert sc.step.max() <= LAST_STEP                        # the fraud-only tail (719-743) is gone


# ---------- training ----------
def test_training_smoke_and_determinism(train_xy):
    X, y = train_xy
    a = train_model(X, y, seed=42, n_jobs=1).predict_proba(X)[:, 1]
    b = train_model(X, y, seed=42, n_jobs=1).predict_proba(X)[:, 1]
    assert len(a) == len(X)
    assert ((a >= 0) & (a <= 1)).all()
    assert a == pytest.approx(b)

def test_zero_amount_cash_out_has_no_chain_link():
    # PaySim has zero-amount rows; they must not link to a transfer by accident
    df = pd.DataFrame({"step": [1, 1], "type": ["TRANSFER", "CASH_OUT"], "dest": [1, 2], "amount": [50.0, 0.0]})
    out = add_graph_features(df)
    assert out["chain_back"].tolist() == [0, 0]
    assert np.isfinite(out["dest_prior_amt"].to_numpy()).all()