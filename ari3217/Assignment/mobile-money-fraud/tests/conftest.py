import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_tx():
    # two fraud-shaped pairs (rows 0-1 and 3-4) and one lone transfer (row 2); dest 10 is hit three times
    return pd.DataFrame({
        "step": [1, 1, 2, 5, 5],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "TRANSFER", "CASH_OUT"],
        "dest": [10, 20, 10, 10, 30],
        "amount": [500.0, 500.0, 40.0, 70.0, 70.0],
    })


@pytest.fixture
def split_df():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "step": np.sort(rng.integers(1, 744, n)),       # in time order, includes the fraud-only tail (> 718)
        "type": pd.Categorical(rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)),
        "isFraud": rng.integers(0, 2, n),
    })


@pytest.fixture
def train_xy():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = (X["a"] + rng.normal(scale=0.5, size=500) > 0).astype(int)
    return X, y