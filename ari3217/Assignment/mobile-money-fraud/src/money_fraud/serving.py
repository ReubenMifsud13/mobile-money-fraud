
import threading

import numpy as np
import pandas as pd

from money_fraud.features import GRAPH


class ReceiverState:
    """Keeps the history required for fraud-scoring features."""

    def __init__(self, accounts=None):
        self.accounts = accounts if accounts is not None else {}
        self.step = None
        self.transfers = set()
        self._lock = threading.Lock()

    @classmethod
    def from_history(cls, df):
        """Build receiver memory from earlier transactions."""
        if df.empty:
            return cls()

        g = df.groupby("dest", sort=False)
        agg = pd.DataFrame({
            "count": g.size(),
            "total": g.amount.sum(),
            "first": g.step.min(),
        }).reset_index()

        state = cls.from_frame(agg)
        state.step = int(df.step.max())

        last = df[
            (df.step == state.step)
            & (df.type == "TRANSFER")
        ]
        state.transfers = set(
            np.round(last.amount.to_numpy() * 100)
            .astype(np.int64)
            .tolist()
        )
        return state

    @classmethod
    def from_frame(cls, df):
        """Restore receiver accounts from a saved table."""
        accounts = {}

        for row in df.itertuples(index=False):
            accounts[int(row.dest)] = [
                int(row.count),
                float(row.total),
                int(row.first),
            ]

        return cls(accounts)

    def to_frame(self):
        """Convert receiver memory into a pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "dest": dest,
                    "count": values[0],
                    "total": values[1],
                    "first": values[2],
                }
                for dest, values in self.accounts.items()
            ],
            columns=["dest", "count", "total", "first"],
        )

    def observe(self, step, type_, amount, dest):
        """Calculate as-of features, then update history."""
        with self._lock:
            if self.step is not None and step < self.step:
                raise ValueError(
                    f"step {step} is earlier than "
                    f"the current step {self.step}"
                )

            if step != self.step:
                self.step = step
                self.transfers = set()

            cents = int(np.round(amount * 100))
            acct = self.accounts.get(dest)

            if acct is None:
                n, total, first = 0, 0.0, step
            else:
                n, total, first = acct

            feats = {
                "type": type_,
                "amount": amount,
                "hour": (step - 1) % 24,
                "dest_prior_in": n,
                "dest_prior_amt": total,
                "dest_age_steps": step - first,
                "chain_back": int(
                    type_ == "CASH_OUT"
                    and cents in self.transfers
                ),
            }

            # Update only AFTER calculating the features.
            if acct is None:
                self.accounts[dest] = [1, amount, step]
            else:
                acct[0] += 1
                acct[1] += amount

            if type_ == "TRANSFER":
                self.transfers.add(cents)

            return feats


def score(model, feats, categories):
    """Calculate the fraud probability from seven features."""
    X = pd.DataFrame([feats], columns=GRAPH)
    X["type"] = pd.Categorical(
        X["type"],
        categories=categories,
    )
    return float(model.predict_proba(X)[0, 1])