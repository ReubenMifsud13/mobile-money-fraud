
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from money_fraud.features import (
    GRAPH,
    add_graph_features,
    add_time_columns,
)
from money_fraud.serving import ReceiverState, score


def main():
    # Read the project configuration.
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load the existing processed PaySim dataset.
    cache = Path(cfg["processed_dir"]) / "05_paysim.parquet"

    if not cache.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {cache}"
        )

    tx = pd.read_parquet(cache)

    # The original notebook requires chronological rows.
    assert tx.step.is_monotonic_increasing

    print("Dataset loaded:", len(tx), "rows")

    # Calculate features using the original batch method.
    batch = add_graph_features(add_time_columns(tx))

    # Build history up to the end of validation.
    history = batch[
        batch.step <= cfg["valid_end"]
    ]

    state = ReceiverState.from_history(history)

    # Replay the holdout in chronological order.
    live = batch[
        (batch.step > cfg["valid_end"])
        & (batch.step <= cfg["last_step"])
    ]

    served = pd.DataFrame(
        [
            state.observe(
                int(row.step),
                str(row.type),
                float(row.amount),
                int(row.dest),
            )
            for row in live.itertuples()
        ],
        index=live.index,
    )

    # Only TRANSFER and CASH_OUT are scored.
    mask = live.type.isin(cfg["scored_types"])

    print("Holdout rows replayed:", len(live))
    print("Scored transactions:", int(mask.sum()))

    # 6. Compare batch and serving features.
    print("\nMaximum feature differences:")

    for col in GRAPH:
        if col == "type":
            assert (
                served.loc[mask, col].astype(str).to_numpy()
                == live.loc[mask, col].astype(str).to_numpy()
            ).all()
            print("type: matched")
            continue

        a = served.loc[mask, col].to_numpy()
        b = live.loc[mask, col].to_numpy()

        difference = float(np.max(np.abs(a - b)))

        print(f"{col}: {difference:.12g}")

        tolerance = (
            1e-5 if col == "dest_prior_amt" else 0
        )

        assert difference <= tolerance, (
            f"Mismatch in {col}: {difference}"
        )

    # 7. Compare actual model predictions.
    model = joblib.load(cfg["model_path"])

    indices = live.index[mask][:2000]

    batch_input = live.loc[indices, GRAPH].copy()
    batch_input["type"] = pd.Categorical(
        batch_input["type"],
        categories=cfg["scored_types"],
    )

    batch_predictions = model.predict_proba(
        batch_input
    )[:, 1]

    serving_predictions = np.array(
        [
            score(
                model,
                served.loc[i].to_dict(),
                cfg["scored_types"],
            )
            for i in indices
        ]
    )

    difference = float(
        np.max(
            np.abs(
                batch_predictions - serving_predictions
            )
        )
    )

    print("\nMaximum prediction difference:", difference)

    assert difference < 1e-7

    print("\nPARITY CHECK PASSED")


if __name__ == "__main__":
    main()
