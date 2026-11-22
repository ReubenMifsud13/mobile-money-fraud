
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from money_fraud.config import load_config
from money_fraud.data import load_paysim
from money_fraud.features import GRAPH, add_graph_features, add_time_columns
from money_fraud.logging_setup import setup_logging
from money_fraud.model import fit_eval, save_model
from money_fraud.split import make_split
from money_fraud.serving import ReceiverState
from money_fraud.seed import set_seed


logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    try:
        # Load configuration
        config = load_config()
        set_seed(config["seed"])

        raw_dir = os.environ.get("RAW_DIR", config["raw_dir"])
        processed_dir = os.environ.get(
            "PROCESSED_DIR", config["processed_dir"]
        )
        model_path = os.environ.get(
            "MODEL_PATH", config["model_path"]
        )
        state_path = os.environ.get(
            "STATE_PATH", config["state_path"]
        )

        # Load the PaySim dataset
        tx = load_paysim(
            raw_dir,
            processed_dir,
            config["csv_file"],
            config["kaggle_handle"],
        )
        logger.info("Data loaded: %d rows", len(tx))

        # Calculate the original notebook features
        tx = add_graph_features(add_time_columns(tx))
        logger.info("Features added")

        # Split data chronologically
        sc, tr, va, te, keep, trs = make_split(
            tx,
            config["last_step"],
            config["train_end"],
            config["valid_end"],
            config["scored_types"],
            config["neg_sample"],
            config["seed"],
        )

        logger.info(
            "Split: train %d (sampled %d), valid %d, holdout %d",
            tr.sum(),
            trs.sum(),
            va.sum(),
            te.sum(),
        )

        # Calculate the review budget for the holdout
        k = round(
            config["review_budget_per_day"]
            * (config["last_step"] - config["valid_end"])
            / 24
        )

        # Train and evaluate the original LightGBM model
        model, p, result = fit_eval(
            sc,
            trs,
            te,
            GRAPH,
            "+ both (who pays whom)",
            k,
            config["seed"],
            config["n_jobs"],
        )

        logger.info("Result: %s", result)

        # Save the trained model
        save_model(model, model_path)
        logger.info("Model saved to %s", model_path)

        # Build receiver memory through the validation period
        history = tx[tx.step <= config["valid_end"]]
        state = ReceiverState.from_history(history)

        # Create the destination folder if necessary
        Path(state_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save the receiver history as Parquet
        state.to_frame().to_parquet(
            state_path,
            index=False,
        )

        logger.info(
            "Receiver memory saved to %s (%d receivers)",
            state_path,
            len(state.accounts),
        )

    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
