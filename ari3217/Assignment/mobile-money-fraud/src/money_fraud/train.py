import logging
import os
import sys

from dotenv import load_dotenv

from money_fraud.config import load_config
from money_fraud.data import load_paysim
from money_fraud.features import GRAPH, add_graph_features, add_time_columns
from money_fraud.logging_setup import setup_logging
from money_fraud.model import fit_eval, save_model
from money_fraud.split import make_split

logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    try:
        config = load_config()
        raw_dir = os.environ.get("RAW_DIR", config["raw_dir"])
        processed_dir = os.environ.get("PROCESSED_DIR", config["processed_dir"])
        model_path = os.environ.get("MODEL_PATH", config["model_path"])

        tx = load_paysim(raw_dir, processed_dir, config["csv_file"], config["kaggle_handle"])
        logger.info("Data loaded: %d rows", len(tx))
        tx = add_graph_features(add_time_columns(tx))
        logger.info("Features added")
        sc, tr, va, te, keep, trs = make_split(tx, config["last_step"], config["train_end"], config["valid_end"],
                                               config["scored_types"], config["neg_sample"], config["seed"])
        logger.info("Split: train %d (sampled %d), valid %d, holdout %d", tr.sum(), trs.sum(), va.sum(), te.sum())
        k = round(config["review_budget_per_day"] * (config["last_step"] - config["valid_end"]) / 24)
        model, p, result = fit_eval(sc, trs, te, GRAPH, "+ both (who pays whom)", k, config["seed"], config["n_jobs"])
        logger.info("Result: %s", result)
        save_model(model, model_path)
        logger.info("Model saved to %s", model_path)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()