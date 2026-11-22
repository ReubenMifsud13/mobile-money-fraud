
import logging
import os
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from money_fraud.config import load_config
from money_fraud.logging_setup import setup_logging
from money_fraud.serving import ReceiverState, score


load_dotenv()
setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

logger = logging.getLogger(__name__)
config = load_config()

MODEL_PATH = os.environ.get(
    "MODEL_PATH", config["model_path"]
)
STATE_PATH = os.environ.get(
    "STATE_PATH", config["state_path"]
)
MODEL_VERSION = os.environ.get(
    "MODEL_VERSION", "dev"
)

model = joblib.load(MODEL_PATH)

state = ReceiverState.from_frame(
    pd.read_parquet(STATE_PATH)
)

logger.info(
    "Model loaded from %s; %d receivers in memory",
    MODEL_PATH,
    len(state.accounts),
)

app = FastAPI(
    title="Mobile-money fraud scoring"
)


class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT",
    ]
    amount: float = Field(ge=0, allow_inf_nan=False)
    dest: int = Field(ge=0)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "receivers": len(state.accounts),
    }


@app.post("/score")
def score_transaction(tx: Transaction):
    try:
        feats = state.observe(
            tx.step,
            tx.type,
            tx.amount,
            tx.dest,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    if tx.type not in config["scored_types"]:
        return {
            "scored": False,
            "probability_fraud": None,
            "model_version": MODEL_VERSION,
        }

    probability = score(
        model,
        feats,
        config["scored_types"],
    )

    logger.info(
        "step=%d type=%s amount=%.2f probability=%.4f",
        tx.step,
        tx.type,
        tx.amount,
        probability,
    )

    return {
        "scored": True,
        "probability_fraud": round(probability, 4),
        "model_version": MODEL_VERSION,
    }
