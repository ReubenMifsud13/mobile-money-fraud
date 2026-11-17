import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# hardcoded for now; Stage 3 moves these into config
KAGGLE_HANDLE = "ealaxi/paysim1"
CSV_FILE = "PS_20174392719_1491204439457_log.csv"
RAW = Path("data/raw/paysim")
PROCESSED = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


def get_raw(filename: str) -> Path:
    local = RAW / filename
    if local.exists():
        return local
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=filename))


COLUMN_TYPES = {"step": pa.int16(), "type": pa.dictionary(pa.int32(), pa.string()), "amount": pa.float64(),
                "nameOrig": pa.string(), "oldbalanceOrg": pa.float64(), "newbalanceOrig": pa.float64(),
                "nameDest": pa.string(), "oldbalanceDest": pa.float64(), "newbalanceDest": pa.float64(),
                "isFraud": pa.int8(), "isFlaggedFraud": pa.int8()}
BALANCES = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def load_paysim() -> pd.DataFrame:
    cache = PROCESSED / "05_paysim.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = time.time()
    tbl = pacsv.read_csv(get_raw(CSV_FILE), convert_options=pacsv.ConvertOptions(column_types=COLUMN_TYPES))
    # ---- governance on load: account names become integer surrogate keys; the names never leave this function ----
    def account_key(col):
        name = tbl[col]
        is_merchant = np.asarray(pc.equal(pc.utf8_slice_codeunits(name, 0, 1), "M"))
        number = np.asarray(pc.cast(pc.utf8_slice_codeunits(name, 1), pa.int64()))
        return number * 2 + is_merchant, is_merchant
    k_orig, m_orig = account_key("nameOrig")
    k_dest, m_dest = account_key("nameDest")
    assert not m_orig.any(), "a merchant appears as a sender"
    codes, _ = pd.factorize(np.concatenate([k_orig, k_dest]))   # one key space for senders and receivers
    n = tbl.num_rows
    df = pd.DataFrame({"step": np.asarray(tbl["step"]), "type": tbl["type"].to_pandas().astype("category"),
                       "amount": np.asarray(tbl["amount"]),
                       "orig": codes[:n].astype(np.int32), "dest": codes[n:].astype(np.int32),
                       "dest_is_merchant": m_dest,
                       **{c: np.asarray(tbl[c]) for c in BALANCES},
                       "isFraud": np.asarray(tbl["isFraud"]), "isFlaggedFraud": np.asarray(tbl["isFlaggedFraud"])})
    del tbl, k_orig, k_dest, codes
    df.to_parquet(cache)
    print(f"loaded + cached in {time.time()-t:.0f}s")
    return df