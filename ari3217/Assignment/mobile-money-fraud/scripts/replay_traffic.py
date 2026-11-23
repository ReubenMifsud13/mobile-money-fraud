
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yaml


URL = "http://localhost:8888/score"
TARGET_SCORED = 300


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    path = Path(cfg["processed_dir"]) / "05_paysim.parquet"
    tx = pd.read_parquet(path)

    holdout = tx[
        (tx.step > cfg["valid_end"])
        & (tx.step <= cfg["last_step"])
    ]

    requests_sent = 0
    scored = 0
    start = time.perf_counter()

    for row in holdout.itertuples(index=False):
        payload = {
            "step": int(row.step),
            "type": str(row.type),
            "amount": float(row.amount),
            "dest": int(row.dest),
        }

        request = Request(
            URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=30) as response:
            result = json.load(response)

        requests_sent += 1

        if result["scored"]:
            scored += 1

        if scored >= TARGET_SCORED:
            break

    elapsed = time.perf_counter() - start

    print("Requests sent:", requests_sent)
    print("Scored transactions:", scored)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Throughput: {requests_sent / elapsed:.2f} requests/s")


if __name__ == "__main__":
    main()
