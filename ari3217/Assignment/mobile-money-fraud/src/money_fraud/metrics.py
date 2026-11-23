
from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "predictions",
    "Number of fraud predictions served",
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to process a scoring request",
)

INPUTS = Histogram(
    "prediction_input",
    "Transaction amounts received",
    buckets=[
        100, 500, 1000, 5000, 10000,
        50000, 100000, 500000,
        1000000, 5000000, 10000000,
    ],
)
