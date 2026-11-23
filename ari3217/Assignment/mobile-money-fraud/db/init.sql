
-- Creates the table when PostgreSQL first starts.

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    step INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    dest BIGINT NOT NULL,
    probability DOUBLE PRECISION,
    model_version TEXT NOT NULL
);