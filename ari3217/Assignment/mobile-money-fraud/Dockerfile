FROM python:3.12-slim

# LightGBM requires the OpenMP runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install our Python package
COPY pyproject.toml .
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir --no-deps .

# Copy the already trained model and receiver history
COPY models/model.joblib ./models/model.joblib
COPY models/receiver_state.parquet ./models/receiver_state.parquet

# API port
EXPOSE 8888

# Start the API when the container runs
CMD ["uvicorn", "money_fraud.api:app", "--host", "0.0.0.0", "--port", "8888"]