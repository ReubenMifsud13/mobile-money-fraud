# Mobile-money fraud scoring

Scores TRANSFER and CASH_OUT transactions (PaySim) for fraud risk. Packages the given model from the task notebook.

## Requirements
Python 3.12 with the venv module, and Docker with Docker Compose. On Ubuntu:
```bash
sudo apt install python3 python3-venv
```
The PaySim CSV is downloaded from Kaggle on first use (kagglehub), or place it in data/raw/paysim/.

## Install
Run these from this folder, the one with pyproject.toml:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e ".[dev]"
cp env.example .env
```

## Test
```bash
pytest
```

## Train
```bash
money-train
```
The model and the receiver memory are saved to models/. Shared settings are in config/config.yaml; settings for your machine go in .env.

## Run the API
Build the image and start the API with PostgreSQL, Prometheus and Grafana:
```bash
docker build -t money-fraud:0.1 .
docker compose up --build
```
Check that the API is running:
```bash
curl http://localhost:8000/health
```
Stop everything:
```bash
docker compose down
```
Do not put models/ in .dockerignore: the image needs the saved files.
