# Mobile-money fraud scoring

Scores TRANSFER and CASH_OUT transactions (PaySim) for fraud risk. Packages the given model from the task notebook.

## Requirements
Python 3.12 with the venv module. On Ubuntu:
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
The model is saved to models/model.joblib. Shared settings are in config/config.yaml; settings for your machine go in .env.