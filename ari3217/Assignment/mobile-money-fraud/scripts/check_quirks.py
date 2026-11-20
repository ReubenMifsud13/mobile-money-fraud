import yaml

from money_fraud.data import load_paysim

cfg = yaml.safe_load(open("config/config.yaml"))
tx = load_paysim(cfg["raw_dir"], cfg["processed_dir"], cfg["csv_file"], cfg["kaggle_handle"])
fraud = tx[tx.isFraud == 1]

print("Q2 frauds:", len(fraud), "| by type:\n", fraud.type.value_counts())
print("Q3 rows after step 718:", (tx.step > 718).sum(), "| fraud share there:", tx[tx.step > 718].isFraud.mean())
print("Q6 isFlaggedFraud fires:", tx.isFlaggedFraud.sum(), "| all fraud:", tx[tx.isFlaggedFraud == 1].isFraud.all())
print("Q6 documented rule (TRANSFER > 200000):", ((tx.type == "TRANSFER") & (tx.amount > 200000)).sum())
print("Q7 distinct steps:", tx.step.nunique(), "| max step:", tx.step.max())
print("Q7 largest fraud amount:", fraud.amount.max())

# candidates for the quirk the card does NOT list
print("zero-amount rows:", (tx.amount == 0).sum()) # Fired positive - quirk
print("sender == receiver:", (tx.orig == tx.dest).sum()) # 0 - not a quirk