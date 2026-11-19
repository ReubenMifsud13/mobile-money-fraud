from money_fraud.data import load_paysim
from money_fraud.features import GRAPH, add_graph_features, add_time_columns
from money_fraud.model import fit_eval, save_model
from money_fraud.split import make_split

# hardcoded for now
MODEL_PATH = "models/model.joblib"


def main():
    tx = load_paysim()
    print("Data loaded:", len(tx), "rows")
    tx = add_graph_features(add_time_columns(tx))
    print("Features added")
    sc, tr, va, te, keep, trs = make_split(tx)
    print("Split done")
    model, p, result = fit_eval(sc, trs, te, GRAPH, "+ both (who pays whom)")
    print(result)
    save_model(model, MODEL_PATH)
    print("Model saved in", MODEL_PATH)


if __name__ == "__main__":
    main()