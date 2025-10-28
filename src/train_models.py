import argparse, os, yaml, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_scaler(name):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    return "passthrough"

def make_selector(cfg, X, y):
    sel = cfg.get("selector", {"method":"none"})
    m = sel.get("method","none")
    if m == "kbest":
        k = int(sel.get("k", 50))
        return SelectKBest(score_func=f_classif, k=min(k, X.shape[1]-1))
    if m == "rfe":
        est = LogisticRegression(max_iter=1000)
        return RFE(estimator=est, n_features_to_select=min(int(sel.get("k", 50)), X.shape[1]-1))
    return "passthrough"

def make_model(m):
    if m["type"] == "svm":
        p = m["params"]
        return SVC(probability=True, **p)
    if m["type"] == "rf":
        p = m["params"]
        return RandomForestClassifier(**p)
    raise ValueError(m["type"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--reports_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    cfg = load_cfg(args.config)
    rs = int(cfg.get("random_state", 42))
    df = pd.read_csv(args.features)
    X = df.drop(columns=["label","id"]).values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=int(cfg["cv"]["folds"]), shuffle=bool(cfg["cv"]["shuffle"]), random_state=rs)

    cfg_metrics = cfg.get("metrics", ["accuracy", "precision", "recall", "f1"])  # bierzemy z YAML
scorers = {}
for name in cfg_metrics:
    if name == "roc_auc":
        # najprostsza i najstabilniejsza forma
        scorers[name] = "roc_auc"
    else:
        scorers[name] = get_scorer(name)

    results = []
    for m in cfg["models"]:
        scaler = make_scaler(cfg.get("scaler", "standard"))
        selector = make_selector(cfg, X, y)
        model = make_model(m)
        pipe = Pipeline([("scaler", scaler), ("selector", selector), ("model", model)])
       cvres = cross_validate(pipe, X, y, cv=skf, scoring=scorers, return_estimator=True)
        row = {"model": m["name"]}
        for k, v in cvres.items():
            if k.startswith("test_"):
                row[k] = float(np.mean(v))
        results.append(row)
        # save best estimator by F1
        best_idx = int(np.argmax(cvres["test_f1"]))
        joblib.dump(cvres["estimator"][best_idx], os.path.join(args.out_dir, f"{m['name']}.pkl"))
    pd.DataFrame(results).to_csv(os.path.join(args.reports_dir, "metrics.csv"), index=False)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
