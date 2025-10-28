import argparse
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import get_scorer


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_scaler(name):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    return "passthrough"


def make_selector(cfg, X):
    sel = cfg.get("selector", {"method": "none"})
    if sel.get("method") == "kbest":
        k = min(int(sel.get("k", 50)), X.shape[1] - 1)
        return SelectKBest(score_func=f_classif, k=max(1, k))
    return "passthrough"


def make_model(m):
    if m["type"] == "svm":
        return SVC(probability=True, **m["params"])
    if m["type"] == "rf":
        return RandomForestClassifier(**m["params"])
    raise ValueError(f"Unknown model type: {m['type']}")


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
    df = pd.read_csv(args.features)

    X = df.drop(columns=["label", "id"]).values
    y = df["label"].values
    skf = StratifiedKFold(
        n_splits=cfg["cv"]["folds"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg.get("random_state", 42),
    )

    cfg_metrics = cfg.get("metrics", ["accuracy", "precision", "recall", "f1"])
    scorers = {m: m for m in cfg_metrics}

    results = []

    for m in cfg["models"]:
        pipe = Pipeline([
            ("scaler", make_scaler(cfg.get("scaler", "standard"))),
            ("selector", make_selector(cfg, X)),
            ("model", make_model(m)),
        ])

        cvres = cross_validate(
            pipe, X, y, cv=skf, scoring=scorers,
            return_estimator=True
        )

        row = {"model": m["name"]}
        for metric in cfg_metrics:
            row[f"test_{metric}"] = float(np.mean(cvres[f"test_{metric}"]))

        results.append(row)

        # model o najlepszym F1
        key = "test_f1" if "test_f1" in cvres else f"test_{cfg_metrics[0]}"
        best_idx = int(np.argmax(cvres[key]))
        joblib.dump(
            cvres["estimator"][best_idx],
            os.path.join(args.out_dir, f"{m['name']}.pkl")
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.reports_dir, "metrics.csv"), index=False)
    print(results_df)


if __name__ == "__main__":
    main()
