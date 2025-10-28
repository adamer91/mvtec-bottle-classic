import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", required=True)
    args = ap.parse_args()

    metrics = pd.read_csv(os.path.join(args.reports_dir, "metrics.csv"))
    print(metrics)

    # wybierz metryki, które naprawdę są w pliku
    test_cols = [c for c in metrics.columns if c.startswith("test_")]
    metric_names = [c.replace("test_", "") for c in test_cols]

    for metric in metric_names:
        series = metrics[f"test_{metric}"]
        if series.isna().all():
            continue  # pomijamy metryki z samymi NaN
        plt.figure()
        metrics.plot(x="model", y=f"test_{metric}", kind="bar", legend=False)
        plt.ylabel(metric.upper())
        plt.title(f"Cross-validated {metric.upper()} by model")
        plt.tight_layout()
        plt.savefig(os.path.join(args.reports_dir, f"{metric}_bar.png"))

if __name__ == "__main__":
    main()
