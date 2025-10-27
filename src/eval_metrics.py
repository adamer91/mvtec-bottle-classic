import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", required=True)
    args = ap.parse_args()

    metrics = pd.read_csv(os.path.join(args.reports_dir, "metrics.csv"))
    print(metrics)

    # simple bar plot for F1 and ROC-AUC
    for metric in ["f1","roc_auc"]:
        plt.figure()
        metrics.plot(x="model", y=f"test_{metric}", kind="bar", legend=False)
        plt.ylabel(metric.upper())
        plt.title(f"Cross-validated {metric.upper()} by model")
        plt.tight_layout()
        plt.savefig(os.path.join(args.reports_dir, f"{metric}_bar.png"))

if __name__ == "__main__":
    main()
