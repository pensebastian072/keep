"""
Threshold tuning and SHAP generation script.
Loads backtest probabilities, scans thresholds, saves tuning results and SHAP summary plot.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = os.path.dirname(__file__)
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
BACKTEST_CSV = os.path.join(OUTPUTS, "backtest_results.csv")
MODEL_PKL = os.path.join(OUTPUTS, "model.pkl")


def run_threshold_scan():
    df = pd.read_csv(BACKTEST_CSV, parse_dates=["date"], index_col="date")
    thresholds = np.arange(0.10, 0.901, 0.01)
    rows = []
    for thr in thresholds:
        preds = (df["prob"] >= thr).astype(int)
        acc = accuracy_score(df["actual"], preds)
        prec = precision_score(df["actual"], preds, zero_division=0)
        rec = recall_score(df["actual"], preds, zero_division=0)
        f1 = f1_score(df["actual"], preds, zero_division=0)
        cm = confusion_matrix(df["actual"], preds)
        rows.append({"threshold": float(thr), "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUTS, "threshold_tuning.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved threshold tuning results to {out_path}")

    best = out.loc[out["accuracy"].idxmax()]
    print("Best threshold by accuracy:")
    print(best.to_dict())

    # Save best threshold
    with open(os.path.join(OUTPUTS, "best_threshold.txt"), "w") as f:
        f.write(str(best["threshold"]))

    return out, float(best["threshold"])


def generate_shap_plot(sample_frac=0.05, out_png=None):
    # Recompute features from data if necessary
    # We'll load the saved model and compute SHAP on a sample of training-like data.
    if not os.path.exists(MODEL_PKL):
        print("No model file found for SHAP generation.")
        return

    # Attempt to find engineered data file. We'll try backtest to deduce dates, and reconstruct features from data folder.
    # For simplicity, we'll load the entire engineered dataset by reusing features.py pipeline if available.
    try:
        from features import engineer_all_features
        # load raw csv files saved during training
        data_dir = os.path.join(PROJECT_ROOT, "data")
        copper_csv = os.path.join(data_dir, "copper_raw.csv")
        dxy_csv = os.path.join(data_dir, "dxy_raw.csv")
        gold_csv = os.path.join(data_dir, "gold_raw.csv")
        copper = pd.read_csv(copper_csv, parse_dates=[0], index_col=0)
        dxy = pd.read_csv(dxy_csv, parse_dates=[0], index_col=0)
        gold = pd.read_csv(gold_csv, parse_dates=[0], index_col=0)

        # Ensure column names for features pipeline
        copper.columns = [c if c in ["close","Open","High","Low","Volume"] else c for c in copper.columns]

        df_full, feature_cols = engineer_all_features(copper, dxy, gold, horizon_days=21)

        if df_full.empty:
            print("Engineered dataframe empty; skipping SHAP.")
            return

        # Load model
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)

        X = df_full[feature_cols]
        sample = X.sample(frac=sample_frac, random_state=42)

        explainer = shap.Explainer(model)
        shap_values = explainer(sample)

        out_png = out_png or os.path.join(OUTPUTS, "shap_summary.png")
        plt.figure(figsize=(10, 6))
        try:
            shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP summary plot to {out_png}")
        except Exception as e:
            print("Failed to render SHAP beeswarm:", e)

    except Exception as e:
        print("SHAP generation failed:", e)


if __name__ == "__main__":
    tuning_df, best_thr = run_threshold_scan()
    generate_shap_plot(sample_frac=0.05)
    print("Done.")
