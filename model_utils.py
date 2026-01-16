"""
Utilities for training, walk-forward backtest, and metrics.
"""
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


def ensure_dirs(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)


def train_xgb_classifier(X, y, params: dict = None):
    params = params or {}
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.03),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        objective=params.get("objective", "binary:logistic"),
        eval_metric=params.get("eval_metric", "logloss"),
        use_label_encoder=False,
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
    )
    model.fit(X, y)
    return model


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_years: int = 5,
    test_step_days: int = 21,
    model_params: dict = None,
    threshold: float = 0.5,
    outputs_path: str = "outputs",
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform walk-forward backtest on dataframe indexed by date.

    Returns backtest dataframe (per-test-sample) and summary metrics.
    """
    ensure_dirs(outputs_path)

    dates = df.index
    start_date = dates.min()
    end_date = dates.max()

    train_window = pd.DateOffset(years=train_years)
    step = pd.Timedelta(days=test_step_days)

    records = []

    window_id = 0
    cur_train_start = start_date
    cur_train_end = cur_train_start + train_window

    last_valid_end = end_date - pd.Timedelta(days=0)

    while cur_train_end + step <= end_date:
        window_id += 1

        train_mask = (df.index >= cur_train_start) & (df.index <= cur_train_end)
        test_mask = (df.index > cur_train_end) & (df.index <= cur_train_end + step)

        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        if len(train_df) < 50 or test_df.empty:
            # Move window forward
            cur_train_start = cur_train_start + step
            cur_train_end = cur_train_start + train_window
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        # Train model
        model = train_xgb_classifier(X_train, y_train, params=model_params)

        # Predict probabilities
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        # Metrics for this window
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        try:
            roc = roc_auc_score(y_test, probs)
        except Exception:
            roc = float("nan")

        # Save per-sample records
        for i, idx in enumerate(test_df.index):
            records.append(
                {
                    "date": idx,
                    "window_id": window_id,
                    "train_start": cur_train_start,
                    "train_end": cur_train_end,
                    "prob": float(probs[i]),
                    "pred": int(preds[i]),
                    "actual": int(y_test[i]),
                }
            )

        # Advance window
        cur_train_start = cur_train_start + step
        cur_train_end = cur_train_start + train_window

    backtest_df = pd.DataFrame(records).set_index("date").sort_index()

    if backtest_df.empty:
        raise RuntimeError("Backtest produced no predictions. Check windows and data range.")

    # Aggregate metrics
    overall_acc = accuracy_score(backtest_df["actual"], backtest_df["pred"])
    overall_f1 = f1_score(backtest_df["actual"], backtest_df["pred"], zero_division=0)
    overall_prec = precision_score(backtest_df["actual"], backtest_df["pred"], zero_division=0)
    overall_rec = recall_score(backtest_df["actual"], backtest_df["pred"], zero_division=0)
    try:
        overall_roc = roc_auc_score(backtest_df["actual"], backtest_df["prob"].values)
    except Exception:
        overall_roc = float("nan")

    cm = confusion_matrix(backtest_df["actual"], backtest_df["pred"])

    # Cumulative accuracy over time
    backtest_df["correct"] = (backtest_df["pred"] == backtest_df["actual"]).astype(int)
    backtest_df = backtest_df.sort_index()
    backtest_df["cumulative_accuracy"] = backtest_df["correct"].expanding().mean()

    summary = {
        "accuracy": float(overall_acc),
        "f1": float(overall_f1),
        "precision": float(overall_prec),
        "recall": float(overall_rec),
        "roc_auc": float(overall_roc) if not np.isnan(overall_roc) else None,
        "confusion_matrix": cm.tolist(),
        "n_predictions": len(backtest_df),
    }

    # Save backtest CSV
    csv_path = os.path.join(outputs_path, "backtest_results.csv")
    backtest_df.to_csv(csv_path)

    # Save last model as artifact (train on full dataset)
    try:
        final_X = df[feature_cols].values
        final_y = df[target_col].values
        final_model = train_xgb_classifier(final_X, final_y, params=model_params)
        model_path = os.path.join(outputs_path, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
    except Exception:
        final_model = None

    return backtest_df, summary
"""
Copper Brain v2 - Model Utilities
=================================
Model training, walk-forward evaluation, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def create_xgb_classifier(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8
) -> XGBClassifier:
    """
    Create XGBoost classifier with optimized parameters.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )

    return model


def calibrate_model_probability_threshold(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[XGBClassifier, float]:
    """
    Calibrate the model and find optimal probability threshold.

    Returns:
        Tuple of (calibrated_model, optimal_threshold)
    """
    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)

    # Get predicted probabilities on validation set
    y_val_prob = calibrated_model.predict_proba(X_val)[:, 1]

    # Try different thresholds
    thresholds = np.arange(0.35, 0.65, 0.01)
    best_accuracy = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_val_pred = (y_val_prob >= threshold).astype(int)
        accuracy = accuracy_score(y_val, y_val_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return calibrated_model, best_threshold


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_window_years: int = 5,
    test_step_days: int = 21,
    probability_threshold: float = 0.45
) -> pd.DataFrame:
    """
    Perform walk-forward rolling validation.

    Parameters:
    - df: Processed dataframe with features and target
    - feature_cols: List of feature column names
    - target_col: Target column name
    - train_window_years: Training window in years
    - test_step_days: Step size for testing in days
    - probability_threshold: Threshold for classification

    Returns:
    - DataFrame with backtest results
    """
    results = []

    # Sort dataframe by date
    df = df.sort_index()

    # Convert train_window_years to approximate trading days (252 per year)
    train_window_days = train_window_years * 252

    start_date = df.index.min()
    end_date = df.index.max()

    current_train_end = start_date + pd.DateOffset(days=train_window_days)

    while current_train_end < end_date:
        # Define training period
        train_start = current_train_end - pd.DateOffset(days=train_window_days)
        train_mask = (df.index >= train_start) & (df.index <= current_train_end)

        # Define test period
        test_start = current_train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(days=test_step_days)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        # Skip if insufficient data
        if not train_mask.sum() or not test_mask.sum():
            current_train_end = test_end
            continue

        # Split data
        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(train_data) < 100 or len(test_data) < 5:
            current_train_end = test_end
            continue

        # Prepare features and target
        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Train model
        model = create_xgb_classifier()
        model.fit(X_train, y_train)

        # Calibrate and get predictions
        calibrated_model, threshold = calibrate_model_probability_threshold(
            model, X_train, y_train, X_train, y_train
        )

        # Use provided threshold if specified
        if probability_threshold != 0.45:
            threshold = probability_threshold

        # Get predictions
        y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_pred_prob)
        except:
            roc_auc = np.nan

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Store results for each test sample
        for i, (idx, row) in enumerate(test_data.iterrows()):
            results.append({
                'date': idx,
                'train_start': train_start,
                'train_end': current_train_end,
                'test_start': test_start,
                'test_end': test_end,
                'actual_direction': y_test[i],
                'predicted_direction': y_pred[i],
                'prediction_probability': y_pred_prob[i],
                'threshold_used': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'close_price': row['close'],
                'future_return_21d': row.get('future_return_21d', np.nan)
            })

        # Move window forward
        current_train_end = test_end

    return pd.DataFrame(results)


def calculate_overall_metrics(backtest_df: pd.DataFrame) -> Dict:
    """
    Calculate overall performance metrics from backtest results.
    """
    if backtest_df.empty:
        return {}

    # Overall accuracy
    overall_accuracy = accuracy_score(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction']
    )

    # Overall precision, recall, f1
    overall_precision = precision_score(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction'],
        zero_division=0
    )

    overall_recall = recall_score(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction'],
        zero_division=0
    )

    overall_f1 = f1_score(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction'],
        zero_division=0
    )

    # ROC-AUC
    try:
        overall_roc_auc = roc_auc_score(
            backtest_df['actual_direction'],
            backtest_df['prediction_probability']
        )
    except:
        overall_roc_auc = np.nan

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction']
    ).ravel()

    # Rolling accuracy (6-month windows)
    backtest_df = backtest_df.sort_values('date')
    backtest_df['rolling_accuracy_126d'] = (
        backtest_df.set_index('date')['predicted_direction']
        .rolling('126D')
        .corr(backtest_df.set_index('date')['actual_direction'])
    )

    return {
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'overall_roc_auc': overall_roc_auc,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'total_predictions': len(backtest_df),
        'positive_predictions': (backtest_df['predicted_direction'] == 1).sum(),
        'negative_predictions': (backtest_df['predicted_direction'] == 0).sum(),
        'actual_positives': (backtest_df['actual_direction'] == 1).sum(),
        'actual_negatives': (backtest_df['actual_direction'] == 0).sum()
    }


def save_backtest_results(backtest_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Save backtest results to CSV and generate summary plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full backtest results
    csv_path = os.path.join(output_dir, "backtest_results.csv")
    backtest_df.to_csv(csv_path, index=False)
    print(f"Saved backtest results to {csv_path}")

    # Generate summary plots
    create_backtest_plots(backtest_df, output_dir)


def create_backtest_plots(backtest_df: pd.DataFrame, output_dir: str):
    """
    Create and save summary plots for backtest results.
    """
    if backtest_df.empty:
        return

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Cumulative accuracy over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Cumulative accuracy
    backtest_df = backtest_df.sort_values('date')
    backtest_df['cumulative_correct'] = (
        backtest_df['actual_direction'] == backtest_df['predicted_direction']
    ).cumsum()
    backtest_df['cumulative_total'] = range(1, len(backtest_df) + 1)
    backtest_df['cumulative_accuracy'] = (
        backtest_df['cumulative_correct'] / backtest_df['cumulative_total']
    )

    axes[0, 0].plot(backtest_df['date'], backtest_df['cumulative_accuracy'])
    axes[0, 0].set_title('Cumulative Directional Accuracy Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Prediction probability distribution
    correct_predictions = backtest_df[backtest_df['actual_direction'] == backtest_df['predicted_direction']]
    incorrect_predictions = backtest_df[backtest_df['actual_direction'] != backtest_df['predicted_direction']]

    axes[0, 1].hist(
        correct_predictions['prediction_probability'],
        alpha=0.7, label='Correct', bins=20, density=True
    )
    axes[0, 1].hist(
        incorrect_predictions['prediction_probability'],
        alpha=0.7, label='Incorrect', bins=20, density=True
    )
    axes[0, 1].set_title('Prediction Probability Distribution')
    axes[0, 1].set_xlabel('Prediction Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Rolling accuracy heatmap (monthly)
    backtest_df['year_month'] = backtest_df['date'].dt.to_period('M')
    monthly_accuracy = backtest_df.groupby('year_month').apply(
        lambda x: accuracy_score(x['actual_direction'], x['predicted_direction'])
        if len(x) > 0 else np.nan
    )

    # Create pivot table for heatmap
    monthly_accuracy_df = monthly_accuracy.reset_index()
    monthly_accuracy_df['year'] = monthly_accuracy_df['year_month'].dt.year
    monthly_accuracy_df['month'] = monthly_accuracy_df['year_month'].dt.month

    pivot_accuracy = monthly_accuracy_df.pivot(
        index='year', columns='month', values=0
    )

    sns.heatmap(
        pivot_accuracy,
        ax=axes[1, 0],
        cmap='RdYlGn',
        vmin=0.4,
        vmax=0.8,
        cbar_kws={'label': 'Directional Accuracy'}
    )
    axes[1, 0].set_title('Monthly Directional Accuracy Heatmap')

    # 4. Confusion matrix
    cm = confusion_matrix(
        backtest_df['actual_direction'],
        backtest_df['predicted_direction']
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        ax=axes[1, 1],
        cmap='Blues',
        xticklabels=['Predicted Down', 'Predicted Up'],
        yticklabels=['Actual Down', 'Actual Up']
    )
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_xlabel('Predicted')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "backtest_summary_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved backtest summary plots to {plot_path}")


def train_final_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    probability_threshold: float = 0.45
) -> Tuple[XGBClassifier, float]:
    """
    Train final model on all available data for production use.

    Returns:
        Tuple of (trained_model, probability_threshold)
    """
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values

    # Train model
    model = create_xgb_classifier()
    model.fit(X, y)

    # Calibrate model
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated_model.fit(X, y)

    return calibrated_model, probability_threshold