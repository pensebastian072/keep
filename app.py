"""
Streamlit dashboard for Copper Brain v2
"""
import os
import subprocess
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

OUTPUTS_DIR = "outputs"
BACKTEST_CSV = os.path.join(OUTPUTS_DIR, "backtest_results.csv")
MODEL_PKL = os.path.join(OUTPUTS_DIR, "model.pkl")


st.set_page_config(page_title="Copper Brain v2", layout="wide")


@st.cache_data(ttl=600)
def load_backtest():
    if not os.path.exists(BACKTEST_CSV):
        return None
    df = pd.read_csv(BACKTEST_CSV, parse_dates=["date"], index_col="date")
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PKL):
        return None
    import pickle
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    return model


def retrain_model():
    # Run the training script
    proc = subprocess.Popen(["python", "copper_brain_v2.py"], cwd=os.getcwd())
    return proc


def main():
    st.title("Copper Brain v2 — Directional Forecast Explorer")

    # Sidebar controls
    st.sidebar.header("Settings")
    horizon = st.sidebar.selectbox("Horizon", [7, 21, 42], index=1)
    threshold = st.sidebar.slider("Probability threshold", 0.1, 0.9, 0.45, 0.01)
    retrain = st.sidebar.button("Retrain model")

    if retrain:
        with st.spinner("Retraining model (this may take a few minutes)..."):
            proc = retrain_model()
            proc.wait()
            st.success("Retrain finished.")

    backtest_df = load_backtest()
    model = load_model()

    st.header("Overview")
    st.markdown("Copper Brain v2 predicts 21-day ahead direction for COMEX copper futures using XGBoost and walk-forward backtesting.")

    if backtest_df is None:
        st.warning("No backtest results found. Run training first (click Retrain).")
        return

    # Metrics
    overall_acc = (backtest_df['pred'] == backtest_df['actual']).mean()
    overall_prec = (backtest_df['pred'][backtest_df['pred']==1] == backtest_df['actual'][backtest_df['pred']==1]).mean()
    overall_f1 = None
    try:
        from sklearn.metrics import f1_score
        overall_f1 = f1_score(backtest_df['actual'], backtest_df['pred'])
    except Exception:
        overall_f1 = None

    col1, col2, col3 = st.columns(3)
    col1.metric("Directional Accuracy", f"{overall_acc:.3f}")
    col2.metric("Precision", f"{overall_prec:.3f}")
    col3.metric("F1", f"{overall_f1:.3f}" if overall_f1 is not None else "n/a")

    st.header("Backtest Explorer")

    # Line chart: predicted vs actual
    df_plot = backtest_df.copy()
    df_plot['pred_dir'] = df_plot['pred']
    df_plot['actual_dir'] = df_plot['actual']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['prob'], name='Predicted probability'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['pred'], name='Predicted direction'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['actual'], name='Actual direction'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Probability Distribution")
    st.plotly_chart(px.histogram(backtest_df, x='prob', nbins=50), use_container_width=True)

    st.subheader("Confusion Matrix")
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(backtest_df['actual'], backtest_df['pred'])
        cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        st.write(cm_df)
    except Exception:
        st.write("Could not compute confusion matrix")

    st.header("Feature Importance & SHAP")
    if model is not None:
        try:
            importances = model.feature_importances_
            feat_names = model.get_booster().feature_names if hasattr(model, 'get_booster') else None
            if feat_names is None:
                # Attempt to read feature names from backtest if stored
                feat_names = [f"f{i}" for i in range(len(importances))]
            imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
            imp_df = imp_df.sort_values('importance', ascending=False).head(25)
            st.plotly_chart(px.bar(imp_df, x='importance', y='feature', orientation='h'), use_container_width=True)

            # SHAP (sampled)
            st.subheader("SHAP summary (sample)")
            # Create sample input for shap
            try:
                # Load training features from data if available
                import pickle
                with open(MODEL_PKL, 'rb') as f:
                    pass
            except Exception:
                pass
            # Attempt to compute SHAP values on a sample (may be slow)
            try:
                explainer = shap.Explainer(model)
                sample_X = backtest_df.drop(columns=['window_id','train_start','train_end','prob','pred','actual','correct','cumulative_accuracy'], errors='ignore').dropna().head(500)
                shap_vals = explainer(sample_X)
                st.pyplot(shap.plots.beeswarm(shap_vals, show=False))
            except Exception:
                st.info("SHAP not available or failed to compute.")
        except Exception as e:
            st.write("Could not load feature importances:", e)
    else:
        st.info("No trained model found in outputs/model.pkl")

    st.header("Recent Predictions")
    recent = backtest_df.tail(30).copy()
    recent_display = recent[['prob','pred','actual']].copy()
    recent_display.index = recent_display.index.strftime('%Y-%m-%d')
    st.dataframe(recent_display)

    st.header("Backtest Threshold Tuning")
    thr = st.slider("Adjust threshold to recompute predictions", 0.1, 0.9, float(threshold), 0.01)
    recomputed_preds = (backtest_df['prob'] >= thr).astype(int)
    new_acc = (recomputed_preds == backtest_df['actual']).mean()
    st.metric("Recomputed accuracy", f"{new_acc:.3f}")


if __name__ == '__main__':
    main()
"""
Copper Brain v2 - Interactive Dashboard
========================================
Comprehensive Streamlit dashboard for copper direction prediction analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import joblib

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from features import engineer_all_features
from model_utils import calculate_overall_metrics

# Page configuration
st.set_page_config(
    page_title="Copper Brain v2",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
    .success-metric {
        border-left-color: #28a745 !important;
    }
    .warning-metric {
        border-left-color: #ffc107 !important;
    }
    .danger-metric {
        border-left-color: #dc3545 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
#  DATA LOADING FUNCTIONS
# =========================

@st.cache_data(ttl=3600)
def load_backtest_results():
    """Load backtest results from CSV."""
    try:
        df = pd.read_csv("outputs/backtest_results.csv", parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error("Backtest results not found. Please run copper_brain_v2.py first.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_performance_metrics():
    """Load performance metrics from text file."""
    try:
        with open("outputs/performance_metrics.txt", 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "Performance metrics not found. Please run copper_brain_v2.py first."

@st.cache_resource
def load_final_model():
    """Load the trained model and feature columns."""
    try:
        model = joblib.load("outputs/final_model.pkl")
        feature_cols = joblib.load("outputs/feature_columns.pkl")
        return model, feature_cols
    except FileNotFoundError:
        return None, None

# =========================
#  DASHBOARD PAGES
# =========================

def overview_page():
    """Main overview page with key metrics."""
    st.markdown('<div class="main-header">🔮 Copper Brain v2</div>', unsafe_allow_html=True)
    st.markdown("*Production-Grade Copper Direction Predictor*")
    st.markdown("---")

    # Load data
    backtest_df = load_backtest_results()
    metrics_text = load_performance_metrics()

    if backtest_df.empty:
        st.error("No backtest data available. Please run the training script first.")
        return

    # Calculate metrics
    metrics = calculate_overall_metrics(backtest_df)

    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)

    accuracy = metrics.get('overall_accuracy', 0)
    precision = metrics.get('overall_precision', 0)
    f1 = metrics.get('overall_f1', 0)
    roc_auc = metrics.get('overall_roc_auc', 0)

    with col1:
        accuracy_class = "success-metric" if accuracy >= 0.60 else "warning-metric"
        st.markdown(f"""
        <div class="metric-card {accuracy_class}">
            <div class="metric-value">{accuracy:.1%}</div>
            <div class="metric-label">Directional Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{precision:.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1:.1%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roc_auc:.3f}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)

    # Model info
    st.subheader("📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Training Data**: {len(backtest_df)} predictions
        **Test Period**: {backtest_df['date'].min().strftime('%Y-%m-%d')} to {backtest_df['date'].max().strftime('%Y-%m-%d')}
        **Forecast Horizon**: 21 trading days
        **Walk-Forward Windows**: 5-year training, 21-day steps
        """)

    with col2:
        confusion_matrix_text = f"""
        **Confusion Matrix**
        - True Positives: {metrics.get('true_positives', 0)}
        - True Negatives: {metrics.get('true_negatives', 0)}
        - False Positives: {metrics.get('false_positives', 0)}
        - False Negatives: {metrics.get('false_negatives', 0)}
        """
        st.info(confusion_matrix_text)

    # Recent performance
    st.subheader("📈 Recent Performance")

    # Last 6 months performance
    recent_df = backtest_df[backtest_df['date'] >= (datetime.now() - timedelta(days=180))]
    if not recent_df.empty:
        recent_accuracy = accuracy_score(recent_df['actual_direction'], recent_df['predicted_direction'])

        st.metric(
            "Last 6 Months Accuracy",
            f"{recent_accuracy:.1%}",
            delta=f"{recent_accuracy - accuracy:.1%}"
        )

    # Performance over time chart
    st.subheader("📉 Performance Over Time")

    # Calculate rolling accuracy
    backtest_df = backtest_df.sort_values('date')
    backtest_df['rolling_accuracy'] = (
        backtest_df.set_index('date')['predicted_direction']
        .rolling('90D')
        .corr(backtest_df.set_index('date')['actual_direction'])
    )

    fig = px.line(
        backtest_df,
        x='date',
        y='rolling_accuracy',
        title='90-Day Rolling Directional Accuracy',
        labels={'rolling_accuracy': 'Accuracy', 'date': 'Date'}
    )
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="60% Target")
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

def backtest_explorer_page():
    """Detailed backtest analysis page."""
    st.header("🔍 Backtest Explorer")

    backtest_df = load_backtest_results()

    if backtest_df.empty:
        st.error("No backtest data available.")
        return

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.3,
            max_value=0.7,
            value=0.45,
            step=0.01,
            help="Adjust classification threshold"
        )

    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(backtest_df['date'].min().date(), backtest_df['date'].max().date()),
            help="Filter predictions by date range"
        )

    with col3:
        show_correct_only = st.checkbox("Show Correct Predictions Only", value=False)

    # Apply filters
    filtered_df = backtest_df.copy()

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]

    if show_correct_only:
        filtered_df = filtered_df[
            filtered_df['actual_direction'] == filtered_df['predicted_direction']
        ]

    # Apply new threshold
    filtered_df['predicted_with_new_threshold'] = (filtered_df['prediction_probability'] >= threshold).astype(int)
    new_accuracy = accuracy_score(filtered_df['actual_direction'], filtered_df['predicted_with_new_threshold'])

    st.metric(f"Accuracy with {threshold:.0%} threshold", f"{new_accuracy:.1%}")

    # Direction vs Actual Chart
    st.subheader("📊 Prediction vs Actual Direction")

    # Create direction chart
    fig = go.Figure()

    # Actual directions
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['actual_direction'],
        mode='lines',
        name='Actual Direction',
        line=dict(color='#1f77b4', width=2),
        yaxis='y1'
    ))

    # Predicted directions
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['predicted_with_new_threshold'],
        mode='lines',
        name='Predicted Direction',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        yaxis='y1'
    ))

    # Prediction probabilities
    fig.add_trace(go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['prediction_probability'],
        mode='lines',
        name='Prediction Probability',
        line=dict(color='#2ca02c', width=1),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Copper Direction Predictions vs Actual',
        xaxis_title='Date',
        yaxis=dict(title='Direction (0=Down, 1=Up)', side='left'),
        yaxis2=dict(title='Probability', overlaying='y', side='right'),
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rolling accuracy heatmap
    st.subheader("🗺️ Rolling Accuracy Heatmap")

    # Create monthly accuracy pivot
    filtered_df['year_month'] = filtered_df['date'].dt.to_period('M')
    monthly_accuracy = filtered_df.groupby('year_month').apply(
        lambda x: accuracy_score(x['actual_direction'], x['predicted_with_new_threshold'])
        if len(x) > 0 else np.nan
    )

    monthly_df = monthly_accuracy.reset_index()
    monthly_df['year'] = monthly_df['year_month'].dt.year
    monthly_df['month'] = monthly_df['year_month'].dt.month

    pivot_accuracy = monthly_df.pivot(index='year', columns='month', values=0)

    fig = px.imshow(
        pivot_accuracy,
        title='Monthly Directional Accuracy',
        labels=dict(x="Month", y="Year", color="Accuracy"),
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("📋 Confusion Matrix")

    cm = confusion_matrix(
        filtered_df['actual_direction'],
        filtered_df['predicted_with_new_threshold']
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        title='Confusion Matrix',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Predicted Down', 'Predicted Up'],
        y=['Actual Down', 'Actual Up']
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    # Probability histogram
    st.subheader("📊 Prediction Probability Distribution")

    fig = px.histogram(
        filtered_df,
        x='prediction_probability',
        color='actual_direction',
        nbins=20,
        title='Prediction Probability Distribution',
        labels={'prediction_probability': 'Prediction Probability', 'actual_direction': 'Actual Direction'},
        color_discrete_map={0: '#dc3545', 1: '#28a745'}
    )
    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

def feature_importance_page():
    """Feature importance and SHAP analysis page."""
    st.header("🔍 Feature Importance Analysis")

    model, feature_cols = load_final_model()

    if model is None:
        st.error("Model not found. Please run training first.")
        return

    # Get feature importances
    try:
        importances = model.estimator_.feature_importances_
    except:
        # For calibrated models, access the base estimator
        importances = model.estimator.feature_importances_

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Top 20 features
    st.subheader("📈 Top 20 Feature Importances")

    fig = px.bar(
        importance_df.head(20),
        x='importance',
        y='feature',
        orientation='h',
        title='XGBoost Feature Importances (Top 20)',
        labels={'importance': 'Importance', 'feature': 'Feature'}
    )
    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    # Feature categories
    st.subheader("🏷️ Feature Categories")

    # Categorize features
    categories = {
        'Momentum': [f for f in feature_cols if 'ret_' in f],
        'Moving Averages': [f for f in feature_cols if 'ma_' in f or 'price_over_ma' in f],
        'Volatility': [f for f in feature_cols if 'vol' in f],
        'Volume': [f for f in feature_cols if 'volume' in f or 'obv' in f],
        'Candles': [f for f in feature_cols if any(x in f for x in ['body', 'upper_wick', 'lower_wick', 'bullish'])],
        'DXY': [f for f in feature_cols if 'dxy' in f.lower()],
        'Gold': [f for f in feature_cols if 'gold' in f.lower()],
        'Time': [f for f in feature_cols if any(x in f for x in ['dow', 'month', 'year'])]
    }

    # Calculate category importances
    category_importance = {}
    for category, features in categories.items():
        if features:
            cat_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
            category_importance[category] = cat_importance

    # Category importance chart
    cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Total_Importance'])
    cat_df = cat_df.sort_values('Total_Importance', ascending=False)

    fig = px.pie(
        cat_df,
        values='Total_Importance',
        names='Category',
        title='Feature Importance by Category'
    )
    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    # SHAP Analysis (if available)
    st.subheader("🎯 SHAP Analysis")

    try:
        import shap

        # Load some sample data for SHAP
        backtest_df = load_backtest_results()
        if not backtest_df.empty:
            # Get a sample of recent data
            sample_data = backtest_df.tail(100).copy()

            # Create feature matrix (this is approximate since we don't have raw features)
            # In a real implementation, you'd want to load the actual feature matrix
            st.info("SHAP analysis requires the original feature matrix. This is a placeholder for demonstration.")

            st.code("""
# Real SHAP implementation would look like this:

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Summary plot
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols)

# Waterfall plot for individual prediction
shap.plots.waterfall(explainer.expected_value, shap_values[0], X_sample.iloc[0])
            """)

    except ImportError:
        st.warning("SHAP library not available. Install with: pip install shap")

def recent_predictions_page():
    """Recent predictions and live data page."""
    st.header("📅 Recent Predictions")

    backtest_df = load_backtest_results()

    if backtest_df.empty:
        st.error("No backtest data available.")
        return

    # Get recent predictions (last 30 days)
    recent_df = backtest_df[backtest_df['date'] >= (datetime.now() - timedelta(days=30))].copy()

    if recent_df.empty:
        st.warning("No predictions in the last 30 days.")
        return

    # Format for display
    display_df = recent_df[['date', 'close_price', 'prediction_probability', 'predicted_direction', 'actual_direction']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['predicted_direction'] = display_df['predicted_direction'].map({0: '📉 Down', 1: '📈 Up'})
    display_df['actual_direction'] = display_df['actual_direction'].map({0: '📉 Down', 1: '📈 Up'})
    display_df['correct'] = display_df['predicted_direction'] == display_df['actual_direction']
    display_df['prediction_probability'] = display_df['prediction_probability'].round(3)

    # Color coding
    def color_correct(val):
        return 'color: green' if val else 'color: red'

    styled_df = display_df.style.applymap(color_correct, subset=['correct'])

    st.dataframe(styled_df, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)

    recent_accuracy = accuracy_score(recent_df['actual_direction'], recent_df['predicted_direction'])

    with col1:
        st.metric("Recent Accuracy (30 days)", f"{recent_accuracy:.1%}")

    with col2:
        avg_probability = recent_df['prediction_probability'].mean()
        st.metric("Avg Prediction Confidence", f"{avg_probability:.1%}")

    with col3:
        total_predictions = len(recent_df)
        st.metric("Total Predictions", total_predictions)

    # Price movement visualization
    st.subheader("💰 Price Movements")

    fig = go.Figure()

    # Copper prices
    fig.add_trace(go.Scatter(
        x=recent_df['date'],
        y=recent_df['close_price'],
        mode='lines+markers',
        name='Copper Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Color markers by prediction correctness
    colors = ['green' if correct else 'red' for correct in (recent_df['actual_direction'] == recent_df['predicted_direction']).values]

    fig.add_trace(go.Scatter(
        x=recent_df['date'],
        y=recent_df['close_price'],
        mode='markers',
        name='Predictions',
        marker=dict(color=colors, size=8, symbol='circle'),
        showlegend=False
    ))

    fig.update_layout(
        title='Recent Copper Prices with Prediction Markers',
        xaxis_title='Date',
        yaxis_title='Copper Price (USD)',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("🟢 Green markers = Correct predictions | 🔴 Red markers = Incorrect predictions")

def settings_page():
    """Settings and model management page."""
    st.header("⚙️ Settings & Model Management")

    # Retrain model section
    st.subheader("🔄 Retrain Model")

    col1, col2 = st.columns(2)

    with col1:
        horizon_options = [7, 14, 21, 42]
        selected_horizon = st.selectbox(
            "Forecast Horizon (days)",
            options=horizon_options,
            index=2,  # Default to 21
            help="Number of days ahead to predict direction"
        )

    with col2:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.3,
            max_value=0.7,
            value=0.45,
            step=0.01,
            help="Classification threshold for predictions"
        )

    if st.button("🚀 Retrain Model", type="primary"):
        st.info("Retraining functionality would be implemented here. In a real application, this would:")
        st.code("""
1. Update configuration parameters
2. Re-run feature engineering
3. Execute walk-forward validation
4. Save new model and results
5. Refresh dashboard data
        """)

    # Model info
    st.subheader("📊 Current Model Info")

    try:
        import os
        model_path = "outputs/final_model.pkl"
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / 1024  # KB
            st.success(f"✅ Model loaded ({model_size:.1f} KB)")

            # Load feature count
            _, feature_cols = load_final_model()
            if feature_cols:
                st.info(f"📈 {len(feature_cols)} features used")
        else:
            st.warning("❌ No trained model found")
    except:
        st.error("Error loading model information")

    # System info
    st.subheader("💻 System Information")

    st.code(f"""
Python Version: {sys.version.split()[0]}
Platform: {sys.platform}
Working Directory: {os.getcwd()}

Last Training: Check outputs/performance_metrics.txt
Backtest Results: outputs/backtest_results.csv
Model File: outputs/final_model.pkl
    """)

# =========================
#  MAIN APP
# =========================

def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")

    pages = {
        "🏠 Overview": overview_page,
        "🔍 Backtest Explorer": backtest_explorer_page,
        "🔍 Feature Importance": feature_importance_page,
        "📅 Recent Predictions": recent_predictions_page,
        "⚙️ Settings": settings_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Copper Brain v2**")
    st.sidebar.markdown("*Built with Streamlit & XGBoost*")
    st.sidebar.markdown("---")

    # Load selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()