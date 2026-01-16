# Copper Brain v2 - Production-Grade Direction Forecasting

🧠 **AI-Powered Copper Price Direction Prediction** using XGBoost binary classification with walk-forward validation.

## Overview

Copper Brain v2 predicts **21-day ahead price direction** (up/down) for COMEX copper futures using:
- Free daily data from yfinance
- Advanced technical features including DXY and Gold correlations
- Walk-forward rolling validation for realistic backtesting
- XGBoost binary classifier targeting 60%+ directional accuracy

## Project Structure

```
copper_brain_v2/
├── app.py                 # Streamlit dashboard
├── copper_brain_v2.py     # Core model training + backtest
├── features.py            # Feature engineering functions
├── model_utils.py         # Training, walk-forward, performance metrics
├── data/                  # Cached data files
├── outputs/               # Backtest results, model artifacts
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model & Run Backtest

```bash
python copper_brain_v2.py
```

This will:
- Download copper, DXY, and gold data from yfinance
- Engineer all technical features
- Run walk-forward validation (5-year rolling window)
- Save results to `outputs/backtest_results.csv`
- Save trained model to `outputs/model.pkl`

### 2. Launch Dashboard

```bash
streamlit run app.py
```

Or if streamlit is not in PATH:

```bash
python -m streamlit run app.py
```

## Features

### Data Sources (Free Only)
- **Copper futures**: HG=F (COMEX)
- **US Dollar Index**: DX-Y.NYB
- **Gold futures**: GC=F

### Feature Groups
1. **Momentum**: 1d, 2d, 3d, 5d, 10d returns, rolling max/min
2. **Moving Averages**: 5d, 10d, 20d MAs and price ratios
3. **Volatility**: 10d and 21d realized volatility
4. **Volume**: Log volume, z-score, OBV
5. **Candle Structure**: Body, wicks, bullish/bearish
6. **Alpha Features**: DXY correlation, copper/gold ratio

### Model Architecture
- XGBoost Binary Classifier
- 500 trees, max depth 6, learning rate 0.03
- Walk-forward validation with 5-year training window
- 21-day test step

## Performance Target

- **Directional Accuracy**: ≥ 60%
- Additional metrics: Precision, Recall, F1, ROC-AUC

## Dashboard Views

1. **Overview**: Project description, metrics summary
2. **Backtest Explorer**: Charts, confusion matrix, threshold tuning
3. **Feature Importance**: XGBoost importances, SHAP plots
4. **Recent Predictions**: Last 30 days with probabilities
5. **Settings**: Retrain, horizon selection, threshold adjustment

## License

MIT License - Use at your own risk for educational purposes only.
