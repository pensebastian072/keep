# options-surface-edge-engine

Production-quality options workflow for scanning options surfaces, generating candidate strategies, simulating scenario P&L, ranking ideas, and operating a daily assistant loop.

## Features
- Provider abstraction with `yfinance` default and optional OpenBB adapter fallback.
- Standardized options chain schema across providers.
- Surface feature pipeline:
  - ATM IV
  - 25-delta skew proxy
  - Term structure slope (`IV90 - IV30`)
  - Smile curvature proxies
  - Gamma zones (OI-weighted gamma)
- Strategy generator:
  - Put credit spread
  - Call debit spread
  - Calendar spread
  - Iron condor
  - Long strangle
- Scenario engine (Black-Scholes repricing):
  - Spot: `-5%, -2.5%, 0, +2.5%, +5%`
  - IV: `-20%, 0, +20%`
  - Time forward: `+7d, +21d`
- EdgeScore (0-100) with missing-data-aware confidence.
- Streamlit dashboard with ranked ideas, scenario views, and CSV export.
- Single-ticker command center default with advanced multi-ticker mode.
- Data freshness modes with staleness metadata (`delayed_eod` default).
- Run registry (`Output-####`) with manual delete + automatic retention pruning.
- Best-trade quality gate with explicit no-trade output when confidence is low.
- Daily autopilot runner and scheduler.
- Scoped in-app chat agent (OpenAI-backed parser + safe confirmations).
- Learning loop: prediction journal, 7-day outcome labeling, weekly score weight tuning.

## Installation
```bash
cd options-surface-edge-engine
python -m pip install -e .
python -m pip install -e ".[streamlit]"
```

Optional extras:
```bash
python -m pip install -e ".[openbb]"
python -m pip install -e ".[polygon]"
python -m pip install -e ".[dev]"
```

## Quick Run
CLI scan:
```bash
osee-scan --tickers SPY --data-mode delayed_eod
```

Run dashboard:
```bash
osee-dashboard
```

Autopilot one-shot:
```bash
osee-autopilot --ticker SPY --time 16:20 --timezone America/New_York
```

Autopilot scheduler (blocking):
```bash
osee-autopilot --ticker SPY --time 16:20 --timezone America/New_York --daemon
```

Run management:
```bash
osee-runs list --limit 30
osee-runs delete --run-id 20260215T164215Z
osee-runs prune --keep-last 30
```

## CLI Usage
```bash
osee-scan --tickers SPY --dte-min 30 --dte-max 120 --top 20 --best-edge-min 60 --best-conf-min 0.7
osee-scan --tickers-csv ./tickers.csv --provider-order yfinance,openbb --risk-free-rate 0.045 --data-mode best_effort_intraday
```

## Output Artifacts
- `outputs/{symbol}_chain.parquet` latest snapshot per symbol
- `outputs/runs/{run_id}/{symbol}_chain.parquet` run-specific raw chain
- `outputs/runs/{run_id}/ranked_trades.csv`
- `outputs/runs/{run_id}/scenario_table.csv`
- `outputs/runs/{run_id}/surface_metrics.csv`
- `outputs/runs/{run_id}/best_trade.csv`
- `outputs/runs/{run_id}/run_metadata.csv`
- `outputs/runs/{run_id}/data_quality.csv`
- `outputs/run_registry.parquet`
- `outputs/learning/trade_journal.parquet`
- `outputs/learning/weights.json`

## OpenBB Integration (Optional)
1. Install optional dependency:
```bash
python -m pip install -e ".[openbb]"
```
2. Configure credentials in `~/.openbb_platform/user_settings.json` when using paid providers.
3. Run with OpenBB fallback enabled:
```bash
osee-scan --tickers SPY,QQQ --provider-order yfinance,openbb --openbb-provider yfinance
```

## Chat Agent
- Backend model defaults to `gpt-4o-mini`.
- Set environment variable `OPENAI_API_KEY` for LLM parsing.
- Without API key, command-style chat still works (`run ticker AAPL`, `list outputs`, etc.).

## Assumptions
- Black-Scholes European approximation only.
- Default risk-free rate is `4.5%` unless overridden.
- Missing provider metrics remain null (never fabricated).
- Best trade is suggested only when edge and confidence pass configured thresholds.

## Development
Run tests:
```bash
python -m pytest
```
