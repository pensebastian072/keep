"""Configuration utilities for the options surface edge engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class ScoringWeights:
    """Weights for edge score components."""

    ev: float = 35.0
    max_loss: float = 20.0
    pop: float = 20.0
    vega: float = 10.0
    liquidity: float = 15.0

    def as_dict(self) -> dict[str, float]:
        """Return weights as dictionary."""
        return {
            "ev_score": self.ev,
            "max_loss_score": self.max_loss,
            "pop_score": self.pop,
            "vega_score": self.vega,
            "liquidity_score": self.liquidity,
        }


@dataclass(slots=True)
class ScenarioGrid:
    """Scenario re-pricing grid configuration."""

    spot_shocks: list[float] = field(default_factory=lambda: [-0.05, -0.025, 0.0, 0.025, 0.05])
    iv_shocks: list[float] = field(default_factory=lambda: [-0.20, 0.0, 0.20])
    days_forward: list[int] = field(default_factory=lambda: [7, 21])


@dataclass(slots=True)
class LiquidityConfig:
    """Liquidity scoring controls."""

    spread_cap_pct: float = 0.25
    min_open_interest: float = 100.0
    min_volume: float = 50.0
    vega_scale: float = 0.20


def _default_output_dir() -> Path:
    """Resolve project output directory."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "outputs"


@dataclass(slots=True)
class EngineConfig:
    """Engine runtime configuration."""

    provider_order: list[str] = field(default_factory=lambda: ["yfinance", "openbb"])
    openbb_provider: str = "yfinance"
    dte_min: int = 30
    dte_max: int = 120
    enabled_strategies: list[str] = field(
        default_factory=lambda: [
            "put_credit_spread",
            "call_debit_spread",
            "calendar_spread",
            "iron_condor",
            "long_strangle",
        ]
    )
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0
    contract_multiplier: int = 100
    trend_lookback_days: int = 180
    premium_per_share_limit: float = 10.0
    data_mode: Literal["delayed_eod", "best_effort_intraday"] = "delayed_eod"
    retention_keep_last: int = 30
    run_label_prefix: str = "Output"
    best_trade_min_edge: float = 60.0
    best_trade_min_confidence: float = 0.70
    autopilot_enabled: bool = False
    autopilot_time: str = "16:20"
    autopilot_timezone: str = "America/New_York"
    autopilot_ticker: str = "SPY"
    openai_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    learning_enabled: bool = True
    learning_horizon_days: int = 7
    tune_weights_weekly: bool = True
    output_dir: Path = field(default_factory=_default_output_dir)
    save_latest_chain: bool = True
    max_tickers: int = 30
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    scenario: ScenarioGrid = field(default_factory=ScenarioGrid)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    run_id: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))

    @property
    def run_dir(self) -> Path:
        """Return run-specific output directory."""
        return self.output_dir / "runs" / self.run_id

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)


def load_tickers_from_csv(path: str | Path) -> list[str]:
    """Load tickers from a CSV file."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return []
    if "ticker" in frame.columns:
        source = frame["ticker"]
    else:
        source = frame.iloc[:, 0]
    tickers = [str(x).strip().upper() for x in source.tolist() if str(x).strip()]
    return sorted(set(tickers))


def parse_tickers(raw: str | Iterable[str]) -> list[str]:
    """Parse and normalize ticker input."""
    if isinstance(raw, str):
        values = [x.strip().upper() for x in raw.split(",")]
    else:
        values = [str(x).strip().upper() for x in raw]
    return [ticker for ticker in values if ticker]


def parse_csv_values(raw: str, lowercase: bool = False) -> list[str]:
    """Parse comma-separated values."""
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if lowercase:
        return [value.lower() for value in values]
    return values
