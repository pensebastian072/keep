"""Tests for data freshness metadata."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from options_surface_edge_engine.pipeline import _stamp_data_mode
from options_surface_edge_engine.utils.config import EngineConfig


def _base_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "provider": "unit",
                "quote_ts": datetime.now(UTC),
                "underlying_price": 500.0,
                "expiration": datetime.now(UTC).date(),
                "dte": 30,
                "option_type": "call",
                "strike": 500.0,
                "mark": 5.0,
                "implied_volatility": 0.2,
            }
        ]
    )


def test_delayed_eod_stamping() -> None:
    cfg = EngineConfig(data_mode="delayed_eod")
    stamped, quality = _stamp_data_mode(_base_chain(), cfg)
    assert str(stamped.iloc[0]["data_mode_used"]) == "delayed_eod"
    assert quality["data_mode_used"] == "delayed_eod"
    assert quality["staleness_minutes"] is not None


def test_intraday_stamping() -> None:
    cfg = EngineConfig(data_mode="best_effort_intraday")
    stamped, quality = _stamp_data_mode(_base_chain(), cfg)
    assert str(stamped.iloc[0]["data_mode_used"]) == "best_effort_intraday"
    assert quality["data_mode_used"] == "best_effort_intraday"
