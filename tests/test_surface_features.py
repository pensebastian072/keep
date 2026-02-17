"""Unit tests for surface feature extraction."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_surface_edge_engine.features.surface_features import build_surface_features


def _make_chain() -> pd.DataFrame:
    today = date.today()
    exp30 = today + timedelta(days=30)
    exp90 = today + timedelta(days=90)
    rows = [
        # 30 DTE
        {"symbol": "TEST", "expiration": exp30, "dte": 30, "option_type": "call", "strike": 100, "implied_volatility": 0.20, "delta": 0.50, "gamma": 0.020, "open_interest": 1000, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp30, "dte": 30, "option_type": "put", "strike": 100, "implied_volatility": 0.22, "delta": -0.50, "gamma": 0.019, "open_interest": 900, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp30, "dte": 30, "option_type": "call", "strike": 105, "implied_volatility": 0.19, "delta": 0.25, "gamma": 0.016, "open_interest": 1200, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp30, "dte": 30, "option_type": "put", "strike": 95, "implied_volatility": 0.24, "delta": -0.25, "gamma": 0.018, "open_interest": 1300, "underlying_price": 100},
        # 90 DTE
        {"symbol": "TEST", "expiration": exp90, "dte": 90, "option_type": "call", "strike": 100, "implied_volatility": 0.26, "delta": 0.50, "gamma": 0.011, "open_interest": 700, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp90, "dte": 90, "option_type": "put", "strike": 100, "implied_volatility": 0.28, "delta": -0.50, "gamma": 0.011, "open_interest": 650, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp90, "dte": 90, "option_type": "call", "strike": 105, "implied_volatility": 0.25, "delta": 0.25, "gamma": 0.010, "open_interest": 600, "underlying_price": 100},
        {"symbol": "TEST", "expiration": exp90, "dte": 90, "option_type": "put", "strike": 95, "implied_volatility": 0.30, "delta": -0.25, "gamma": 0.012, "open_interest": 800, "underlying_price": 100},
    ]
    return pd.DataFrame(rows)


def test_surface_metrics_basic() -> None:
    expiry, summary = build_surface_features(_make_chain())
    assert not expiry.empty
    assert summary["symbol"] == "TEST"
    assert abs(summary["iv_30"] - 0.21) < 1e-6
    assert abs(summary["iv_90"] - 0.27) < 1e-6
    assert abs(summary["term_slope"] - 0.06) < 1e-6
    assert abs(summary["skew_25d"] - 0.05) < 1e-6
    assert abs(summary["put_curvature"] - 0.03) < 1e-6
    assert abs(summary["call_curvature"] + 0.02) < 1e-6


def test_surface_missing_values_stay_null() -> None:
    chain = _make_chain()
    chain["implied_volatility"] = pd.NA
    expiry, summary = build_surface_features(chain)
    assert not expiry.empty
    assert summary["iv_30"] is None
    assert summary["term_slope"] is None
    assert summary["metric_available_term_slope"] is False
