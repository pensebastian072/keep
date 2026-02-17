"""Unit tests for strategy generation."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_surface_edge_engine.strategies.generators import generate_candidate_trades
from options_surface_edge_engine.utils.config import EngineConfig


def _chain_for_put_credit(expensive: bool = False) -> pd.DataFrame:
    exp = date.today() + timedelta(days=45)
    short_price = 12.0 if expensive else 2.2
    long_price = 1.0 if expensive else 0.9
    rows = [
        {"symbol": "TEST", "provider": "unit", "quote_ts": pd.Timestamp.utcnow(), "underlying_price": 100.0, "expiration": exp, "dte": 45, "option_type": "put", "strike": 100.0, "contract_symbol": "TESTP100", "bid": short_price - 0.1, "ask": short_price + 0.1, "mark": short_price, "last_trade_price": short_price, "volume": 1000, "open_interest": 2000, "implied_volatility": 0.45, "delta": -0.30, "gamma": 0.02, "theta": -0.01, "vega": 0.2, "currency": "USD"},
        {"symbol": "TEST", "provider": "unit", "quote_ts": pd.Timestamp.utcnow(), "underlying_price": 100.0, "expiration": exp, "dte": 45, "option_type": "put", "strike": 95.0, "contract_symbol": "TESTP95", "bid": long_price - 0.1, "ask": long_price + 0.1, "mark": long_price, "last_trade_price": long_price, "volume": 900, "open_interest": 1500, "implied_volatility": 0.43, "delta": -0.15, "gamma": 0.015, "theta": -0.008, "vega": 0.16, "currency": "USD"},
        {"symbol": "TEST", "provider": "unit", "quote_ts": pd.Timestamp.utcnow(), "underlying_price": 100.0, "expiration": exp, "dte": 45, "option_type": "call", "strike": 105.0, "contract_symbol": "TESTC105", "bid": 1.2, "ask": 1.4, "mark": 1.3, "last_trade_price": 1.3, "volume": 500, "open_interest": 1100, "implied_volatility": 0.40, "delta": 0.25, "gamma": 0.01, "theta": -0.006, "vega": 0.14, "currency": "USD"},
    ]
    return pd.DataFrame(rows)


def test_put_credit_trade_leg_selection() -> None:
    cfg = EngineConfig(enabled_strategies=["put_credit_spread"], dte_min=30, dte_max=120)
    symbol_metrics = {"iv_30": 0.45, "skew_25d": 0.05, "term_slope": 0.01, "spot": 100.0, "expansion_proxy": 0.02}
    trades = generate_candidate_trades("TEST", _chain_for_put_credit(expensive=False), symbol_metrics, "bullish", cfg)
    assert trades
    trade = trades[0]
    assert trade.strategy == "put_credit_spread"
    assert len(trade.legs) == 2
    short_leg = [leg for leg in trade.legs if leg.position == "short"][0]
    long_leg = [leg for leg in trade.legs if leg.position == "long"][0]
    assert short_leg.delta is not None and abs(short_leg.delta + 0.30) < 0.05
    assert long_leg.delta is not None and abs(long_leg.delta + 0.15) < 0.05
    assert short_leg.dte >= 30 and short_leg.dte <= 60


def test_dte_window_enforced() -> None:
    cfg = EngineConfig(enabled_strategies=["put_credit_spread"], dte_min=70, dte_max=120)
    symbol_metrics = {"iv_30": 0.45, "skew_25d": 0.05, "term_slope": 0.01, "spot": 100.0, "expansion_proxy": 0.02}
    trades = generate_candidate_trades("TEST", _chain_for_put_credit(expensive=False), symbol_metrics, "bullish", cfg)
    assert trades == []


def test_premium_constraint_flag() -> None:
    cfg = EngineConfig(enabled_strategies=["put_credit_spread"], dte_min=30, dte_max=120, premium_per_share_limit=10.0)
    symbol_metrics = {"iv_30": 0.45, "skew_25d": 0.05, "term_slope": 0.01, "spot": 100.0, "expansion_proxy": 0.02}
    trades = generate_candidate_trades("TEST", _chain_for_put_credit(expensive=True), symbol_metrics, "bullish", cfg)
    assert trades
    assert trades[0].premium_constraint is False
