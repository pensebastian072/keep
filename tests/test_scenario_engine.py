"""Unit tests for scenario engine."""

from __future__ import annotations

from datetime import date, timedelta

from options_surface_edge_engine.pricing.scenario_engine import run_scenarios
from options_surface_edge_engine.strategies.templates import CandidateTrade, OptionLeg
from options_surface_edge_engine.utils.config import EngineConfig


def _sample_trade() -> CandidateTrade:
    leg = OptionLeg(
        symbol="TEST",
        option_type="call",
        position="long",
        expiration=date.today() + timedelta(days=60),
        dte=60,
        strike=100.0,
        quantity=1,
        entry_price=3.5,
        implied_volatility=0.25,
        delta=0.5,
        gamma=0.02,
        theta=-0.01,
        vega=0.15,
        bid=3.4,
        ask=3.6,
        mark=3.5,
        volume=1000,
        open_interest=1200,
        contract_symbol="TESTC100",
    )
    return CandidateTrade(
        trade_id="T1",
        symbol="TEST",
        strategy="call_debit_spread",
        regime="bullish",
        iv_regime="moderate",
        spot=100.0,
        dte=60,
        legs=[leg],
        net_premium_per_share=3.5,
        max_loss=350.0,
        max_gain=999.0,
        breakeven_low=103.5,
        breakeven_high=None,
        pop_proxy=0.4,
    )


def test_scenario_grid_shape() -> None:
    cfg = EngineConfig()
    scenarios, summary = run_scenarios([_sample_trade()], cfg)
    assert len(scenarios) == 30
    assert not summary.empty
    assert summary.iloc[0]["trade_id"] == "T1"


def test_directional_monotonicity() -> None:
    cfg = EngineConfig()
    scenarios, _ = run_scenarios([_sample_trade()], cfg)
    base = scenarios[(scenarios["days_forward"] == 7) & (scenarios["iv_shock"] == 0.0)]
    pnl_up = float(base[base["spot_shock"] == 0.05]["pnl"].iloc[0])
    pnl_dn = float(base[base["spot_shock"] == -0.05]["pnl"].iloc[0])
    assert pnl_up > pnl_dn
