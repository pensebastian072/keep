"""Unit tests for edge score computation."""

from __future__ import annotations

import pandas as pd

from options_surface_edge_engine.ranking.edge_score import compute_edge_scores
from options_surface_edge_engine.utils.config import LiquidityConfig, ScoringWeights


def test_score_bounds_and_order() -> None:
    frame = pd.DataFrame(
        [
            {"trade_id": "A", "scenario_ev": 100, "max_loss": 400, "pop_proxy": 0.7, "net_vega": 0.05, "liquidity_proxy": 0.8},
            {"trade_id": "B", "scenario_ev": -10, "max_loss": 1800, "pop_proxy": 0.4, "net_vega": 0.2, "liquidity_proxy": 0.4},
        ]
    )
    ranked = compute_edge_scores(frame, ScoringWeights(), LiquidityConfig())
    assert ranked["edge_score"].between(0, 100).all()
    assert ranked.iloc[0]["trade_id"] == "A"


def test_weight_renormalization_with_missing_components() -> None:
    frame = pd.DataFrame(
        [
            {"trade_id": "A", "scenario_ev": 50, "max_loss": 500, "pop_proxy": 0.6, "net_vega": None, "liquidity_proxy": None},
        ]
    )
    ranked = compute_edge_scores(frame, ScoringWeights(), LiquidityConfig())
    confidence = float(ranked.iloc[0]["edge_score_confidence"])
    assert 0.0 < confidence < 1.0
