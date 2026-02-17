"""Tests for best-trade selection gate."""

from __future__ import annotations

import pandas as pd

from options_surface_edge_engine.ranking.selection import select_best_trade


def test_select_best_trade_below_threshold_returns_none() -> None:
    frame = pd.DataFrame(
        [
            {"trade_id": "A", "edge_score": 55, "edge_score_confidence": 0.8},
            {"trade_id": "B", "edge_score": 59, "edge_score_confidence": 0.9},
        ]
    )
    selected = select_best_trade(frame, min_edge=60, min_conf=0.7)
    assert selected is None


def test_select_best_trade_picks_highest_gated() -> None:
    frame = pd.DataFrame(
        [
            {"trade_id": "A", "edge_score": 65, "edge_score_confidence": 0.75},
            {"trade_id": "B", "edge_score": 70, "edge_score_confidence": 0.80},
            {"trade_id": "C", "edge_score": 90, "edge_score_confidence": 0.60},
        ]
    )
    selected = select_best_trade(frame, min_edge=60, min_conf=0.7)
    assert selected is not None
    assert selected["trade_id"] == "B"
