"""Tests for learning journal and tuner."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from options_surface_edge_engine.learning import journal as journal_module
from options_surface_edge_engine.learning.journal import evaluate_matured_predictions, read_journal, record_prediction
from options_surface_edge_engine.learning.tuner import tune_weights
from options_surface_edge_engine.utils.config import ScoringWeights


def test_record_prediction_writes_journal(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    trade = pd.Series(
        {
            "trade_id": "T1",
            "symbol": "SPY",
            "strategy": "call_debit_spread",
            "spot": 500.0,
            "edge_score": 70.0,
            "edge_score_confidence": 0.8,
            "scenario_ev": 10.0,
            "pop_proxy": 0.55,
            "net_delta": 0.3,
            "net_vega": 0.1,
            "max_loss": 400.0,
            "premium_constraint": True,
            "ev_score": 65.0,
            "max_loss_score": 80.0,
            "pop_score": 55.0,
            "vega_score": 70.0,
            "liquidity_score": 60.0,
        }
    )
    record_prediction(output_dir, run_id="r1", trade_row=trade, context={"run_label": "Output-0001"}, horizon_days=7)
    journal = read_journal(output_dir)
    assert len(journal) == 1
    assert journal.iloc[0]["status"] == "pending"


def test_tuner_returns_valid_weights() -> None:
    history = pd.DataFrame(
        [
            {"ev_score": 80, "max_loss_score": 70, "pop_score": 60, "vega_score": 50, "liquidity_score": 75, "outcome": "right"},
            {"ev_score": 30, "max_loss_score": 40, "pop_score": 35, "vega_score": 45, "liquidity_score": 40, "outcome": "wrong"},
            {"ev_score": 75, "max_loss_score": 65, "pop_score": 55, "vega_score": 60, "liquidity_score": 70, "outcome": "right"},
            {"ev_score": 20, "max_loss_score": 30, "pop_score": 25, "vega_score": 35, "liquidity_score": 30, "outcome": "wrong"},
        ]
    )
    tuned = tune_weights(history, ScoringWeights())
    values = tuned.as_dict()
    assert abs(sum(values.values()) - 100.0) < 1e-6
    assert all(5.0 <= v <= 50.0 for v in values.values())


def test_evaluate_matured_predictions(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    trade = pd.Series(
        {
            "trade_id": "T2",
            "symbol": "SPY",
            "strategy": "put_credit_spread",
            "spot": 100.0,
            "edge_score": 65.0,
            "edge_score_confidence": 0.8,
            "scenario_ev": 5.0,
            "pop_proxy": 0.6,
            "net_delta": 0.5,
            "net_vega": 0.1,
            "max_loss": 500.0,
            "premium_constraint": False,
            "ev_score": 60.0,
            "max_loss_score": 70.0,
            "pop_score": 60.0,
            "vega_score": 55.0,
            "liquidity_score": 45.0,
        }
    )
    record_prediction(output_dir, run_id="r2", trade_row=trade, context={"run_label": "Output-0002"}, horizon_days=7)
    monkeypatch.setattr(journal_module, "_latest_spot", lambda symbol, as_of: 95.0)
    as_of = date.today() + timedelta(days=8)
    evaluated = evaluate_matured_predictions(output_dir, as_of_date=as_of)
    assert not evaluated.empty
    row = evaluated.iloc[0]
    assert row["status"] == "evaluated"
    assert row["outcome"] == "wrong"
    assert "spot_move_unfavorable" in row["reason_codes"]
