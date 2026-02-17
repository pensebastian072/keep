"""Tests for autopilot one-shot behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_surface_edge_engine.autopilot import scheduler
from options_surface_edge_engine.pipeline import ScanResult
from options_surface_edge_engine.utils.config import EngineConfig


def test_autopilot_one_shot_writes_state(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "outputs"
    cfg = EngineConfig(output_dir=output_dir, autopilot_ticker="SPY")

    def _fake_run_scan(tickers, config):  # noqa: ANN001, ANN202
        return ScanResult(
            run_id="r123",
            run_label="Output-0001",
            ranked_trades=pd.DataFrame(),
            scenario_table=pd.DataFrame(),
            surface_metrics=pd.DataFrame(),
            expiry_features=pd.DataFrame(),
            raw_candidates=pd.DataFrame(),
            best_trade=pd.DataFrame(),
            run_metadata=pd.DataFrame(),
            data_quality=pd.DataFrame(),
        )

    monkeypatch.setattr(scheduler, "run_scan", _fake_run_scan)
    result = scheduler.run_autopilot_once(cfg)
    assert result.run_id == "r123"
    state_path = output_dir / "autopilot" / "autopilot_state.json"
    assert state_path.exists()
