"""Optional integration smoke test."""

from __future__ import annotations

import pytest

from options_surface_edge_engine.pipeline import run_scan
from options_surface_edge_engine.utils.config import EngineConfig


@pytest.mark.integration
def test_live_smoke_yfinance() -> None:
    cfg = EngineConfig(provider_order=["yfinance"], max_tickers=4)
    result = run_scan(["SPY", "QQQ", "GLD", "NVDA"], config=cfg)
    assert not result.surface_metrics.empty
