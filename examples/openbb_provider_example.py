"""Example: run scan with OpenBB adapter fallback."""

from options_surface_edge_engine.pipeline import run_scan
from options_surface_edge_engine.utils.config import EngineConfig


if __name__ == "__main__":
    cfg = EngineConfig(
        provider_order=["yfinance", "openbb"],
        openbb_provider="yfinance",
    )
    result = run_scan(["SPY", "QQQ", "GLD", "NVDA"], config=cfg)
    print(result.ranked_trades.head(10).to_string(index=False))
