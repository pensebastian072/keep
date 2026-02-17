"""CLI entrypoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from options_surface_edge_engine.autopilot.scheduler import AutopilotConfig, run_autopilot_once, start_daily_scheduler
from options_surface_edge_engine.pipeline import run_scan
from options_surface_edge_engine.storage.run_store import delete_run, list_runs, prune_runs
from options_surface_edge_engine.utils import config as config_utils

EngineConfig = config_utils.EngineConfig
load_tickers_from_csv = config_utils.load_tickers_from_csv
parse_tickers = config_utils.parse_tickers


def parse_csv_values(raw: str, lowercase: bool = False) -> list[str]:
    """Parse comma-separated values with backward-compatible fallback."""
    parser = getattr(config_utils, "parse_csv_values", None)
    if callable(parser):
        return parser(raw, lowercase=lowercase)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if lowercase:
        return [value.lower() for value in values]
    return values

DEFAULT_STRATEGIES = "put_credit_spread,call_debit_spread,calendar_spread,iron_condor,long_strangle"


def _scan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run options surface edge scan.")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers, e.g. SPY,QQQ,GLD,NVDA.")
    parser.add_argument("--tickers-csv", type=str, default="", help="Path to CSV file with `ticker` column or first column of symbols.")
    parser.add_argument("--provider-order", type=str, default="yfinance,openbb", help="Provider fallback order.")
    parser.add_argument("--openbb-provider", type=str, default="yfinance", help="Provider name to pass through OpenBB adapter.")
    parser.add_argument("--dte-min", type=int, default=30, help="Global minimum DTE to include.")
    parser.add_argument("--dte-max", type=int, default=120, help="Global maximum DTE to include.")
    parser.add_argument("--risk-free-rate", type=float, default=0.045, help="Annual risk-free rate.")
    parser.add_argument("--data-mode", type=str, default="delayed_eod", choices=["delayed_eod", "best_effort_intraday"])
    parser.add_argument("--best-edge-min", type=float, default=60.0)
    parser.add_argument("--best-conf-min", type=float, default=0.70)
    parser.add_argument("--retention-keep-last", type=int, default=30)
    parser.add_argument("--learning-enabled", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--strategies", type=str, default=DEFAULT_STRATEGIES, help="Comma-separated strategy identifiers.")
    parser.add_argument("--top", type=int, default=20, help="Rows to print from ranked output.")
    return parser


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers_csv:
        return load_tickers_from_csv(args.tickers_csv)
    if args.tickers:
        return parse_tickers(args.tickers)
    return ["SPY"]


def _scan_config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        provider_order=parse_csv_values(args.provider_order, lowercase=True),
        openbb_provider=args.openbb_provider,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        enabled_strategies=parse_csv_values(args.strategies, lowercase=True),
        risk_free_rate=args.risk_free_rate,
        data_mode=args.data_mode,
        best_trade_min_edge=args.best_edge_min,
        best_trade_min_confidence=args.best_conf_min,
        retention_keep_last=args.retention_keep_last,
        learning_enabled=(args.learning_enabled == "true"),
    )


def main_scan(argv: list[str] | None = None) -> int:
    """CLI command for scans."""
    parser = _scan_parser()
    args = parser.parse_args(argv)
    tickers = _resolve_tickers(args)
    if not tickers:
        print("No tickers provided.")
        return 1

    config = _scan_config_from_args(args)
    result = run_scan(tickers, config=config)

    print(f"Run ID: {result.run_id} | Label: {result.run_label}")
    if result.best_trade.empty:
        print("Best Trade: No trade today (did not pass score/confidence gates).")
    else:
        row = result.best_trade.iloc[0]
        print(
            "Best Trade: "
            f"{row.get('symbol')} {row.get('strategy')} "
            f"score={row.get('edge_score')} conf={row.get('edge_score_confidence')}"
        )
    if result.ranked_trades.empty:
        print("No ranked trades were generated.")
        print(f"Surface metrics rows: {len(result.surface_metrics)}")
        return 0

    display_cols = [
        "symbol",
        "strategy",
        "dte",
        "net_premium_per_share",
        "max_loss",
        "max_gain",
        "edge_score",
        "edge_score_confidence",
        "scenario_ev",
        "scenario_worst",
        "scenario_best",
        "premium_constraint",
    ]
    available_cols = [col for col in display_cols if col in result.ranked_trades.columns]
    preview = result.ranked_trades[available_cols].head(args.top)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(preview.to_string(index=False))
    print(f"Artifacts: {config.run_dir}")
    return 0


def main_dashboard(argv: list[str] | None = None) -> int:
    """Launch Streamlit dashboard."""
    _ = argv
    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(command)


def main_runs(argv: list[str] | None = None) -> int:
    """Run history management command."""
    parser = argparse.ArgumentParser(description="Manage scan outputs.")
    sub = parser.add_subparsers(dest="command", required=True)
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=20)
    del_cmd = sub.add_parser("delete")
    del_cmd.add_argument("--run-id", required=True)
    prune_cmd = sub.add_parser("prune")
    prune_cmd.add_argument("--keep-last", type=int, required=True)
    args = parser.parse_args(argv)

    cfg = EngineConfig()
    if args.command == "list":
        runs = list_runs(cfg.output_dir, limit=args.limit)
        if runs.empty:
            print("No runs found.")
            return 0
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(runs.to_string(index=False))
        return 0
    if args.command == "delete":
        ok = delete_run(cfg.output_dir, args.run_id)
        print(f"Delete {args.run_id}: {'done' if ok else 'not found'}")
        return 0
    if args.command == "prune":
        stale = prune_runs(cfg.output_dir, args.keep_last)
        print(f"Pruned {len(stale)} runs.")
        return 0
    return 1


def main_autopilot(argv: list[str] | None = None) -> int:
    """Run autopilot once or start scheduler."""
    parser = argparse.ArgumentParser(description="Autopilot scheduler.")
    parser.add_argument("--time", default="16:20", help="Daily trigger time HH:MM.")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--daemon", action="store_true", help="Run blocking daily scheduler.")
    parser.add_argument("--data-mode", default="delayed_eod", choices=["delayed_eod", "best_effort_intraday"])
    args = parser.parse_args(argv)

    cfg = EngineConfig(
        autopilot_enabled=True,
        autopilot_time=args.time,
        autopilot_timezone=args.timezone,
        autopilot_ticker=args.ticker.upper(),
        data_mode=args.data_mode,
    )
    if args.daemon:
        auto_cfg = AutopilotConfig(time_str=args.time, timezone=args.timezone, ticker=args.ticker.upper())
        start_daily_scheduler(auto_cfg, cfg)
        return 0
    result = run_autopilot_once(cfg)
    print(f"Autopilot run complete: {result.run_id} ({result.run_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_scan())
