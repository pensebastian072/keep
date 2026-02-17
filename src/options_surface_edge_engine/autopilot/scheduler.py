"""Autopilot scheduler service."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from options_surface_edge_engine.pipeline import ScanResult, run_scan
from options_surface_edge_engine.utils.config import EngineConfig
from options_surface_edge_engine.utils.logging import get_logger


@dataclass(slots=True)
class AutopilotConfig:
    """Autopilot runtime configuration."""

    time_str: str = "16:20"
    timezone: str = "America/New_York"
    ticker: str = "SPY"
    output_dir: Path | None = None


def _state_path(config: EngineConfig) -> Path:
    return config.output_dir / "autopilot" / "autopilot_state.json"


def _write_state(config: EngineConfig, run_id: str, ticker: str) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_id": run_id,
        "last_ticker": ticker,
        "last_run_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_autopilot_once(config: EngineConfig) -> ScanResult:
    """Run one autopilot cycle."""
    logger = get_logger("osee.autopilot")
    ticker = config.autopilot_ticker.strip().upper() or "SPY"
    logger.info("Autopilot run started for ticker=%s", ticker)
    result = run_scan([ticker], config=config)
    _write_state(config, run_id=result.run_id, ticker=ticker)
    logger.info("Autopilot run completed run_id=%s", result.run_id)
    return result


def start_daily_scheduler(config: AutopilotConfig, engine_config: EngineConfig) -> None:
    """Start blocking daily scheduler."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("APScheduler is required for scheduler mode.") from exc

    logger = get_logger("osee.autopilot")
    hour, minute = config.time_str.split(":")
    scheduler = BlockingScheduler(timezone=config.timezone)

    def _job() -> None:
        local_config = replace(engine_config)
        local_config.autopilot_ticker = config.ticker
        run_autopilot_once(local_config)

    trigger = CronTrigger(hour=int(hour), minute=int(minute), timezone=config.timezone)
    scheduler.add_job(_job, trigger=trigger, id="osee_daily_autopilot", replace_existing=True)
    logger.info(
        "Starting autopilot scheduler ticker=%s time=%s timezone=%s",
        config.ticker,
        config.time_str,
        config.timezone,
    )
    scheduler.start()
