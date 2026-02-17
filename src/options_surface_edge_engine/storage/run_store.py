"""Run registry and retention manager."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REGISTRY_COLUMNS = [
    "run_id",
    "run_label",
    "created_at",
    "tickers",
    "ticker_count",
    "data_mode",
    "status",
    "best_trade_id",
    "best_trade_symbol",
    "best_trade_strategy",
    "best_trade_score",
    "best_trade_confidence",
]


@dataclass(slots=True)
class RunMetadata:
    """Metadata for one scan run."""

    run_id: str
    run_label: str
    created_at: str
    tickers: list[str]
    ticker_count: int
    data_mode: str
    status: str = "completed"
    best_trade_id: str | None = None
    best_trade_symbol: str | None = None
    best_trade_strategy: str | None = None
    best_trade_score: float | None = None
    best_trade_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dict."""
        payload = asdict(self)
        payload["tickers"] = ",".join(self.tickers)
        return payload


def _registry_path(output_dir: Path) -> Path:
    return output_dir / "run_registry.parquet"


def _run_dir(output_dir: Path, run_id: str) -> Path:
    return output_dir / "runs" / run_id


def _load_registry(output_dir: Path) -> pd.DataFrame:
    path = _registry_path(output_dir)
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    frame = pd.read_parquet(path)
    for column in REGISTRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[REGISTRY_COLUMNS]
    return frame


def _save_registry(output_dir: Path, frame: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _registry_path(output_dir)
    frame.to_parquet(path, index=False)


def next_run_label(output_dir: Path, prefix: str = "Output") -> str:
    """Generate next run label from registry."""
    registry = _load_registry(output_dir)
    if registry.empty or "run_label" not in registry.columns:
        return f"{prefix}-0001"
    labels = registry["run_label"].dropna().astype(str).tolist()
    numbers: list[int] = []
    for label in labels:
        if not label.startswith(f"{prefix}-"):
            continue
        suffix = label.split("-")[-1]
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_value = (max(numbers) + 1) if numbers else 1
    return f"{prefix}-{next_value:04d}"


def register_run(output_dir: Path, meta: RunMetadata) -> None:
    """Register or update run metadata and write run meta file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir(output_dir, meta.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(output_dir)
    row = pd.DataFrame([meta.to_dict()])
    if not registry.empty and (registry["run_id"] == meta.run_id).any():
        registry = registry[registry["run_id"] != meta.run_id]
    if registry.empty:
        registry = row
    else:
        registry = pd.DataFrame(registry.to_dict(orient="records") + row.to_dict(orient="records"))
    registry = registry.sort_values("created_at").reset_index(drop=True)
    _save_registry(output_dir, registry)

    meta_path = run_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(meta), file, indent=2)


def list_runs(output_dir: Path, limit: int | None = None) -> pd.DataFrame:
    """List runs from registry."""
    registry = _load_registry(output_dir)
    if registry.empty:
        return registry
    registry = registry.sort_values("created_at", ascending=False).reset_index(drop=True)
    if limit is not None:
        return registry.head(limit).copy()
    return registry


def delete_run(output_dir: Path, run_id: str) -> bool:
    """Delete run folder and remove registry row."""
    deleted = False
    registry = _load_registry(output_dir)
    if not registry.empty and (registry["run_id"] == run_id).any():
        registry = registry[registry["run_id"] != run_id].reset_index(drop=True)
        _save_registry(output_dir, registry)
        deleted = True

    run_dir = _run_dir(output_dir, run_id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        deleted = True
    return deleted


def prune_runs(output_dir: Path, keep_last: int) -> list[str]:
    """Keep newest runs, delete older run artifacts."""
    if keep_last <= 0:
        keep_last = 1
    registry = list_runs(output_dir)
    if registry.empty or len(registry) <= keep_last:
        return []
    stale = registry.iloc[keep_last:].copy()
    stale_ids = stale["run_id"].dropna().astype(str).tolist()
    for run_id in stale_ids:
        delete_run(output_dir, run_id)
    return stale_ids


def build_run_metadata(
    run_id: str,
    run_label: str,
    tickers: list[str],
    data_mode: str,
    best_trade: pd.DataFrame | None = None,
) -> RunMetadata:
    """Create metadata object from run components."""
    best_id = None
    best_symbol = None
    best_strategy = None
    best_score = None
    best_conf = None
    if best_trade is not None and not best_trade.empty:
        row = best_trade.iloc[0]
        best_id = row.get("trade_id")
        best_symbol = row.get("symbol")
        best_strategy = row.get("strategy")
        best_score = row.get("edge_score")
        best_conf = row.get("edge_score_confidence")
    return RunMetadata(
        run_id=run_id,
        run_label=run_label,
        created_at=datetime.now(UTC).isoformat(),
        tickers=tickers,
        ticker_count=len(tickers),
        data_mode=data_mode,
        best_trade_id=str(best_id) if best_id is not None else None,
        best_trade_symbol=str(best_symbol) if best_symbol is not None else None,
        best_trade_strategy=str(best_strategy) if best_strategy is not None else None,
        best_trade_score=float(best_score) if best_score is not None and pd.notna(best_score) else None,
        best_trade_confidence=float(best_conf) if best_conf is not None and pd.notna(best_conf) else None,
    )
