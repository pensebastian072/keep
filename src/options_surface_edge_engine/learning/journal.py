"""Prediction journal and outcome evaluation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

JOURNAL_COLUMNS = [
    "prediction_id",
    "run_id",
    "run_label",
    "created_at",
    "entry_date",
    "target_date",
    "symbol",
    "strategy",
    "trade_id",
    "entry_spot",
    "edge_score",
    "edge_score_confidence",
    "scenario_ev",
    "pop_proxy",
    "net_delta",
    "net_vega",
    "max_loss",
    "premium_constraint",
    "ev_score",
    "max_loss_score",
    "pop_score",
    "vega_score",
    "liquidity_score",
    "status",
    "evaluated_at",
    "realized_spot",
    "realized_proxy_pnl",
    "outcome",
    "reason_codes",
]


def _journal_path(output_dir: Path) -> Path:
    return output_dir / "learning" / "trade_journal.parquet"


def _load_journal(output_dir: Path) -> pd.DataFrame:
    path = _journal_path(output_dir)
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    frame = pd.read_parquet(path)
    for column in JOURNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[JOURNAL_COLUMNS]


def read_journal(output_dir: Path) -> pd.DataFrame:
    """Public reader for journal."""
    return _load_journal(output_dir)


def _save_journal(output_dir: Path, frame: pd.DataFrame) -> None:
    path = _journal_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_date(value: Any) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return date.today()
    return parsed.date()


def record_prediction(
    output_dir: Path,
    run_id: str,
    trade_row: pd.Series | dict[str, Any],
    context: dict[str, Any] | None = None,
    horizon_days: int = 7,
) -> None:
    """Record best-trade prediction to journal."""
    if isinstance(trade_row, pd.Series):
        row = trade_row.to_dict()
    else:
        row = dict(trade_row)
    context = context or {}

    created_at = datetime.now(UTC)
    entry_date = created_at.date()
    target_date = entry_date + timedelta(days=horizon_days)
    symbol = str(row.get("symbol", "")).upper().strip()
    if not symbol:
        return

    prediction_id = f"{run_id}:{row.get('trade_id', 'unknown')}"
    journal = _load_journal(output_dir)
    if not journal.empty and (journal["prediction_id"] == prediction_id).any():
        return

    payload = {
        "prediction_id": prediction_id,
        "run_id": run_id,
        "run_label": context.get("run_label"),
        "created_at": created_at.isoformat(),
        "entry_date": entry_date.isoformat(),
        "target_date": target_date.isoformat(),
        "symbol": symbol,
        "strategy": row.get("strategy"),
        "trade_id": row.get("trade_id"),
        "entry_spot": _to_float(row.get("spot")),
        "edge_score": _to_float(row.get("edge_score")),
        "edge_score_confidence": _to_float(row.get("edge_score_confidence")),
        "scenario_ev": _to_float(row.get("scenario_ev")),
        "pop_proxy": _to_float(row.get("pop_proxy")),
        "net_delta": _to_float(row.get("net_delta")),
        "net_vega": _to_float(row.get("net_vega")),
        "max_loss": _to_float(row.get("max_loss")),
        "premium_constraint": bool(row.get("premium_constraint", True)),
        "ev_score": _to_float(row.get("ev_score")),
        "max_loss_score": _to_float(row.get("max_loss_score")),
        "pop_score": _to_float(row.get("pop_score")),
        "vega_score": _to_float(row.get("vega_score")),
        "liquidity_score": _to_float(row.get("liquidity_score")),
        "status": "pending",
        "evaluated_at": pd.NA,
        "realized_spot": pd.NA,
        "realized_proxy_pnl": pd.NA,
        "outcome": pd.NA,
        "reason_codes": pd.NA,
    }
    payload_frame = pd.DataFrame([payload])
    journal = payload_frame if journal.empty else pd.concat([journal, payload_frame], ignore_index=True)
    _save_journal(output_dir, journal)


def _latest_spot(symbol: str, as_of: date) -> float | None:
    try:
        from yfinance import Ticker
    except Exception:
        return None
    try:
        history = Ticker(symbol).history(period="1mo", interval="1d")
    except Exception:
        return None
    if history.empty:
        return None
    frame = history.reset_index()
    frame.columns = [str(c).lower() for c in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    filtered = frame[frame["date"] <= as_of]
    if filtered.empty:
        filtered = frame
    close = pd.to_numeric(filtered["close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _reason_codes(
    row: pd.Series,
    entry_spot: float,
    realized_spot: float,
    realized_proxy_pnl: float,
) -> str:
    reasons: list[str] = []
    net_delta = _to_float(row.get("net_delta"))
    if net_delta is not None:
        move = realized_spot - entry_spot
        if (net_delta > 0 and move < 0) or (net_delta < 0 and move > 0):
            reasons.append("spot_move_unfavorable")
    if entry_spot != 0 and abs(realized_spot - entry_spot) / entry_spot < 0.01:
        reasons.append("time_decay_unfavorable")
    net_vega = _to_float(row.get("net_vega"))
    if net_vega is not None and abs(net_vega) > 0.15:
        reasons.append("iv_move_unfavorable")
    if bool(row.get("premium_constraint")) is False:
        reasons.append("liquidity_slippage_proxy")
    if realized_proxy_pnl > 0:
        reasons = ["model_alignment"]
    if not reasons:
        reasons = ["model_uncertain"]
    return ",".join(reasons)


def evaluate_matured_predictions(
    output_dir: Path,
    as_of_date: date | datetime | str | None = None,
) -> pd.DataFrame:
    """Evaluate pending journal entries with matured horizon."""
    as_of = _to_date(as_of_date) if as_of_date is not None else date.today()
    journal = _load_journal(output_dir)
    if journal.empty:
        return journal

    journal["entry_date"] = pd.to_datetime(journal["entry_date"], errors="coerce").dt.date
    journal["target_date"] = pd.to_datetime(journal["target_date"], errors="coerce").dt.date
    pending = journal[(journal["status"] == "pending") & (journal["target_date"] <= as_of)]
    if pending.empty:
        return pd.DataFrame(columns=journal.columns)

    updates: list[dict[str, Any]] = []
    for _, row in pending.iterrows():
        symbol = str(row.get("symbol", "")).upper().strip()
        entry_spot = _to_float(row.get("entry_spot"))
        if not symbol or entry_spot is None:
            continue
        realized_spot = _latest_spot(symbol, as_of)
        if realized_spot is None:
            continue
        net_delta = _to_float(row.get("net_delta")) or 0.0
        net_vega = _to_float(row.get("net_vega")) or 0.0
        scenario_ev = _to_float(row.get("scenario_ev")) or 0.0
        spot_component = net_delta * (realized_spot - entry_spot) * 100.0
        vega_component = net_vega * 10.0
        realized_proxy_pnl = spot_component + vega_component + (scenario_ev * 0.1)
        outcome = "right" if realized_proxy_pnl > 0 else "wrong"
        reasons = _reason_codes(row, entry_spot, realized_spot, realized_proxy_pnl)
        updates.append(
            {
                "prediction_id": row["prediction_id"],
                "status": "evaluated",
                "evaluated_at": datetime.now(UTC).isoformat(),
                "realized_spot": realized_spot,
                "realized_proxy_pnl": realized_proxy_pnl,
                "outcome": outcome,
                "reason_codes": reasons,
            }
        )

    if not updates:
        return pd.DataFrame(columns=journal.columns)

    update_frame = pd.DataFrame(updates)
    merged = journal.merge(update_frame, on="prediction_id", how="left", suffixes=("", "_new"))
    for col in ["status", "evaluated_at", "realized_spot", "realized_proxy_pnl", "outcome", "reason_codes"]:
        new_col = f"{col}_new"
        merged[col] = merged[new_col].where(merged[new_col].notna(), merged[col])
        merged = merged.drop(columns=[new_col])
    _save_journal(output_dir, merged)
    return merged[merged["prediction_id"].isin(update_frame["prediction_id"])]
