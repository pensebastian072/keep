"""Tests for chat agent safety and action routing."""

from __future__ import annotations

import pandas as pd

from options_surface_edge_engine.agent.chat_agent import ChatAgent


def _tools() -> dict:
    return {
        "run_scan": lambda ticker: {"run_id": "r1", "ticker": ticker},
        "set_param": lambda key, value: f"{key}={value}",
        "show_top_trade": lambda: pd.DataFrame([{"trade_id": "T1"}]),
        "list_outputs": lambda: pd.DataFrame([{"run_id": "r1"}]),
        "delete_output": lambda run_id: run_id == "r1",
        "explain_trade": lambda trade_id: f"explain {trade_id}",
    }


def test_delete_requires_confirmation() -> None:
    agent = ChatAgent()
    pending = {}
    first = agent.handle("delete output r1", tools=_tools(), pending=pending)
    assert first["requires_confirmation"] is True
    token = first["token"]
    second = agent.handle(f"confirm {token}", tools=_tools(), pending=pending)
    assert second["ok"] is True
    assert "done" in second["message"]


def test_invalid_confirmation_token_rejected() -> None:
    agent = ChatAgent()
    pending = {}
    result = agent.handle("confirm BADTOKEN", tools=_tools(), pending=pending)
    assert result["ok"] is False


def test_run_ticker_action() -> None:
    agent = ChatAgent()
    pending = {}
    result = agent.handle("run ticker AAPL", tools=_tools(), pending=pending)
    assert result["ok"] is True
