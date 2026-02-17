"""Streamlit dashboard for options surface edge engine."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from options_surface_edge_engine.agent.chat_agent import ChatAgent
from options_surface_edge_engine.learning.journal import read_journal
from options_surface_edge_engine.pipeline import ScanResult, run_scan
from options_surface_edge_engine.storage.run_store import delete_run, list_runs
from options_surface_edge_engine.utils import config as config_utils

EngineConfig = config_utils.EngineConfig
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

ALL_STRATEGIES = [
    "put_credit_spread",
    "call_debit_spread",
    "calendar_spread",
    "iron_condor",
    "long_strangle",
]


def _init_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_pending" not in st.session_state:
        st.session_state.chat_pending = {}
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "runtime_params" not in st.session_state:
        st.session_state.runtime_params = {
            "provider_order": "yfinance,openbb",
            "openbb_provider": "yfinance",
            "data_mode": "delayed_eod",
            "dte_min": 30,
            "dte_max": 120,
            "risk_free_rate": 0.045,
            "best_edge_min": 60.0,
            "best_conf_min": 0.70,
            "retention_keep_last": 30,
            "run_label_prefix": "Output",
            "enabled_strategies": ALL_STRATEGIES.copy(),
        }


def _build_config(
    provider_order: str,
    openbb_provider: str,
    data_mode: str,
    dte_min: int,
    dte_max: int,
    risk_free_rate: float,
    best_edge_min: float,
    best_conf_min: float,
    retention_keep_last: int,
    run_label_prefix: str,
    enabled_strategies: list[str],
) -> EngineConfig:
    return EngineConfig(
        provider_order=parse_csv_values(provider_order, lowercase=True),
        openbb_provider=openbb_provider,
        data_mode=data_mode,  # type: ignore[arg-type]
        dte_min=dte_min,
        dte_max=dte_max,
        risk_free_rate=float(risk_free_rate),
        best_trade_min_edge=float(best_edge_min),
        best_trade_min_confidence=float(best_conf_min),
        retention_keep_last=int(retention_keep_last),
        run_label_prefix=run_label_prefix,
        enabled_strategies=enabled_strategies,
    )


def _execute_scan(tickers: list[str], cfg: EngineConfig) -> ScanResult:
    with st.spinner(f"Running scan for {', '.join(tickers)}..."):
        result = run_scan(tickers, config=cfg)
    st.session_state.latest_result = result
    return result


def _render_best_trade(result: ScanResult) -> None:
    st.subheader("Best Trade (Gated Suggestion)")
    if result.best_trade.empty:
        st.info("No trade today. No candidate passed score/confidence gate.")
        return
    row = result.best_trade.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", str(row.get("symbol")))
    c2.metric("Strategy", str(row.get("strategy")))
    c3.metric("Edge Score", f"{row.get('edge_score'):.2f}" if pd.notna(row.get("edge_score")) else "N/A")
    c4.metric(
        "Confidence",
        f"{row.get('edge_score_confidence'):.2f}" if pd.notna(row.get("edge_score_confidence")) else "N/A",
    )
    st.caption(f"Run {result.run_label} ({result.run_id})")


def _render_data_quality(result: ScanResult) -> None:
    st.subheader("Data Quality")
    if result.data_quality.empty:
        st.info("No data quality rows.")
        return
    dq = result.data_quality.copy()
    st.dataframe(dq, use_container_width=True, hide_index=True)
    first = dq.iloc[0]
    st.caption(
        f"Mode: {first.get('data_mode_used')} | "
        f"Quote Date: {first.get('quote_date')} | "
        f"Staleness (min): {first.get('staleness_minutes')}"
    )


def _render_ranked_and_scenarios(result: ScanResult) -> None:
    st.subheader("Ranked Ideas")
    if result.ranked_trades.empty:
        st.warning("No ranked ideas for this run.")
        return
    cols = [
        "trade_id",
        "symbol",
        "strategy",
        "dte",
        "edge_score",
        "edge_score_confidence",
        "scenario_ev",
        "scenario_worst",
        "scenario_best",
        "premium_constraint",
    ]
    show = result.ranked_trades[[c for c in cols if c in result.ranked_trades.columns]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Ranked CSV",
        data=result.ranked_trades.to_csv(index=False).encode("utf-8"),
        file_name=f"ranked_trades_{result.run_id}.csv",
        mime="text/csv",
    )

    st.subheader("Scenario Explorer")
    if result.scenario_table.empty:
        st.info("No scenarios generated.")
        return
    st.download_button(
        "Download Scenarios CSV",
        data=result.scenario_table.to_csv(index=False).encode("utf-8"),
        file_name=f"scenarios_{result.run_id}.csv",
        mime="text/csv",
    )
    top_ids = result.ranked_trades["trade_id"].dropna().astype(str).head(5).tolist()
    for trade_id in top_ids:
        rows = result.scenario_table[result.scenario_table["trade_id"] == trade_id].copy()
        if rows.empty:
            continue
        with st.expander(f"{trade_id}"):
            base_iv = rows[rows["iv_shock"] == 0.0]
            if not base_iv.empty:
                heat = base_iv.pivot(index="days_forward", columns="spot_shock", values="pnl")
                st.dataframe(heat, use_container_width=True)
            st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_run_history() -> None:
    st.subheader("Outputs")
    cfg = EngineConfig()
    runs = list_runs(cfg.output_dir, limit=30)
    if runs.empty:
        st.info("No outputs saved yet.")
        return
    for _, row in runs.iterrows():
        run_id = str(row["run_id"])
        run_label = str(row.get("run_label"))
        line = f"{run_label} | {run_id} | {row.get('created_at')}"
        with st.container(border=True):
            st.caption(line)
            st.write(
                f"Ticker(s): {row.get('tickers')} | "
                f"Best: {row.get('best_trade_symbol')} {row.get('best_trade_strategy')} "
                f"(score={row.get('best_trade_score')}, conf={row.get('best_trade_confidence')})"
            )
            if st.button("Delete", key=f"del_{run_id}", type="secondary"):
                delete_run(cfg.output_dir, run_id)
                st.success(f"Deleted {run_id}")
                st.rerun()


def _render_learning_diagnostics() -> None:
    st.subheader("Learning Diagnostics")
    cfg = EngineConfig()
    journal = read_journal(cfg.output_dir)
    if journal.empty:
        st.info("No prediction journal entries yet.")
        return
    evaluated = journal[journal["status"] == "evaluated"].copy()
    pending = journal[journal["status"] == "pending"].copy()
    st.caption(f"Journal rows: {len(journal)} | Evaluated: {len(evaluated)} | Pending: {len(pending)}")
    if not evaluated.empty:
        win_rate = (evaluated["outcome"] == "right").mean()
        st.metric("Win Rate", f"{win_rate:.2%}")
        by_strategy = (
            evaluated.groupby(["strategy", "outcome"]).size().unstack(fill_value=0).reset_index()
            if "strategy" in evaluated.columns
            else pd.DataFrame()
        )
        if not by_strategy.empty:
            st.dataframe(by_strategy, use_container_width=True, hide_index=True)
        detail_cols = [
            "entry_date",
            "target_date",
            "symbol",
            "strategy",
            "trade_id",
            "realized_proxy_pnl",
            "outcome",
            "reason_codes",
        ]
        st.dataframe(
            evaluated[[c for c in detail_cols if c in evaluated.columns]].sort_values("entry_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    weights_path = cfg.output_dir / "learning" / "weights.json"
    if weights_path.exists():
        st.caption("Current tuned weights")
        st.code(weights_path.read_text(encoding="utf-8"), language="json")


def _get_trade_explanation(result: ScanResult | None, trade_id: str) -> str:
    if result is None or result.ranked_trades.empty:
        return "No run results loaded."
    rows = result.ranked_trades[result.ranked_trades["trade_id"].astype(str) == trade_id]
    if rows.empty:
        return f"Trade {trade_id} not found in current run."
    row = rows.iloc[0]
    return (
        f"{row.get('symbol')} {row.get('strategy')} | edge={row.get('edge_score')} | "
        f"conf={row.get('edge_score_confidence')} | EV={row.get('scenario_ev')} | "
        f"Worst={row.get('scenario_worst')} | Best={row.get('scenario_best')}"
    )


def _render_chat_panel(
    agent: ChatAgent,
    ticker: str,
    cfg_builder: callable,
) -> None:
    st.subheader("Assistant Chat")
    st.caption("Commands: run ticker AAPL, show top trade, list outputs, explain trade <id>, delete output <run_id>")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if isinstance(message.get("table"), pd.DataFrame):
                st.dataframe(message["table"], use_container_width=True, hide_index=True)

    prompt = st.chat_input("Ask assistant...")
    if not prompt:
        return
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    def _tool_run_scan(ticker_value: str) -> dict[str, Any]:
        cfg = cfg_builder()
        result = _execute_scan([ticker_value], cfg)
        return {"run_id": result.run_id, "run_label": result.run_label}

    def _tool_set_param(key: str, value: Any) -> str:
        params = st.session_state.runtime_params
        if key not in params:
            return f"Unknown setting: {key}"
        params[key] = value
        return f"Updated {key}={value}"

    def _tool_show_top_trade() -> pd.DataFrame:
        result = st.session_state.latest_result
        if result is None:
            return pd.DataFrame()
        return result.best_trade

    def _tool_list_outputs() -> pd.DataFrame:
        cfg = EngineConfig()
        return list_runs(cfg.output_dir, limit=20)

    def _tool_delete_output(run_id: str) -> bool:
        cfg = EngineConfig()
        return delete_run(cfg.output_dir, run_id)

    def _tool_explain_trade(trade_id: str) -> str:
        return _get_trade_explanation(st.session_state.latest_result, trade_id)

    tools = {
        "run_scan": _tool_run_scan,
        "set_param": _tool_set_param,
        "show_top_trade": _tool_show_top_trade,
        "list_outputs": _tool_list_outputs,
        "delete_output": _tool_delete_output,
        "explain_trade": _tool_explain_trade,
    }
    response = agent.handle(prompt, tools=tools, pending=st.session_state.chat_pending)

    payload: dict[str, Any] = {"role": "assistant", "content": response.get("message", "")}
    result_data = response.get("result")
    if isinstance(result_data, pd.DataFrame):
        payload["table"] = result_data
    elif isinstance(result_data, dict):
        payload["content"] = f"{payload['content']}\n\n{result_data}"
    st.session_state.chat_messages.append(payload)
    st.rerun()


def render_app() -> None:
    """Render Streamlit dashboard."""
    _init_state()
    st.set_page_config(page_title="Options Surface Edge Engine", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("Options Surface Edge Engine")
    st.caption("Single-ticker command center with delayed EOD default, run history, autopilot-ready outputs, and assistant chat.")

    params = st.session_state.runtime_params
    command_col, center_col, history_col = st.columns([1.1, 2.2, 1.2], gap="large")
    with command_col:
        st.subheader("Command Center")
        ticker = st.text_input("Ticker", value="SPY", max_chars=10).upper().strip()
        advanced_multi = st.checkbox("Advanced: multi-ticker input", value=False)
        multi_input = st.text_input("Tickers CSV list", value="SPY,QQQ,GLD,NVDA") if advanced_multi else ""
        params["provider_order"] = st.text_input("Provider order", value=str(params["provider_order"]))
        params["openbb_provider"] = st.text_input("OpenBB provider", value=str(params["openbb_provider"]))
        params["data_mode"] = st.selectbox("Data mode", options=["delayed_eod", "best_effort_intraday"], index=0 if params["data_mode"] == "delayed_eod" else 1)
        dte_range = st.slider("DTE range", min_value=7, max_value=365, value=(int(params["dte_min"]), int(params["dte_max"])))
        params["dte_min"], params["dte_max"] = dte_range
        params["risk_free_rate"] = st.number_input("Risk-free rate", min_value=0.0, max_value=0.25, value=float(params["risk_free_rate"]), step=0.005)
        params["best_edge_min"] = st.number_input("Min edge score", min_value=0.0, max_value=100.0, value=float(params["best_edge_min"]), step=1.0)
        params["best_conf_min"] = st.number_input("Min confidence", min_value=0.0, max_value=1.0, value=float(params["best_conf_min"]), step=0.01)
        params["retention_keep_last"] = st.number_input("Retention keep last", min_value=1, max_value=200, value=int(params["retention_keep_last"]), step=1)
        params["run_label_prefix"] = st.text_input("Run label prefix", value=str(params["run_label_prefix"]))
        params["enabled_strategies"] = st.multiselect("Strategies", options=ALL_STRATEGIES, default=list(params["enabled_strategies"]))

        cfg_builder = lambda: _build_config(
            provider_order=str(params["provider_order"]),
            openbb_provider=str(params["openbb_provider"]),
            data_mode=str(params["data_mode"]),
            dte_min=int(params["dte_min"]),
            dte_max=int(params["dte_max"]),
            risk_free_rate=float(params["risk_free_rate"]),
            best_edge_min=float(params["best_edge_min"]),
            best_conf_min=float(params["best_conf_min"]),
            retention_keep_last=int(params["retention_keep_last"]),
            run_label_prefix=str(params["run_label_prefix"]),
            enabled_strategies=list(params["enabled_strategies"]),
        )

        next_label = list_runs(EngineConfig().output_dir, limit=1)
        next_hint = "Output-0001"
        if not next_label.empty:
            latest = str(next_label.iloc[0]["run_label"])
            next_hint = latest
        st.info(f"Next run will be saved as next {params.get('run_label_prefix', 'Output')}-#### (latest: {next_hint})")

        if st.button("Run Scan", type="primary"):
            cfg = cfg_builder()
            tickers = parse_tickers(multi_input) if advanced_multi else [ticker]
            _execute_scan(tickers, cfg)
            st.rerun()

        agent = ChatAgent(model=EngineConfig().openai_model, api_key_env=EngineConfig().openai_api_key_env)
        _render_chat_panel(agent, ticker=ticker, cfg_builder=cfg_builder)

    with center_col:
        result = st.session_state.latest_result
        if result is None:
            st.info("Run a scan from Command Center to see results.")
        else:
            _render_best_trade(result)
            _render_data_quality(result)
            _render_ranked_and_scenarios(result)

    with history_col:
        _render_run_history()
        _render_learning_diagnostics()


if __name__ == "__main__":
    render_app()
