"""End-to-end scan pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from options_surface_edge_engine.data_providers.base import BaseOptionsProvider, ProviderFetchError, chain_is_complete
from options_surface_edge_engine.data_providers.openbb_provider import OpenBBProvider
from options_surface_edge_engine.data_providers.yfinance_provider import YFinanceProvider
from options_surface_edge_engine.features.surface_features import build_surface_features
from options_surface_edge_engine.learning.journal import evaluate_matured_predictions, read_journal, record_prediction
from options_surface_edge_engine.learning.tuner import load_weights, save_weights, tune_weights
from options_surface_edge_engine.pricing.black_scholes import bs_greeks, safe_time_to_expiry
from options_surface_edge_engine.pricing.scenario_engine import run_scenarios
from options_surface_edge_engine.ranking.edge_score import compute_edge_scores
from options_surface_edge_engine.ranking.selection import select_best_trade
from options_surface_edge_engine.storage.run_store import (
    build_run_metadata,
    next_run_label,
    prune_runs,
    register_run,
)
from options_surface_edge_engine.strategies.generators import generate_candidate_trades
from options_surface_edge_engine.strategies.templates import CandidateTrade
from options_surface_edge_engine.utils.config import EngineConfig
from options_surface_edge_engine.utils.logging import get_logger


@dataclass(slots=True)
class ScanResult:
    """Pipeline output container."""

    run_id: str
    run_label: str
    ranked_trades: pd.DataFrame
    scenario_table: pd.DataFrame
    surface_metrics: pd.DataFrame
    expiry_features: pd.DataFrame
    raw_candidates: pd.DataFrame
    best_trade: pd.DataFrame
    run_metadata: pd.DataFrame
    data_quality: pd.DataFrame


def _provider_registry(config: EngineConfig, logger_name: str = "osee") -> dict[str, BaseOptionsProvider]:
    logger = get_logger(logger_name)
    providers: dict[str, BaseOptionsProvider] = {"yfinance": YFinanceProvider()}
    try:
        providers["openbb"] = OpenBBProvider(openbb_provider=config.openbb_provider)
    except Exception as exc:
        logger.warning("OpenBB provider unavailable: %s", exc)
    return providers


def _first_non_null(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.iloc[0])


def _fetch_chain_with_fallback(
    symbol: str,
    providers: dict[str, BaseOptionsProvider],
    provider_order: list[str],
    logger_name: str = "osee",
) -> tuple[pd.DataFrame | None, str | None]:
    logger = get_logger(logger_name)
    partial: pd.DataFrame | None = None
    partial_name: str | None = None
    for provider_name in provider_order:
        provider = providers.get(provider_name)
        if provider is None:
            continue
        try:
            chain = provider.fetch_symbol_chain(symbol)
            if chain_is_complete(chain):
                return chain, provider_name
            if partial is None and not chain.empty:
                partial = chain
                partial_name = provider_name
            logger.warning("Provider %s returned incomplete chain for %s", provider_name, symbol)
        except ProviderFetchError as exc:
            logger.warning("Provider %s failed for %s: %s", provider_name, symbol, exc)
        except Exception as exc:
            logger.warning("Unexpected provider error %s for %s: %s", provider_name, symbol, exc)
    return partial, partial_name


def _fetch_spot_history(
    symbol: str,
    primary_provider: str | None,
    providers: dict[str, BaseOptionsProvider],
    config: EngineConfig,
    logger_name: str = "osee",
) -> pd.DataFrame:
    logger = get_logger(logger_name)
    attempts = []
    if primary_provider:
        attempts.append(primary_provider)
    attempts.extend([name for name in config.provider_order if name != primary_provider])
    for provider_name in attempts:
        provider = providers.get(provider_name)
        if provider is None:
            continue
        try:
            return provider.fetch_spot_history(symbol, lookback_days=config.trend_lookback_days)
        except Exception as exc:
            logger.warning("Spot history fetch failed (%s, %s): %s", symbol, provider_name, exc)
    return pd.DataFrame()


def _trend_regime(spot_history: pd.DataFrame) -> tuple[str, float | None]:
    if spot_history.empty:
        return "neutral", None
    close_col = "close" if "close" in spot_history.columns else "adj close" if "adj close" in spot_history.columns else None
    if close_col is None:
        return "neutral", None
    closes = pd.to_numeric(spot_history[close_col], errors="coerce").dropna()
    if closes.empty:
        return "neutral", None
    sma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else None
    sma60 = closes.rolling(60).mean().iloc[-1] if len(closes) >= 60 else None
    last_close = float(closes.iloc[-1])
    expansion_proxy = None
    if sma20 is not None and pd.notna(sma20) and sma20 != 0:
        expansion_proxy = abs(last_close / float(sma20) - 1.0)
    if sma20 is None or sma60 is None or pd.isna(sma20) or pd.isna(sma60):
        return "neutral", expansion_proxy
    if last_close > float(sma20) > float(sma60):
        return "bullish", expansion_proxy
    if last_close < float(sma20) < float(sma60):
        return "bearish", expansion_proxy
    return "neutral", expansion_proxy


def _fill_mark_price(chain: pd.DataFrame) -> pd.DataFrame:
    data = chain.copy()
    if "mark" not in data.columns:
        data["mark"] = pd.NA
    mark = pd.to_numeric(data["mark"], errors="coerce")
    bid = pd.to_numeric(data.get("bid"), errors="coerce")
    ask = pd.to_numeric(data.get("ask"), errors="coerce")
    last = pd.to_numeric(data.get("last_trade_price"), errors="coerce")
    mid = (bid + ask) / 2.0
    mark = mark.where(mark.notna(), mid)
    mark = mark.where(mark.notna(), last)
    data["mark"] = mark
    return data


def _stamp_data_mode(chain: pd.DataFrame, config: EngineConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = chain.copy()
    now = datetime.now(UTC)
    if config.data_mode == "delayed_eod":
        quote_date = date.today() - timedelta(days=1)
        quote_ts = datetime(quote_date.year, quote_date.month, quote_date.day, 21, 0, tzinfo=UTC)
        staleness = max((now - quote_ts).total_seconds() / 60.0, 0.0)
        data["quote_date"] = quote_date
        data["quote_ts"] = quote_ts
        data["staleness_minutes"] = staleness
        data["data_mode_used"] = "delayed_eod"
    else:
        qts = pd.to_datetime(data.get("quote_ts"), errors="coerce", utc=True)
        if qts.isna().all():
            qts = pd.Series([now] * len(data))
        staleness_series = (now - qts).dt.total_seconds() / 60.0
        data["quote_ts"] = qts
        data["quote_date"] = pd.to_datetime(qts, errors="coerce").dt.date
        data["staleness_minutes"] = staleness_series.clip(lower=0.0)
        data["data_mode_used"] = "best_effort_intraday"
    quality = {
        "quote_date": str(data["quote_date"].iloc[0]) if not data.empty else None,
        "staleness_minutes": float(pd.to_numeric(data["staleness_minutes"], errors="coerce").dropna().median()) if data["staleness_minutes"].notna().any() else None,
        "data_mode_used": str(data["data_mode_used"].iloc[0]) if not data.empty else config.data_mode,
    }
    return data, quality


def _enrich_greeks(chain: pd.DataFrame, config: EngineConfig) -> pd.DataFrame:
    data = chain.copy()
    for col in ["delta", "gamma", "theta", "vega"]:
        if col not in data.columns:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["implied_volatility"] = pd.to_numeric(data["implied_volatility"], errors="coerce")
    data["underlying_price"] = pd.to_numeric(data["underlying_price"], errors="coerce")
    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data["dte"] = pd.to_numeric(data["dte"], errors="coerce")

    needs = data[["delta", "gamma", "theta", "vega"]].isna().any(axis=1)
    for idx, row in data[needs].iterrows():
        option_type = str(row.get("option_type", "")).lower()
        sigma = row.get("implied_volatility")
        spot = row.get("underlying_price")
        strike = row.get("strike")
        dte = row.get("dte")
        if option_type not in {"call", "put"}:
            continue
        if pd.isna(sigma) or pd.isna(spot) or pd.isna(strike) or pd.isna(dte):
            continue
        t = safe_time_to_expiry(float(dte))
        if t is None:
            continue
        try:
            greeks = bs_greeks(
                spot=float(spot),
                strike=float(strike),
                t=float(t),
                rate=config.risk_free_rate,
                sigma=max(float(sigma), 0.01),
                option_type=option_type,
                dividend=config.dividend_yield,
            )
        except Exception:
            continue
        if pd.isna(data.at[idx, "delta"]):
            data.at[idx, "delta"] = greeks.delta
        if pd.isna(data.at[idx, "gamma"]):
            data.at[idx, "gamma"] = greeks.gamma
        if pd.isna(data.at[idx, "theta"]):
            data.at[idx, "theta"] = greeks.theta
        if pd.isna(data.at[idx, "vega"]):
            data.at[idx, "vega"] = greeks.vega
    return data


def _persist_chain(chain: pd.DataFrame, symbol: str, config: EngineConfig) -> None:
    symbol_upper = symbol.upper()
    run_path = config.run_dir / f"{symbol_upper}_chain.parquet"
    chain.to_parquet(run_path, index=False)
    if config.save_latest_chain:
        latest_path = config.output_dir / f"{symbol_upper}_chain.parquet"
        chain.to_parquet(latest_path, index=False)


def _persist_tables(
    ranked: pd.DataFrame,
    scenarios: pd.DataFrame,
    surfaces: pd.DataFrame,
    expiry_features: pd.DataFrame,
    best_trade: pd.DataFrame,
    run_meta: pd.DataFrame,
    data_quality: pd.DataFrame,
    config: EngineConfig,
) -> None:
    ranked.to_csv(config.run_dir / "ranked_trades.csv", index=False)
    scenarios.to_csv(config.run_dir / "scenario_table.csv", index=False)
    surfaces.to_csv(config.run_dir / "surface_metrics.csv", index=False)
    expiry_features.to_csv(config.run_dir / "expiry_features.csv", index=False)
    best_trade.to_csv(config.run_dir / "best_trade.csv", index=False)
    run_meta.to_csv(config.run_dir / "run_metadata.csv", index=False)
    data_quality.to_csv(config.run_dir / "data_quality.csv", index=False)
    if not scenarios.empty:
        scenarios.to_parquet(config.run_dir / "scenarios.parquet", index=False)


def _maybe_tune_weights(config: EngineConfig, logger_name: str = "osee") -> None:
    if not config.learning_enabled or not config.tune_weights_weekly:
        return
    if datetime.now().weekday() != 0:
        return
    logger = get_logger(logger_name)
    journal = read_journal(config.output_dir)
    if journal.empty:
        return
    evaluated = journal[journal["status"] == "evaluated"].copy()
    if len(evaluated) < 20:
        return
    tuned = tune_weights(evaluated, config.scoring)
    config.scoring = tuned
    weights_path = config.output_dir / "learning" / "weights.json"
    save_weights(weights_path, tuned)
    logger.info("Tuned score weights saved to %s", weights_path)


def run_scan(tickers: list[str], config: EngineConfig | None = None) -> ScanResult:
    """Run full options surface scan and return ranked candidates."""
    cfg = config or EngineConfig()
    cfg.ensure_dirs()
    weights_path = cfg.output_dir / "learning" / "weights.json"
    if cfg.learning_enabled:
        cfg.scoring = load_weights(weights_path, cfg.scoring)

    run_label = next_run_label(cfg.output_dir, prefix=cfg.run_label_prefix)
    logger = get_logger("osee", log_file=cfg.run_dir / "scan.log")
    symbols = [ticker.strip().upper() for ticker in tickers if ticker and str(ticker).strip()]
    symbols = list(dict.fromkeys(symbols))[: cfg.max_tickers]
    providers = _provider_registry(cfg)

    all_candidates: list[CandidateTrade] = []
    surface_rows: list[dict[str, Any]] = []
    expiry_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        logger.info("Scanning symbol=%s", symbol)
        chain, provider_used = _fetch_chain_with_fallback(symbol, providers, cfg.provider_order)
        if chain is None or chain.empty:
            logger.warning("No chain available for %s", symbol)
            continue

        chain = _fill_mark_price(chain)
        chain, quality = _stamp_data_mode(chain, cfg)
        spot_history = _fetch_spot_history(symbol, provider_used, providers, cfg)
        latest_close = None
        if not spot_history.empty:
            close_col = "close" if "close" in spot_history.columns else "adj close" if "adj close" in spot_history.columns else None
            if close_col:
                latest_close = _first_non_null(spot_history[close_col])
        if latest_close is not None and chain["underlying_price"].isna().any():
            chain.loc[chain["underlying_price"].isna(), "underlying_price"] = latest_close

        chain = _enrich_greeks(chain, cfg)
        _persist_chain(chain, symbol, cfg)

        expiry_features, summary = build_surface_features(chain)
        if not expiry_features.empty:
            expiry_frames.append(expiry_features.assign(provider_used=provider_used))
        summary["provider_used"] = provider_used
        trend_regime, expansion_proxy = _trend_regime(spot_history)
        summary["trend_regime"] = trend_regime
        summary["expansion_proxy"] = expansion_proxy
        surface_rows.append(summary)

        quality_rows.append(
            {
                "symbol": symbol,
                "provider_used": provider_used,
                "data_mode_used": quality.get("data_mode_used"),
                "quote_date": quality.get("quote_date"),
                "staleness_minutes": quality.get("staleness_minutes"),
                "chain_rows": int(len(chain)),
                "iv_available": bool(pd.to_numeric(chain.get("implied_volatility"), errors="coerce").notna().any()),
                "greeks_available": bool(pd.to_numeric(chain.get("delta"), errors="coerce").notna().any()),
            }
        )

        candidates = generate_candidate_trades(
            symbol=symbol,
            chain=chain,
            symbol_metrics=summary,
            trend_regime=trend_regime,
            config=cfg,
        )
        all_candidates.extend(candidates)
        logger.info("Generated %d candidates for %s", len(candidates), symbol)

    raw_candidates = pd.DataFrame([candidate.to_record() for candidate in all_candidates])
    scenario_table = pd.DataFrame()
    ranked = raw_candidates.copy()

    if all_candidates:
        scenario_table, scenario_summary = run_scenarios(all_candidates, cfg)
        ranked = ranked.merge(scenario_summary, on="trade_id", how="left")
        ranked = compute_edge_scores(ranked, cfg.scoring, cfg.liquidity)

    best_trade_row = select_best_trade(
        ranked=ranked if not ranked.empty else pd.DataFrame(),
        min_edge=cfg.best_trade_min_edge,
        min_conf=cfg.best_trade_min_confidence,
    )
    best_trade = pd.DataFrame([best_trade_row.to_dict()]) if best_trade_row is not None else pd.DataFrame()

    surface_metrics = pd.DataFrame(surface_rows)
    expiry_features_frame = pd.concat(expiry_frames, ignore_index=True) if expiry_frames else pd.DataFrame()
    data_quality = pd.DataFrame(quality_rows)

    run_meta_obj = build_run_metadata(
        run_id=cfg.run_id,
        run_label=run_label,
        tickers=symbols,
        data_mode=cfg.data_mode,
        best_trade=best_trade,
    )
    register_run(cfg.output_dir, run_meta_obj)
    stale_ids = prune_runs(cfg.output_dir, cfg.retention_keep_last)
    if stale_ids:
        logger.info("Pruned stale runs: %s", ", ".join(stale_ids))
    run_metadata = pd.DataFrame([run_meta_obj.to_dict()])

    _persist_tables(
        ranked=ranked if not ranked.empty else pd.DataFrame(columns=["trade_id"]),
        scenarios=scenario_table if not scenario_table.empty else pd.DataFrame(columns=["trade_id"]),
        surfaces=surface_metrics if not surface_metrics.empty else pd.DataFrame(columns=["symbol"]),
        expiry_features=expiry_features_frame if not expiry_features_frame.empty else pd.DataFrame(columns=["symbol"]),
        best_trade=best_trade if not best_trade.empty else pd.DataFrame(columns=["trade_id"]),
        run_meta=run_metadata,
        data_quality=data_quality if not data_quality.empty else pd.DataFrame(columns=["symbol"]),
        config=cfg,
    )

    if cfg.learning_enabled and not best_trade.empty:
        record_prediction(
            output_dir=cfg.output_dir,
            run_id=cfg.run_id,
            trade_row=best_trade.iloc[0],
            context={"run_label": run_label},
            horizon_days=cfg.learning_horizon_days,
        )
    if cfg.learning_enabled:
        evaluated = evaluate_matured_predictions(output_dir=cfg.output_dir)
        if not evaluated.empty:
            logger.info("Evaluated %d matured predictions", len(evaluated))
    _maybe_tune_weights(cfg)

    return ScanResult(
        run_id=cfg.run_id,
        run_label=run_label,
        ranked_trades=ranked,
        scenario_table=scenario_table,
        surface_metrics=surface_metrics,
        expiry_features=expiry_features_frame,
        raw_candidates=raw_candidates,
        best_trade=best_trade,
        run_metadata=run_metadata,
        data_quality=data_quality,
    )


def run_scan_from_csv(csv_path: str | Path, config: EngineConfig | None = None) -> ScanResult:
    """Run scan from CSV file path."""
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return run_scan([], config=config)
    tickers = frame["ticker"].tolist() if "ticker" in frame.columns else frame.iloc[:, 0].tolist()
    return run_scan([str(x) for x in tickers], config=config)
