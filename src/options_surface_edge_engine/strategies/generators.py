"""Strategy candidate generation."""

from __future__ import annotations

from datetime import date
from math import inf, log1p
from typing import Any

import pandas as pd

from options_surface_edge_engine.strategies.templates import CandidateTrade, OptionLeg
from options_surface_edge_engine.utils.config import EngineConfig


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_price(row: pd.Series) -> float | None:
    mark = _as_float(row.get("mark"))
    if mark is not None:
        return mark
    bid = _as_float(row.get("bid"))
    ask = _as_float(row.get("ask"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return _as_float(row.get("last_trade_price"))


def _expiration_to_date(value: Any) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return date.today()
    return parsed.date()


def _trade_liquidity(legs: list[OptionLeg]) -> float | None:
    if not legs:
        return None
    scores = []
    for leg in legs:
        spread_score = None
        if leg.bid is not None and leg.ask is not None:
            mid = leg.mark if leg.mark not in [None, 0] else (leg.bid + leg.ask) / 2.0
            if mid and mid > 0:
                spread_pct = max((leg.ask - leg.bid) / mid, 0.0)
                spread_score = max(0.0, 1.0 - min(spread_pct / 0.25, 1.0))
        vol_score = min(log1p(max(leg.volume or 0.0, 0.0)) / log1p(1000.0), 1.0)
        oi_score = min(log1p(max(leg.open_interest or 0.0, 0.0)) / log1p(10000.0), 1.0)
        components = [value for value in [spread_score, vol_score, oi_score] if value is not None]
        if components:
            scores.append(sum(components) / len(components))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _make_leg(symbol: str, row: pd.Series, position: str, quantity: int = 1) -> OptionLeg:
    return OptionLeg(
        symbol=symbol,
        option_type=str(row["option_type"]),
        position=position,
        expiration=_expiration_to_date(row.get("expiration")),
        dte=int(_as_float(row.get("dte")) or 0),
        strike=float(_as_float(row.get("strike")) or 0.0),
        quantity=quantity,
        entry_price=_resolve_price(row),
        implied_volatility=_as_float(row.get("implied_volatility")),
        delta=_as_float(row.get("delta")),
        gamma=_as_float(row.get("gamma")),
        theta=_as_float(row.get("theta")),
        vega=_as_float(row.get("vega")),
        bid=_as_float(row.get("bid")),
        ask=_as_float(row.get("ask")),
        mark=_as_float(row.get("mark")),
        volume=_as_float(row.get("volume")),
        open_interest=_as_float(row.get("open_interest")),
        contract_symbol=str(row.get("contract_symbol")) if row.get("contract_symbol") is not None else None,
    )


def _pick_by_delta(frame: pd.DataFrame, target_delta: float, option_type: str, spot: float | None) -> pd.Series | None:
    side = frame[frame["option_type"] == option_type].copy()
    if side.empty:
        return None
    if side["delta"].notna().any():
        idx = (pd.to_numeric(side["delta"], errors="coerce") - target_delta).abs().idxmin()
        return side.loc[idx]
    if spot is None:
        return side.iloc[len(side) // 2]
    if option_type == "call":
        side["distance"] = (side["strike"] - spot).abs()
    else:
        side["distance"] = (spot - side["strike"]).abs()
    return side.sort_values("distance").iloc[0]


def _nearest_expiry_window(chain: pd.DataFrame, min_dte: int, max_dte: int, target_dte: int, config: EngineConfig) -> pd.DataFrame:
    lower = max(min_dte, config.dte_min)
    upper = min(max_dte, config.dte_max)
    if lower > upper:
        return pd.DataFrame()
    window = chain[(chain["dte"] >= lower) & (chain["dte"] <= upper)].copy()
    if window.empty:
        return pd.DataFrame()
    expiry_dte = window.groupby("expiration")["dte"].median().sort_values()
    chosen_expiry = (expiry_dte - target_dte).abs().idxmin()
    return window[window["expiration"] == chosen_expiry].copy()


def _calc_net_delta(legs: list[OptionLeg]) -> float | None:
    values = [leg.signed_quantity * leg.delta for leg in legs if leg.delta is not None]
    if not values:
        return None
    return float(sum(values))


def _calc_net_vega(legs: list[OptionLeg]) -> float | None:
    values = [leg.signed_quantity * leg.vega for leg in legs if leg.vega is not None]
    if not values:
        return None
    return float(sum(values))


def _classify_iv_regime(symbol_metrics: dict[str, Any]) -> str:
    iv_anchor = _as_float(symbol_metrics.get("iv_30"))
    if iv_anchor is None:
        return "unknown"
    if iv_anchor < 0.20:
        return "low"
    if iv_anchor <= 0.40:
        return "moderate"
    return "high"


def _safe_trade(
    trade: CandidateTrade,
    premium_per_share_limit: float,
) -> CandidateTrade:
    trade.net_delta = _calc_net_delta(trade.legs)
    trade.net_vega = _calc_net_vega(trade.legs)
    trade.liquidity_proxy = _trade_liquidity(trade.legs)
    net_premium = trade.net_premium_per_share
    trade.premium_constraint = bool(net_premium is not None and abs(net_premium) <= premium_per_share_limit)
    if net_premium is None:
        trade.premium_constraint = False
    return trade


def _build_put_credit(
    symbol: str,
    chain: pd.DataFrame,
    spot: float | None,
    regime: str,
    iv_regime: str,
    config: EngineConfig,
    trade_id: str,
) -> CandidateTrade | None:
    expiry = _nearest_expiry_window(chain, 30, 60, 45, config)
    if expiry.empty:
        return None
    short_put = _pick_by_delta(expiry, -0.30, "put", spot)
    long_put = _pick_by_delta(expiry, -0.15, "put", spot)
    if short_put is None or long_put is None:
        return None
    if float(long_put["strike"]) >= float(short_put["strike"]):
        candidates = expiry[(expiry["option_type"] == "put") & (expiry["strike"] < float(short_put["strike"]))]
        if candidates.empty:
            return None
        long_put = candidates.loc[candidates["strike"].idxmax()]
    short_leg = _make_leg(symbol, short_put, "short")
    long_leg = _make_leg(symbol, long_put, "long")
    if short_leg.entry_price is None or long_leg.entry_price is None:
        return None
    credit = short_leg.entry_price - long_leg.entry_price
    width = max(short_leg.strike - long_leg.strike, 0.0)
    max_loss = max((width - credit) * config.contract_multiplier, 0.0)
    max_gain = max(credit * config.contract_multiplier, 0.0)
    pop_proxy = max(0.0, min(1.0, 1.0 - abs(short_leg.delta or 0.30)))
    return _safe_trade(
        CandidateTrade(
            trade_id=trade_id,
            symbol=symbol,
            strategy="put_credit_spread",
            regime=regime,
            iv_regime=iv_regime,
            spot=spot,
            dte=short_leg.dte,
            legs=[short_leg, long_leg],
            net_premium_per_share=credit,
            max_loss=max_loss,
            max_gain=max_gain,
            breakeven_low=short_leg.strike - credit,
            breakeven_high=None,
            pop_proxy=pop_proxy,
        ),
        config.premium_per_share_limit,
    )


def _build_call_debit(
    symbol: str,
    chain: pd.DataFrame,
    spot: float | None,
    regime: str,
    iv_regime: str,
    config: EngineConfig,
    trade_id: str,
) -> CandidateTrade | None:
    expiry = _nearest_expiry_window(chain, 45, 120, 75, config)
    if expiry.empty:
        return None
    long_call = _pick_by_delta(expiry, 0.40, "call", spot)
    short_call = _pick_by_delta(expiry, 0.25, "call", spot)
    if long_call is None or short_call is None:
        return None
    if float(long_call["strike"]) >= float(short_call["strike"]):
        candidates = expiry[(expiry["option_type"] == "call") & (expiry["strike"] > float(long_call["strike"]))]
        if candidates.empty:
            return None
        short_call = candidates.sort_values("strike").iloc[0]

    long_leg = _make_leg(symbol, long_call, "long")
    short_leg = _make_leg(symbol, short_call, "short")
    if long_leg.entry_price is None or short_leg.entry_price is None:
        return None
    debit = long_leg.entry_price - short_leg.entry_price
    width = max(short_leg.strike - long_leg.strike, 0.0)
    max_loss = max(debit * config.contract_multiplier, 0.0)
    max_gain = max((width - debit) * config.contract_multiplier, 0.0)
    pop_proxy = max(0.0, min(1.0, abs(short_leg.delta or 0.25)))
    return _safe_trade(
        CandidateTrade(
            trade_id=trade_id,
            symbol=symbol,
            strategy="call_debit_spread",
            regime=regime,
            iv_regime=iv_regime,
            spot=spot,
            dte=long_leg.dte,
            legs=[long_leg, short_leg],
            net_premium_per_share=debit,
            max_loss=max_loss,
            max_gain=max_gain,
            breakeven_low=long_leg.strike + debit,
            breakeven_high=None,
            pop_proxy=pop_proxy,
        ),
        config.premium_per_share_limit,
    )


def _build_calendar(
    symbol: str,
    chain: pd.DataFrame,
    spot: float | None,
    regime: str,
    iv_regime: str,
    config: EngineConfig,
    trade_id: str,
) -> CandidateTrade | None:
    near = _nearest_expiry_window(chain, 30, 45, 37, config)
    far = _nearest_expiry_window(chain, 90, 120, 105, config)
    if near.empty or far.empty:
        return None
    opt_type = "put" if regime == "bearish" else "call"
    near_side = near[near["option_type"] == opt_type]
    far_side = far[far["option_type"] == opt_type]
    if near_side.empty or far_side.empty:
        return None
    if spot is None:
        spot_series = pd.to_numeric(chain["underlying_price"], errors="coerce").dropna()
        if spot_series.empty:
            return None
        spot = float(spot_series.iloc[0])
    atm_near = near_side.loc[(near_side["strike"] - float(spot)).abs().idxmin()]
    strike = float(atm_near["strike"])
    far_match = far_side.loc[(far_side["strike"] - strike).abs().idxmin()]
    short_leg = _make_leg(symbol, atm_near, "short")
    long_leg = _make_leg(symbol, far_match, "long")
    if short_leg.entry_price is None or long_leg.entry_price is None:
        return None
    debit = long_leg.entry_price - short_leg.entry_price
    return _safe_trade(
        CandidateTrade(
            trade_id=trade_id,
            symbol=symbol,
            strategy="calendar_spread",
            regime=regime,
            iv_regime=iv_regime,
            spot=spot,
            dte=long_leg.dte,
            legs=[short_leg, long_leg],
            net_premium_per_share=debit,
            max_loss=max(debit * config.contract_multiplier, 0.0),
            max_gain=None,
            breakeven_low=None,
            breakeven_high=None,
            pop_proxy=0.50,
            notes="Calendar spread max gain approximated as unknown.",
        ),
        config.premium_per_share_limit,
    )


def _build_iron_condor(
    symbol: str,
    chain: pd.DataFrame,
    spot: float | None,
    regime: str,
    iv_regime: str,
    config: EngineConfig,
    trade_id: str,
) -> CandidateTrade | None:
    expiry = _nearest_expiry_window(chain, 30, 60, 45, config)
    if expiry.empty:
        return None
    short_put = _pick_by_delta(expiry, -0.20, "put", spot)
    long_put = _pick_by_delta(expiry, -0.10, "put", spot)
    short_call = _pick_by_delta(expiry, 0.20, "call", spot)
    long_call = _pick_by_delta(expiry, 0.10, "call", spot)
    if any(x is None for x in [short_put, long_put, short_call, long_call]):
        return None

    short_put_leg = _make_leg(symbol, short_put, "short")
    long_put_leg = _make_leg(symbol, long_put, "long")
    short_call_leg = _make_leg(symbol, short_call, "short")
    long_call_leg = _make_leg(symbol, long_call, "long")
    legs = [short_put_leg, long_put_leg, short_call_leg, long_call_leg]
    if any(leg.entry_price is None for leg in legs):
        return None

    if not (long_put_leg.strike < short_put_leg.strike < short_call_leg.strike < long_call_leg.strike):
        return None

    credit = (
        (short_put_leg.entry_price or 0.0)
        + (short_call_leg.entry_price or 0.0)
        - (long_put_leg.entry_price or 0.0)
        - (long_call_leg.entry_price or 0.0)
    )
    width_put = short_put_leg.strike - long_put_leg.strike
    width_call = long_call_leg.strike - short_call_leg.strike
    width = max(width_put, width_call)
    max_loss = max((width - credit) * config.contract_multiplier, 0.0)
    max_gain = max(credit * config.contract_multiplier, 0.0)
    pop_proxy = max(0.0, min(1.0, 1.0 - (abs(short_put_leg.delta or 0.2) + abs(short_call_leg.delta or 0.2))))
    return _safe_trade(
        CandidateTrade(
            trade_id=trade_id,
            symbol=symbol,
            strategy="iron_condor",
            regime=regime,
            iv_regime=iv_regime,
            spot=spot,
            dte=short_put_leg.dte,
            legs=legs,
            net_premium_per_share=credit,
            max_loss=max_loss,
            max_gain=max_gain,
            breakeven_low=short_put_leg.strike - credit,
            breakeven_high=short_call_leg.strike + credit,
            pop_proxy=pop_proxy,
        ),
        config.premium_per_share_limit,
    )


def _build_long_strangle(
    symbol: str,
    chain: pd.DataFrame,
    spot: float | None,
    regime: str,
    iv_regime: str,
    config: EngineConfig,
    trade_id: str,
) -> CandidateTrade | None:
    expiry = _nearest_expiry_window(chain, 60, 120, 90, config)
    if expiry.empty:
        return None
    long_put = _pick_by_delta(expiry, -0.20, "put", spot)
    long_call = _pick_by_delta(expiry, 0.20, "call", spot)
    if long_put is None or long_call is None:
        return None
    put_leg = _make_leg(symbol, long_put, "long")
    call_leg = _make_leg(symbol, long_call, "long")
    if put_leg.entry_price is None or call_leg.entry_price is None:
        return None
    debit = put_leg.entry_price + call_leg.entry_price
    pop_proxy = max(0.0, min(1.0, abs(put_leg.delta or 0.2) + abs(call_leg.delta or 0.2)))
    return _safe_trade(
        CandidateTrade(
            trade_id=trade_id,
            symbol=symbol,
            strategy="long_strangle",
            regime=regime,
            iv_regime=iv_regime,
            spot=spot,
            dte=max(put_leg.dte, call_leg.dte),
            legs=[put_leg, call_leg],
            net_premium_per_share=debit,
            max_loss=max(debit * config.contract_multiplier, 0.0),
            max_gain=inf,
            breakeven_low=put_leg.strike - debit,
            breakeven_high=call_leg.strike + debit,
            pop_proxy=pop_proxy,
        ),
        config.premium_per_share_limit,
    )


def generate_candidate_trades(
    symbol: str,
    chain: pd.DataFrame,
    symbol_metrics: dict[str, Any],
    trend_regime: str,
    config: EngineConfig,
) -> list[CandidateTrade]:
    """Generate strategy candidates for a single symbol."""
    if chain.empty:
        return []

    iv_regime = _classify_iv_regime(symbol_metrics)
    skew_25d = _as_float(symbol_metrics.get("skew_25d"))
    term_slope = _as_float(symbol_metrics.get("term_slope"))
    expansion_proxy = _as_float(symbol_metrics.get("expansion_proxy")) or 0.0
    spot = _as_float(symbol_metrics.get("spot"))
    if spot is None and chain["underlying_price"].notna().any():
        spot = float(chain["underlying_price"].dropna().iloc[0])

    candidates: list[CandidateTrade] = []
    trade_count = 0

    def next_id(name: str) -> str:
        nonlocal trade_count
        trade_count += 1
        return f"{symbol}_{name}_{trade_count:03d}"

    if (
        "put_credit_spread" in config.enabled_strategies
        and iv_regime == "high"
        and (skew_25d is not None and skew_25d > 0.04)
        and trend_regime in {"bullish", "neutral"}
    ):
        trade = _build_put_credit(symbol, chain, spot, trend_regime, iv_regime, config, next_id("put_credit"))
        if trade is not None:
            candidates.append(trade)

    if "call_debit_spread" in config.enabled_strategies and iv_regime == "moderate" and trend_regime == "bullish":
        trade = _build_call_debit(symbol, chain, spot, trend_regime, iv_regime, config, next_id("call_debit"))
        if trade is not None:
            candidates.append(trade)

    if "calendar_spread" in config.enabled_strategies and (term_slope is not None and term_slope > 0.02):
        trade = _build_calendar(symbol, chain, spot, trend_regime, iv_regime, config, next_id("calendar"))
        if trade is not None:
            candidates.append(trade)

    if "iron_condor" in config.enabled_strategies and iv_regime == "high" and trend_regime == "neutral":
        trade = _build_iron_condor(symbol, chain, spot, trend_regime, iv_regime, config, next_id("iron_condor"))
        if trade is not None:
            candidates.append(trade)

    if "long_strangle" in config.enabled_strategies and iv_regime == "low" and expansion_proxy > 0.015:
        trade = _build_long_strangle(symbol, chain, spot, trend_regime, iv_regime, config, next_id("long_strangle"))
        if trade is not None:
            candidates.append(trade)

    return candidates
