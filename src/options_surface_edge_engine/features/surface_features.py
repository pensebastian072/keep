"""Options surface feature engineering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _nearest_value(values: Iterable[float], target: float) -> float | None:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return None
    return min(clean, key=lambda x: abs(x - target))


def _interpolate_iv(expiry_features: pd.DataFrame, target_dte: int) -> float | None:
    valid = expiry_features.dropna(subset=["dte", "atm_iv"]).copy()
    if valid.empty:
        return None
    valid = valid.sort_values("dte")
    if target_dte <= valid["dte"].min():
        return float(valid.iloc[0]["atm_iv"])
    if target_dte >= valid["dte"].max():
        return float(valid.iloc[-1]["atm_iv"])

    lower = valid[valid["dte"] <= target_dte].iloc[-1]
    upper = valid[valid["dte"] >= target_dte].iloc[0]
    dte_low = float(lower["dte"])
    dte_high = float(upper["dte"])
    if dte_low == dte_high:
        return float(lower["atm_iv"])
    weight = (target_dte - dte_low) / (dte_high - dte_low)
    return float(lower["atm_iv"] + weight * (upper["atm_iv"] - lower["atm_iv"]))


def _atm_iv_for_expiry(group: pd.DataFrame, spot: float | None) -> tuple[float | None, float | None]:
    if spot is None or pd.isna(spot):
        return None, None
    strike = _nearest_value(group["strike"], float(spot))
    if strike is None:
        return None, None
    calls = group[(group["option_type"] == "call") & (group["strike"] == strike)]
    puts = group[(group["option_type"] == "put") & (group["strike"] == strike)]
    call_iv = pd.to_numeric(calls["implied_volatility"], errors="coerce").dropna().mean()
    put_iv = pd.to_numeric(puts["implied_volatility"], errors="coerce").dropna().mean()
    iv_values = [v for v in [call_iv, put_iv] if pd.notna(v)]
    if not iv_values:
        return None, strike
    return float(sum(iv_values) / len(iv_values)), strike


def build_surface_features(chain: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute expiry-level and symbol-level surface metrics."""
    if chain.empty:
        return pd.DataFrame(), {}

    data = chain.copy()
    data["dte"] = pd.to_numeric(data["dte"], errors="coerce")
    data["implied_volatility"] = pd.to_numeric(data["implied_volatility"], errors="coerce")
    data["delta"] = pd.to_numeric(data["delta"], errors="coerce")
    data["gamma"] = pd.to_numeric(data["gamma"], errors="coerce")
    data["open_interest"] = pd.to_numeric(data["open_interest"], errors="coerce").fillna(0.0)
    data["strike"] = pd.to_numeric(data["strike"], errors="coerce")
    data = data.dropna(subset=["expiration", "strike"])
    if data.empty:
        return pd.DataFrame(), {}

    symbol = str(data["symbol"].dropna().iloc[0]) if data["symbol"].notna().any() else "UNKNOWN"
    spot = pd.to_numeric(data["underlying_price"], errors="coerce").dropna()
    spot_value = float(spot.iloc[0]) if not spot.empty else None

    expiry_rows: list[dict[str, Any]] = []
    for expiration, group in data.groupby("expiration", dropna=True):
        dte = pd.to_numeric(group["dte"], errors="coerce").dropna().median()
        atm_iv, atm_strike = _atm_iv_for_expiry(group, spot_value)

        calls = group[group["option_type"] == "call"].copy()
        puts = group[group["option_type"] == "put"].copy()
        calls = calls.dropna(subset=["implied_volatility"])
        puts = puts.dropna(subset=["implied_volatility"])

        skew_25d = None
        call_25 = calls.dropna(subset=["delta"])
        put_25 = puts.dropna(subset=["delta"])
        if not call_25.empty and not put_25.empty:
            call_idx = (call_25["delta"] - 0.25).abs().idxmin()
            put_idx = (put_25["delta"] + 0.25).abs().idxmin()
            call_iv = call_25.loc[call_idx, "implied_volatility"]
            put_iv = put_25.loc[put_idx, "implied_volatility"]
            if pd.notna(call_iv) and pd.notna(put_iv):
                skew_25d = float(put_iv - call_iv)

        put_curve = None
        call_curve = None
        if atm_iv is not None:
            put_band = puts[(puts["delta"] >= -0.35) & (puts["delta"] <= -0.15)]
            call_band = calls[(calls["delta"] >= 0.15) & (calls["delta"] <= 0.35)]
            if not put_band.empty:
                put_curve = float(put_band["implied_volatility"].mean() - atm_iv)
            if not call_band.empty:
                call_curve = float(call_band["implied_volatility"].mean() - atm_iv)

        gamma_zones = []
        if group["gamma"].notna().any():
            gamma_frame = group[["strike", "gamma", "open_interest"]].copy()
            gamma_frame["oi_weighted_gamma"] = gamma_frame["gamma"].abs() * gamma_frame["open_interest"]
            gamma_frame = gamma_frame.sort_values("oi_weighted_gamma", ascending=False).dropna(subset=["strike"])
            gamma_zones = gamma_frame.head(3)["strike"].astype(float).tolist()

        expiry_rows.append(
            {
                "symbol": symbol,
                "expiration": expiration,
                "dte": float(dte) if pd.notna(dte) else None,
                "spot": spot_value,
                "atm_strike": atm_strike,
                "atm_iv": atm_iv,
                "skew_25d": skew_25d,
                "put_curvature": put_curve,
                "call_curvature": call_curve,
                "gamma_zones": gamma_zones,
            }
        )

    expiry_features = pd.DataFrame(expiry_rows).sort_values("dte", na_position="last").reset_index(drop=True)
    if expiry_features.empty:
        return expiry_features, {}

    iv_30 = _interpolate_iv(expiry_features, 30)
    iv_90 = _interpolate_iv(expiry_features, 90)
    term_slope = None if iv_30 is None or iv_90 is None else float(iv_90 - iv_30)

    skew_window = expiry_features[(expiry_features["dte"] >= 30) & (expiry_features["dte"] <= 45)]
    skew_25d = None
    if not skew_window.empty and skew_window["skew_25d"].notna().any():
        skew_25d = float(skew_window["skew_25d"].dropna().mean())
    elif expiry_features["skew_25d"].notna().any():
        skew_25d = float(expiry_features["skew_25d"].dropna().iloc[0])

    put_curv = None
    call_curv = None
    if expiry_features["put_curvature"].notna().any():
        put_curv = float(expiry_features["put_curvature"].dropna().mean())
    if expiry_features["call_curvature"].notna().any():
        call_curv = float(expiry_features["call_curvature"].dropna().mean())

    gamma_all: list[float] = []
    for zones in expiry_features["gamma_zones"].tolist():
        if isinstance(zones, list):
            gamma_all.extend([float(x) for x in zones[:3]])
    gamma_top3 = []
    if gamma_all:
        gamma_top3 = sorted(gamma_all, key=lambda x: abs(x - (spot_value or x)))[:3]

    summary = {
        "symbol": symbol,
        "spot": spot_value,
        "iv_30": iv_30,
        "iv_90": iv_90,
        "term_slope": term_slope,
        "skew_25d": skew_25d,
        "put_curvature": put_curv,
        "call_curvature": call_curv,
        "gamma_zones": gamma_top3,
        "metric_available_iv_30": iv_30 is not None,
        "metric_available_iv_90": iv_90 is not None,
        "metric_available_term_slope": term_slope is not None,
        "metric_available_skew_25d": skew_25d is not None,
        "metric_available_put_curvature": put_curv is not None,
        "metric_available_call_curvature": call_curv is not None,
    }
    return expiry_features, summary
