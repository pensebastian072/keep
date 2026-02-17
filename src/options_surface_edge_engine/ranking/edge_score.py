"""Edge score computation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from options_surface_edge_engine.utils.config import LiquidityConfig, ScoringWeights


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(value, high))


def _ev_percentile(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="float64")
    ranked = series.rank(pct=True)
    return ranked * 100.0


def compute_edge_scores(
    frame: pd.DataFrame,
    weights: ScoringWeights,
    liquidity_cfg: LiquidityConfig,
) -> pd.DataFrame:
    """Compute component and final edge scores."""
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    data["ev_score"] = _ev_percentile(pd.to_numeric(data.get("scenario_ev"), errors="coerce"))

    max_loss = pd.to_numeric(data.get("max_loss"), errors="coerce")
    data["max_loss_score"] = 100.0 * (1.0 - (max_loss / 5000.0).clip(lower=0.0, upper=1.0))

    pop_proxy = pd.to_numeric(data.get("pop_proxy"), errors="coerce")
    data["pop_score"] = (pop_proxy * 100.0).clip(lower=0.0, upper=100.0)

    net_vega = pd.to_numeric(data.get("net_vega"), errors="coerce")
    data["vega_score"] = 100.0 - ((net_vega.abs() / liquidity_cfg.vega_scale) * 100.0).clip(lower=0.0, upper=100.0)

    liquidity_proxy = pd.to_numeric(data.get("liquidity_proxy"), errors="coerce")
    data["liquidity_score"] = (liquidity_proxy * 100.0).clip(lower=0.0, upper=100.0)

    weight_map = weights.as_dict()
    component_cols = list(weight_map.keys())

    scores: list[float | None] = []
    confidence: list[float] = []
    for _, row in data.iterrows():
        weighted_sum = 0.0
        used_weight = 0.0
        for col, weight in weight_map.items():
            value = row.get(col)
            if value is None or pd.isna(value):
                continue
            weighted_sum += float(value) * weight
            used_weight += weight
        if used_weight == 0:
            scores.append(None)
            confidence.append(0.0)
        else:
            scores.append(_clamp(weighted_sum / used_weight))
            confidence.append(used_weight / 100.0)

    data["edge_score"] = scores
    data["edge_score_confidence"] = confidence

    for col in component_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["edge_score"] = pd.to_numeric(data["edge_score"], errors="coerce")
    data["edge_score_confidence"] = pd.to_numeric(data["edge_score_confidence"], errors="coerce")

    return data.sort_values(["edge_score", "scenario_ev"], ascending=[False, False], na_position="last").reset_index(drop=True)
