"""Weight tuning from evaluated journal outcomes."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

from options_surface_edge_engine.utils.config import ScoringWeights

COMPONENTS = ["ev_score", "max_loss_score", "pop_score", "vega_score", "liquidity_score"]


def _weights_to_dict(weights: ScoringWeights) -> dict[str, float]:
    return weights.as_dict()


def _dict_to_weights(values: dict[str, float]) -> ScoringWeights:
    return ScoringWeights(
        ev=float(values["ev_score"]),
        max_loss=float(values["max_loss_score"]),
        pop=float(values["pop_score"]),
        vega=float(values["vega_score"]),
        liquidity=float(values["liquidity_score"]),
    )


def _objective(history: pd.DataFrame, weights: dict[str, float]) -> float:
    if history.empty:
        return -1e9
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return -1e9

    scored = history.copy()
    for col in COMPONENTS:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored = scored.dropna(subset=COMPONENTS + ["outcome"])
    if scored.empty:
        return -1e9

    score = 0.0
    for _, row in scored.iterrows():
        weighted = sum(float(row[col]) * weights[col] for col in COMPONENTS) / total_weight
        y = 1.0 if str(row["outcome"]).lower() == "right" else 0.0
        score += (weighted if y == 1.0 else (100.0 - weighted))
    return score / len(scored)


def tune_weights(history_df: pd.DataFrame, current_weights: ScoringWeights) -> ScoringWeights:
    """Tune weights with local constrained grid search."""
    base = _weights_to_dict(current_weights)
    if history_df.empty:
        return current_weights

    candidate_ranges: dict[str, list[float]] = {}
    for key, value in base.items():
        values = sorted(set([max(5.0, value - 10.0), max(5.0, value - 5.0), value, min(50.0, value + 5.0), min(50.0, value + 10.0)]))
        candidate_ranges[key] = values

    best = base
    best_score = _objective(history_df, base)
    for ev, max_loss, pop, vega, liquidity in product(
        candidate_ranges["ev_score"],
        candidate_ranges["max_loss_score"],
        candidate_ranges["pop_score"],
        candidate_ranges["vega_score"],
        candidate_ranges["liquidity_score"],
    ):
        total = ev + max_loss + pop + vega + liquidity
        if total <= 0:
            continue
        normalized = {
            "ev_score": ev * 100.0 / total,
            "max_loss_score": max_loss * 100.0 / total,
            "pop_score": pop * 100.0 / total,
            "vega_score": vega * 100.0 / total,
            "liquidity_score": liquidity * 100.0 / total,
        }
        if any(v < 5.0 or v > 50.0 for v in normalized.values()):
            continue
        score = _objective(history_df, normalized)
        if score > best_score:
            best_score = score
            best = normalized

    return _dict_to_weights(best)


def load_weights(path: Path, fallback: ScoringWeights) -> ScoringWeights:
    """Load tuned weights from JSON if available."""
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    required = {"ev_score", "max_loss_score", "pop_score", "vega_score", "liquidity_score"}
    if not required.issubset(set(payload.keys())):
        return fallback
    return _dict_to_weights({k: float(payload[k]) for k in required})


def save_weights(path: Path, weights: ScoringWeights) -> None:
    """Persist tuned weights to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = weights.as_dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
