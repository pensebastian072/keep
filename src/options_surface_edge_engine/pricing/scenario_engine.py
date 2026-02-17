"""Scenario repricing engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from options_surface_edge_engine.pricing.black_scholes import bs_price, safe_time_to_expiry
from options_surface_edge_engine.strategies.templates import CandidateTrade, OptionLeg
from options_surface_edge_engine.utils.config import EngineConfig


@dataclass(slots=True)
class ScenarioSummary:
    """Scenario summary for one trade."""

    trade_id: str
    scenario_ev: float | None
    scenario_worst: float | None
    scenario_best: float | None
    scenario_win_rate: float | None

    def to_record(self) -> dict[str, Any]:
        """Serialize summary."""
        return {
            "trade_id": self.trade_id,
            "scenario_ev": self.scenario_ev,
            "scenario_worst": self.scenario_worst,
            "scenario_best": self.scenario_best,
            "scenario_win_rate": self.scenario_win_rate,
        }


def _scenario_weight(spot_shock: float, iv_shock: float, days_forward: int) -> float:
    spot_weights = {-0.05: 0.15, -0.025: 0.20, 0.0: 0.30, 0.025: 0.20, 0.05: 0.15}
    iv_weights = {-0.20: 0.25, 0.0: 0.50, 0.20: 0.25}
    time_weights = {7: 0.60, 21: 0.40}
    return spot_weights.get(round(spot_shock, 3), 0.20) * iv_weights.get(round(iv_shock, 2), 0.33) * time_weights.get(days_forward, 0.50)


def _leg_pnl(
    leg: OptionLeg,
    spot: float,
    rate: float,
    dividend: float,
    spot_shock: float,
    iv_shock: float,
    days_forward: int,
    multiplier: int,
) -> float | None:
    if leg.entry_price is None:
        return None
    if leg.implied_volatility is None or leg.strike is None:
        return None
    t_years = safe_time_to_expiry(leg.dte - days_forward)
    if t_years is None:
        return None

    scenario_spot = spot * (1.0 + spot_shock)
    scenario_sigma = max(leg.implied_volatility * (1.0 + iv_shock), 0.01)
    scenario_price = bs_price(
        spot=scenario_spot,
        strike=leg.strike,
        t=t_years,
        rate=rate,
        sigma=scenario_sigma,
        option_type=leg.option_type,
        dividend=dividend,
    )
    signed_move = (scenario_price - leg.entry_price) * leg.signed_quantity * multiplier
    return float(signed_move)


def run_scenarios(candidates: list[CandidateTrade], config: EngineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run scenario grid for candidate trades."""
    scenario_rows: list[dict[str, Any]] = []
    summaries: list[ScenarioSummary] = []

    for trade in candidates:
        trade_spot = trade.spot
        if trade_spot is None:
            summaries.append(ScenarioSummary(trade_id=trade.trade_id, scenario_ev=None, scenario_worst=None, scenario_best=None, scenario_win_rate=None))
            continue

        weighted_total = 0.0
        total_weight = 0.0
        valid_pnls: list[float] = []

        for days_forward in config.scenario.days_forward:
            for spot_shock in config.scenario.spot_shocks:
                for iv_shock in config.scenario.iv_shocks:
                    leg_pnls = [
                        _leg_pnl(
                            leg=leg,
                            spot=trade_spot,
                            rate=config.risk_free_rate,
                            dividend=config.dividend_yield,
                            spot_shock=spot_shock,
                            iv_shock=iv_shock,
                            days_forward=days_forward,
                            multiplier=config.contract_multiplier,
                        )
                        for leg in trade.legs
                    ]
                    if any(value is None for value in leg_pnls):
                        trade_pnl = None
                        valid = False
                    else:
                        trade_pnl = float(sum(v for v in leg_pnls if v is not None))
                        valid = True
                        weight = _scenario_weight(spot_shock, iv_shock, days_forward)
                        weighted_total += trade_pnl * weight
                        total_weight += weight
                        valid_pnls.append(trade_pnl)

                    scenario_rows.append(
                        {
                            "trade_id": trade.trade_id,
                            "symbol": trade.symbol,
                            "strategy": trade.strategy,
                            "days_forward": days_forward,
                            "spot_shock": spot_shock,
                            "iv_shock": iv_shock,
                            "pnl": trade_pnl,
                            "scenario_valid": valid,
                        }
                    )

        if valid_pnls and total_weight > 0:
            ev = weighted_total / total_weight
            worst = min(valid_pnls)
            best = max(valid_pnls)
            win_rate = sum(1 for x in valid_pnls if x > 0) / len(valid_pnls)
            summaries.append(
                ScenarioSummary(
                    trade_id=trade.trade_id,
                    scenario_ev=float(ev),
                    scenario_worst=float(worst),
                    scenario_best=float(best),
                    scenario_win_rate=float(win_rate),
                )
            )
        else:
            summaries.append(
                ScenarioSummary(
                    trade_id=trade.trade_id,
                    scenario_ev=None,
                    scenario_worst=None,
                    scenario_best=None,
                    scenario_win_rate=None,
                )
            )

    return pd.DataFrame(scenario_rows), pd.DataFrame([summary.to_record() for summary in summaries])
