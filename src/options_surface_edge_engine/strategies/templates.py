"""Strategy and leg templates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(slots=True)
class OptionLeg:
    """Single option leg."""

    symbol: str
    option_type: str
    position: str
    expiration: date
    dte: int
    strike: float
    quantity: int
    entry_price: float | None
    implied_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    contract_symbol: str | None = None

    @property
    def signed_quantity(self) -> int:
        """Return signed quantity for P&L aggregation."""
        sign = 1 if self.position == "long" else -1
        return sign * self.quantity

    def to_dict(self) -> dict[str, Any]:
        """Serialize leg to dictionary."""
        payload = asdict(self)
        payload["expiration"] = self.expiration.isoformat()
        payload["signed_quantity"] = self.signed_quantity
        return payload


@dataclass(slots=True)
class CandidateTrade:
    """Candidate strategy trade."""

    trade_id: str
    symbol: str
    strategy: str
    regime: str
    iv_regime: str
    spot: float | None
    dte: int | None
    legs: list[OptionLeg] = field(default_factory=list)
    net_premium_per_share: float | None = None
    max_loss: float | None = None
    max_gain: float | None = None
    breakeven_low: float | None = None
    breakeven_high: float | None = None
    premium_constraint: bool = True
    pop_proxy: float | None = None
    net_delta: float | None = None
    net_vega: float | None = None
    liquidity_proxy: float | None = None
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        """Serialize candidate trade to record."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "regime": self.regime,
            "iv_regime": self.iv_regime,
            "spot": self.spot,
            "dte": self.dte,
            "legs": [leg.to_dict() for leg in self.legs],
            "legs_text": " | ".join(
                f"{leg.position} {leg.option_type} {leg.strike:g} {leg.expiration.isoformat()}"
                for leg in self.legs
            ),
            "net_premium_per_share": self.net_premium_per_share,
            "max_loss": self.max_loss,
            "max_gain": self.max_gain,
            "breakeven_low": self.breakeven_low,
            "breakeven_high": self.breakeven_high,
            "premium_constraint": self.premium_constraint,
            "pop_proxy": self.pop_proxy,
            "net_delta": self.net_delta,
            "net_vega": self.net_vega,
            "liquidity_proxy": self.liquidity_proxy,
            "notes": self.notes,
        }
