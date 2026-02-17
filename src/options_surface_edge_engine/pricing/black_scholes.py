"""Black-Scholes pricing and greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class Greeks:
    """Greeks container."""

    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None


def norm_cdf(x: float) -> float:
    """Normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def safe_time_to_expiry(dte: float | int | None) -> float | None:
    """Convert DTE into years."""
    if dte is None:
        return None
    try:
        dte_value = float(dte)
    except (TypeError, ValueError):
        return None
    if dte_value <= 0:
        return 1.0 / 365.0
    return dte_value / 365.0


def _d1_d2(spot: float, strike: float, t: float, rate: float, dividend: float, sigma: float) -> tuple[float, float]:
    sigma_t = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * sigma * sigma) * t) / sigma_t
    d2 = d1 - sigma_t
    return d1, d2


def bs_price(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
    dividend: float = 0.0,
) -> float:
    """Price an option with Black-Scholes."""
    opt = option_type.lower()
    if sigma <= 0 or t <= 0:
        intrinsic = max(spot - strike, 0.0) if opt == "call" else max(strike - spot, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(spot=spot, strike=strike, t=t, rate=rate, dividend=dividend, sigma=sigma)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend * t)
    if opt == "call":
        return spot * disc_q * norm_cdf(d1) - strike * disc_r * norm_cdf(d2)
    if opt == "put":
        return strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)
    raise ValueError(f"Unsupported option_type={option_type!r}")


def bs_greeks(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    sigma: float,
    option_type: str,
    dividend: float = 0.0,
) -> Greeks:
    """Compute Black-Scholes greeks."""
    if sigma <= 0 or t <= 0:
        intrinsic_delta = 1.0 if (option_type == "call" and spot > strike) else 0.0
        if option_type == "put":
            intrinsic_delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=intrinsic_delta, gamma=0.0, theta=0.0, vega=0.0)

    d1, d2 = _d1_d2(spot=spot, strike=strike, t=t, rate=rate, dividend=dividend, sigma=sigma)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-dividend * t)
    disc_r = math.exp(-rate * t)

    if option_type == "call":
        delta = disc_q * norm_cdf(d1)
        theta = (
            -(spot * disc_q * norm_pdf(d1) * sigma) / (2 * sqrt_t)
            - rate * strike * disc_r * norm_cdf(d2)
            + dividend * spot * disc_q * norm_cdf(d1)
        )
    elif option_type == "put":
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (
            -(spot * disc_q * norm_pdf(d1) * sigma) / (2 * sqrt_t)
            + rate * strike * disc_r * norm_cdf(-d2)
            - dividend * spot * disc_q * norm_cdf(-d1)
        )
    else:
        raise ValueError(f"Unsupported option_type={option_type!r}")

    gamma = (disc_q * norm_pdf(d1)) / (spot * sigma * sqrt_t)
    vega = spot * disc_q * norm_pdf(d1) * sqrt_t
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega)
