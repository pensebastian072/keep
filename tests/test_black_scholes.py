"""Unit tests for Black-Scholes pricing and greeks."""

from __future__ import annotations

from options_surface_edge_engine.pricing.black_scholes import bs_greeks, bs_price


def test_black_scholes_known_values() -> None:
    call = bs_price(spot=100, strike=100, t=1.0, rate=0.05, sigma=0.2, option_type="call")
    put = bs_price(spot=100, strike=100, t=1.0, rate=0.05, sigma=0.2, option_type="put")
    assert abs(call - 10.4506) < 0.02
    assert abs(put - 5.5735) < 0.02


def test_put_call_parity() -> None:
    s = 100.0
    k = 110.0
    t = 0.5
    r = 0.03
    sigma = 0.25
    call = bs_price(spot=s, strike=k, t=t, rate=r, sigma=sigma, option_type="call")
    put = bs_price(spot=s, strike=k, t=t, rate=r, sigma=sigma, option_type="put")
    lhs = call - put
    rhs = s - k * (2.718281828459045 ** (-r * t))
    assert abs(lhs - rhs) < 0.05


def test_greek_signs() -> None:
    call_greeks = bs_greeks(spot=100, strike=100, t=1.0, rate=0.05, sigma=0.2, option_type="call")
    put_greeks = bs_greeks(spot=100, strike=100, t=1.0, rate=0.05, sigma=0.2, option_type="put")
    assert call_greeks.delta is not None and call_greeks.delta > 0
    assert put_greeks.delta is not None and put_greeks.delta < 0
    assert call_greeks.gamma is not None and call_greeks.gamma > 0
    assert put_greeks.gamma is not None and put_greeks.gamma > 0
    assert call_greeks.vega is not None and call_greeks.vega > 0
    assert put_greeks.vega is not None and put_greeks.vega > 0


def test_delta_finite_difference() -> None:
    base = bs_price(spot=100, strike=100, t=0.5, rate=0.02, sigma=0.3, option_type="call")
    bumped = bs_price(spot=100.1, strike=100, t=0.5, rate=0.02, sigma=0.3, option_type="call")
    fd_delta = (bumped - base) / 0.1
    greeks = bs_greeks(spot=100, strike=100, t=0.5, rate=0.02, sigma=0.3, option_type="call")
    assert greeks.delta is not None
    assert abs(greeks.delta - fd_delta) < 0.03
