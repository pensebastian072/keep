"""Base provider interface and normalization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from typing import Final

import pandas as pd

STANDARD_CHAIN_COLUMNS: Final[list[str]] = [
    "symbol",
    "provider",
    "quote_ts",
    "quote_date",
    "staleness_minutes",
    "data_mode_used",
    "underlying_price",
    "expiration",
    "dte",
    "option_type",
    "strike",
    "contract_symbol",
    "bid",
    "ask",
    "mark",
    "last_trade_price",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "currency",
]


class ProviderFetchError(RuntimeError):
    """Raised when a provider cannot fetch data."""


class BaseOptionsProvider(ABC):
    """Provider contract for options and spot data."""

    @abstractmethod
    def provider_name(self) -> str:
        """Return provider short name."""

    @abstractmethod
    def fetch_symbol_chain(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain for one symbol."""

    @abstractmethod
    def fetch_spot_history(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch spot history for trend regime detection."""


def _coerce_expiration(values: pd.Series) -> pd.Series:
    """Convert expiration field to date."""
    return pd.to_datetime(values, errors="coerce").dt.date


def _compute_dte(expiration: pd.Series) -> pd.Series:
    """Compute DTE from expiration date."""
    today = date.today()
    return expiration.apply(lambda x: (x - today).days if isinstance(x, date) else pd.NA)


def ensure_standard_columns(frame: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
    """Normalize incoming provider frame into standard schema."""
    data = frame.copy()
    data.columns = [str(col) for col in data.columns]

    rename_map = {
        "contractSymbol": "contract_symbol",
        "lastPrice": "last_trade_price",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_volatility",
        "underlying_symbol": "symbol",
    }
    data = data.rename(columns=rename_map)

    for col in STANDARD_CHAIN_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data["symbol"] = data["symbol"].fillna(symbol.upper())
    data["provider"] = data["provider"].fillna(provider)
    data["quote_ts"] = data["quote_ts"].fillna(datetime.now(UTC))
    if "quote_date" not in data.columns:
        data["quote_date"] = pd.to_datetime(data["quote_ts"], errors="coerce").dt.date
    if "staleness_minutes" not in data.columns:
        data["staleness_minutes"] = pd.NA
    if "data_mode_used" not in data.columns:
        data["data_mode_used"] = pd.NA
    data["option_type"] = data["option_type"].astype("string").str.lower()
    data["expiration"] = _coerce_expiration(data["expiration"])
    if data["dte"].isna().all():
        data["dte"] = _compute_dte(data["expiration"])

    numeric_cols = [
        "underlying_price",
        "strike",
        "bid",
        "ask",
        "mark",
        "last_trade_price",
        "volume",
        "open_interest",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "dte",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data["mark"].isna().all():
        data["mark"] = (data["bid"] + data["ask"]) / 2.0
    data["mark"] = data["mark"].where(data["mark"].notna(), data["last_trade_price"])

    data = data[STANDARD_CHAIN_COLUMNS]
    data = data.sort_values(["symbol", "expiration", "option_type", "strike"], na_position="last").reset_index(drop=True)
    return data


def chain_is_complete(frame: pd.DataFrame) -> bool:
    """Return True when chain has enough fields to build tradable legs."""
    if frame.empty:
        return False
    required_cols = ["expiration", "strike", "option_type", "underlying_price"]
    for col in required_cols:
        if col not in frame.columns or frame[col].dropna().empty:
            return False
    has_pricing = (
        ("mark" in frame.columns and frame["mark"].notna().any())
        or (
            "bid" in frame.columns
            and "ask" in frame.columns
            and ((frame["bid"].notna()) & (frame["ask"].notna())).any()
        )
        or ("last_trade_price" in frame.columns and frame["last_trade_price"].notna().any())
    )
    return has_pricing
