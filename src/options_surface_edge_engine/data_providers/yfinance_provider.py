"""yfinance options provider."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from options_surface_edge_engine.data_providers.base import (
    BaseOptionsProvider,
    ProviderFetchError,
    ensure_standard_columns,
)


class YFinanceProvider(BaseOptionsProvider):
    """Fetch options data from yfinance."""

    def provider_name(self) -> str:
        """Provider identifier."""
        return "yfinance"

    def fetch_symbol_chain(self, symbol: str) -> pd.DataFrame:
        """Fetch full options chain across expiries."""
        try:
            from yfinance import Ticker
        except Exception as exc:  # pragma: no cover - import guard
            raise ProviderFetchError("yfinance import failed") from exc

        ticker_symbol = symbol.upper()
        try:
            ticker = Ticker(ticker_symbol)
            expiries = list(ticker.options)
        except Exception as exc:
            raise ProviderFetchError(f"Unable to get expirations for {ticker_symbol}") from exc

        if not expiries:
            raise ProviderFetchError(f"No options expirations returned for {ticker_symbol}")

        spot = self._extract_spot_price(ticker)
        currency = self._extract_currency(ticker)
        quote_ts = datetime.now(UTC)
        frames: list[pd.DataFrame] = []

        for expiration_str in expiries:
            try:
                expiration = datetime.strptime(expiration_str, "%Y-%m-%d").date()
                dte = (expiration - date.today()).days
                chain = ticker.option_chain(expiration_str)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["option_type"] = "call"
                puts["option_type"] = "put"
                block = pd.concat([calls, puts], axis=0, ignore_index=True)
                block["expiration"] = expiration
                block["dte"] = dte
                block["symbol"] = ticker_symbol
                block["provider"] = self.provider_name()
                block["quote_ts"] = quote_ts
                block["underlying_price"] = spot
                block["currency"] = currency
                frames.append(block)
            except Exception:
                continue

        if not frames:
            raise ProviderFetchError(f"No options chain rows retrieved for {ticker_symbol}")

        combined = pd.concat(frames, ignore_index=True)
        normalized = ensure_standard_columns(combined, symbol=ticker_symbol, provider=self.provider_name())
        return normalized

    def fetch_spot_history(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch daily spot history for trend features."""
        try:
            from yfinance import Ticker
        except Exception as exc:  # pragma: no cover - import guard
            raise ProviderFetchError("yfinance import failed") from exc

        ticker_symbol = symbol.upper()
        try:
            history = Ticker(ticker_symbol).history(period=f"{lookback_days}d", interval="1d")
            if history.empty:
                raise ProviderFetchError(f"No spot history available for {ticker_symbol}")
            frame = history.reset_index()
            frame.columns = [str(col).lower() for col in frame.columns]
            frame["symbol"] = ticker_symbol
            return frame
        except ProviderFetchError:
            raise
        except Exception as exc:
            raise ProviderFetchError(f"Unable to fetch spot history for {ticker_symbol}") from exc

    @staticmethod
    def _extract_spot_price(ticker: "Ticker") -> float | None:
        """Extract best available spot price."""
        info = {}
        fast_info = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}
        try:
            fast_info = dict(ticker.fast_info or {})
        except Exception:
            fast_info = {}
        candidates = [
            fast_info.get("last_price"),
            fast_info.get("regular_market_price"),
            info.get("postMarketPrice"),
            info.get("regularMarketPrice"),
            info.get("currentPrice"),
            info.get("previousClose"),
        ]
        for value in candidates:
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_currency(ticker: "Ticker") -> str | None:
        """Extract underlying currency."""
        try:
            info = ticker.info or {}
        except Exception:
            return None
        currency = info.get("currency")
        if currency is None:
            return None
        return str(currency)
