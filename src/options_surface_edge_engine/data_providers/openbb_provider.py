"""Optional OpenBB provider adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from options_surface_edge_engine.data_providers.base import (
    BaseOptionsProvider,
    ProviderFetchError,
    ensure_standard_columns,
)


class OpenBBProvider(BaseOptionsProvider):
    """Fetch options and spot history via OpenBB."""

    def __init__(self, openbb_provider: str = "yfinance") -> None:
        """Initialize OpenBB adapter."""
        self._openbb_provider = openbb_provider

    def provider_name(self) -> str:
        """Provider identifier."""
        return "openbb"

    def fetch_symbol_chain(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain via OpenBB endpoint."""
        obb = self._get_obb()
        ticker = symbol.upper()
        try:
            response = obb.derivatives.options.chains(ticker, provider=self._openbb_provider)
        except Exception as exc:
            raise ProviderFetchError(f"OpenBB chain fetch failed for {ticker}") from exc

        frame = self._obbject_to_frame(response)
        if frame.empty:
            raise ProviderFetchError(f"OpenBB returned empty chain for {ticker}")

        frame["symbol"] = ticker
        frame["provider"] = f"openbb:{self._openbb_provider}"
        frame["quote_ts"] = datetime.now(UTC)
        currency = None
        try:
            currency = (getattr(response, "metadata", {}) or {}).get("currency")
        except Exception:
            currency = None
        frame["currency"] = currency
        return ensure_standard_columns(frame, symbol=ticker, provider=self.provider_name())

    def fetch_spot_history(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch spot history via OpenBB equity historical endpoint."""
        obb = self._get_obb()
        ticker = symbol.upper()
        try:
            response = obb.equity.price.historical(ticker, provider=self._openbb_provider, interval="1d")
            frame = self._obbject_to_frame(response)
            if frame.empty:
                raise ProviderFetchError(f"OpenBB returned empty history for {ticker}")
            frame.columns = [str(c).lower() for c in frame.columns]
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.sort_values("date").tail(lookback_days).reset_index(drop=True)
            frame["symbol"] = ticker
            return frame
        except ProviderFetchError:
            raise
        except Exception as exc:
            raise ProviderFetchError(f"OpenBB history fetch failed for {ticker}") from exc

    @staticmethod
    def _get_obb():
        """Import OpenBB client lazily."""
        try:
            from openbb import obb
        except Exception as exc:  # pragma: no cover - import guard
            raise ProviderFetchError("OpenBB is not installed. Install with `pip install openbb`.") from exc
        return obb

    @staticmethod
    def _obbject_to_frame(response: object) -> pd.DataFrame:
        """Convert OpenBB response object into DataFrame."""
        if hasattr(response, "to_df"):
            try:
                frame = response.to_df()  # type: ignore[no-any-return]
                if isinstance(frame, pd.DataFrame):
                    return frame
            except Exception:
                pass
        results = getattr(response, "results", None)
        if results is None:
            return pd.DataFrame()
        if isinstance(results, pd.DataFrame):
            return results
        if isinstance(results, list):
            return pd.DataFrame(results)
        return pd.DataFrame()
