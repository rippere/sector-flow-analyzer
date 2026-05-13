from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class BaseCollector(ABC):
    """
    Contract all data collectors must satisfy.
    Swap data sources by swapping the collector — callers don't change.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable identifier shown in logs and metrics."""
        ...

    @abstractmethod
    def fetch(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        """
        Return a list of records for the given ticker and date range.
        Each dict must contain at minimum: date, close.
        Optional keys: open, high, low, volume, adjusted_close,
                       net_inflow_usd, shares_outstanding, aum_usd.
        Raises CollectorError on unrecoverable failure.
        """
        ...

    def fetch_all(self, tickers: list[str], start: date, end: date) -> dict[str, list[dict]]:
        return {ticker: self.fetch(ticker, start, end) for ticker in tickers}


class CollectorError(Exception):
    """Raised when a collector cannot retrieve data."""
