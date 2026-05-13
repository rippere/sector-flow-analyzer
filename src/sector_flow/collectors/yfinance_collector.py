from datetime import date, datetime
from typing import Any

import yfinance as yf
from loguru import logger

from sector_flow.collectors.base import BaseCollector, CollectorError


class YFinanceCollector(BaseCollector):
    """Fetches daily OHLCV price data via yfinance. Free, no API key required."""

    @property
    def source_name(self) -> str:
        return "yfinance"

    def fetch(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        logger.debug(f"[yfinance] fetching {ticker} {start} → {end}")
        try:
            raw = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
            )
        except Exception as exc:
            raise CollectorError(f"yfinance fetch failed for {ticker}: {exc}") from exc

        if raw.empty:
            logger.warning(f"[yfinance] no data returned for {ticker} {start}→{end}")
            return []

        records = []
        for ts, row in raw.iterrows():
            dt = ts.to_pydatetime().replace(tzinfo=None)
            records.append(
                {
                    "date": dt,
                    "open": float(row.get("Open", 0)),
                    "high": float(row.get("High", 0)),
                    "low": float(row.get("Low", 0)),
                    "close": float(row.get("Close", 0)),
                    "volume": float(row.get("Volume", 0)),
                    "adjusted_close": float(row.get("Adj Close", row.get("Close", 0))),
                }
            )

        logger.info(f"[yfinance] {ticker}: {len(records)} rows fetched")
        return records
