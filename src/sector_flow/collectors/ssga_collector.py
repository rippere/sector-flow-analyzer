"""
SSGA (State Street Global Advisors) daily fund flow collector.

State Street publishes their own Excel snapshot for all SPDR ETFs each trading day,
updated ~8:15am ET. This is the primary-source data that Bloomberg/Morningstar also
use to report ETF flows.

Flow formula (industry standard):
    net_inflow_usd = (shares_outstanding_today - shares_outstanding_yesterday) * nav_yesterday

No API key required. No cost. ToS permits personal/research use.
"""

import io
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests
from loguru import logger

from sector_flow.collectors.base import BaseCollector, CollectorError

SSGA_EXCEL_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "spdr-product-data-us-en.xlsx"
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; personal research)"}


class SSGACollector(BaseCollector):
    """
    Downloads the SSGA daily Excel snapshot and returns shares_outstanding,
    aum_usd, and nav per ticker. Call once per day after 8:15am ET.

    Net inflow computation requires two consecutive days of data — the pipeline
    layer (not this collector) is responsible for computing the delta.
    """

    @property
    def source_name(self) -> str:
        return "ssga"

    def fetch(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        snapshot = self._download_snapshot()
        row = snapshot[snapshot["Ticker"] == ticker]
        if row.empty:
            raise CollectorError(f"SSGA snapshot missing ticker {ticker}")

        r = row.iloc[0]
        # SSGA snapshot is today's data only — date range params are ignored.
        # The caller should only invoke this for today's date.
        today = datetime.combine(date.today(), datetime.min.time())
        return [
            {
                "date": today,
                "shares_outstanding": _to_float(r.get("Shares Outstanding")),
                "aum_usd": _to_float(r.get("Total Net Assets")),
                "nav": _to_float(r.get("NAV")),
                "close": _to_float(r.get("Closing Price")),
                # These will be filled by yfinance; SSGA doesn't provide OHLCV detail
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "adjusted_close": _to_float(r.get("Closing Price")),
            }
        ]

    def fetch_snapshot(self) -> pd.DataFrame:
        """Return the full parsed SSGA snapshot for all SPDR ETFs."""
        return self._download_snapshot()

    def _download_snapshot(self) -> pd.DataFrame:
        logger.debug(f"[ssga] downloading snapshot from {SSGA_EXCEL_URL}")
        try:
            resp = requests.get(SSGA_EXCEL_URL, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise CollectorError(f"SSGA download failed: {exc}") from exc

        try:
            df = pd.read_excel(io.BytesIO(resp.content), header=1)
        except Exception as exc:
            raise CollectorError(f"SSGA Excel parse failed: {exc}") from exc

        df.columns = df.columns.str.strip()
        if "Ticker" not in df.columns:
            raise CollectorError(f"SSGA Excel format changed — 'Ticker' column not found. Columns: {list(df.columns)}")

        df = df[df["Ticker"].notna()].copy()
        logger.info(f"[ssga] snapshot downloaded: {len(df)} ETFs, columns: {list(df.columns[:6])}")
        return df


def _to_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
