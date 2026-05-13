"""
SSGA (State Street Global Advisors) daily fund flow collector.

State Street publishes their own Excel snapshot for all SPDR ETFs each trading day,
updated ~8:15am ET. This is the primary-source data that Bloomberg/Morningstar also
use to report ETF flows.

Flow formula (industry standard):
    net_inflow_usd = (shares_outstanding_today - shares_outstanding_yesterday) * nav_yesterday

No API key required. No cost. ToS permits personal/research use.

Column format notes (as of 2026-05):
    - "Shares Outstanding" is a string like "653.46 M" (millions)
    - "Total Net Assets" is a string like "$114,503.03 M" (millions USD)
    - "NAV" is a string like "$175.23"
    - "Closing Price" is a string like "$175.20"

The _parse_ssga_value() function handles all known SSGA formatting.
"""

import io
import re
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

_REQUIRED_COLUMNS = {"Shares Outstanding", "Total Net Assets"}


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
                "shares_outstanding": _parse_ssga_value(r.get("Shares Outstanding")),
                "aum_usd": _parse_ssga_value(r.get("Total Net Assets")),
                "nav": _parse_ssga_value(r.get("NAV")),
                "close": _parse_ssga_value(r.get("Closing Price")),
                # These will be filled by yfinance; SSGA doesn't provide OHLCV detail
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "adjusted_close": _parse_ssga_value(r.get("Closing Price")),
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
            raise CollectorError(
                f"SSGA Excel format changed — 'Ticker' column not found. "
                f"Columns: {list(df.columns)}"
            )

        df = df[df["Ticker"].notna()].copy()

        # Validate required columns
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise CollectorError(
                f"SSGA Excel missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Check that key columns are not all-null / all-unparseable
        for col in _REQUIRED_COLUMNS:
            parsed = df[col].apply(_parse_ssga_value)
            if parsed.notna().sum() == 0:
                raise CollectorError(
                    f"SSGA column '{col}' is present but all values failed to parse. "
                    f"Sample values: {df[col].head(3).tolist()}"
                )

        logger.info(f"[ssga] snapshot downloaded: {len(df)} ETFs, columns: {list(df.columns[:6])}")
        return df


def _parse_ssga_value(val: Any) -> float | None:
    """
    Parse SSGA-formatted numeric strings into floats.

    Handles:
        - Plain floats: 123.45 → 123.45
        - Dollar amounts: "$175.23" → 175.23
        - Millions with suffix: "653.46 M" → 653_460_000.0
        - Dollar+millions: "$114,503.03 M" → 114_503_030_000.0
        - Billions with suffix: "1.5 B" → 1_500_000_000.0
        - Comma-formatted: "1,234.56" → 1234.56
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            import math
            f = float(val)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    s = str(val).strip()
    if not s or s in ("-", "N/A", "n/a", "--"):
        return None

    # Remove leading $ and commas
    s = s.replace("$", "").replace(",", "").strip()

    multiplier = 1.0
    if s.upper().endswith(" M"):
        multiplier = 1_000_000.0
        s = s[:-2].strip()
    elif s.upper().endswith("M"):
        multiplier = 1_000_000.0
        s = s[:-1].strip()
    elif s.upper().endswith(" B"):
        multiplier = 1_000_000_000.0
        s = s[:-2].strip()
    elif s.upper().endswith("B"):
        multiplier = 1_000_000_000.0
        s = s[:-1].strip()

    try:
        return float(s) * multiplier
    except ValueError:
        return None


# Keep _to_float as an alias for backward compatibility with pipeline.py
def _to_float(val: Any) -> float | None:
    return _parse_ssga_value(val)
