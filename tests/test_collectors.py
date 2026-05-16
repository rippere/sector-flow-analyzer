"""
Tests for sector_flow.collectors — YFinanceCollector, SSGACollector, _parse_ssga_value.

All external calls (yfinance, requests) are mocked; no network required.
"""

from __future__ import annotations

import io
import math
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sector_flow.collectors.base import CollectorError
from sector_flow.collectors.ssga_collector import (
    SSGACollector,
    _parse_ssga_value,
    _to_float,
)
from sector_flow.collectors.yfinance_collector import YFinanceCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_START = date(2024, 1, 2)
_END = date(2024, 1, 10)


def _make_yf_history(n: int = 3) -> "pd.DataFrame":
    """Build a minimal yfinance-style DataFrame."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2024-01-02", periods=n, freq="B", tz="America/New_York")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [102.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [101.0 + i for i in range(n)],
            "Volume": [1_000_000.0] * n,
            "Adj Close": [101.5 + i for i in range(n)],
        },
        index=dates,
    )
    return df


def _ssga_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize a DataFrame to Excel bytes mimicking SSGA format:
      row 0 — blank metadata row (skipped by header=1)
      row 1 — column headers
      row 2+ — data rows
    `pd.read_excel(buf, header=1)` reads this correctly.
    """
    import openpyxl
    buf = io.BytesIO()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([""] * len(df.columns))       # row 0: blank (skipped by header=1)
    ws.append(list(df.columns))             # row 1: headers
    for _, row in df.iterrows():
        ws.append([row[c] for c in df.columns])
    wb.save(buf)
    return buf.getvalue()


def _make_ssga_dataframe(tickers: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal SSGA-style DataFrame."""
    if tickers is None:
        tickers = ["XLK", "XLF", "XLE"]
    rows = []
    for t in tickers:
        rows.append({
            "Ticker": t,
            "Shares Outstanding": "653.46 M",
            "Total Net Assets": "$114,503.03 M",
            "NAV": "$175.23",
            "Closing Price": "$175.20",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _parse_ssga_value
# ---------------------------------------------------------------------------

class TestParseSSGAValue:
    def test_none_returns_none(self):
        assert _parse_ssga_value(None) is None

    def test_plain_float(self):
        assert _parse_ssga_value(175.23) == pytest.approx(175.23)

    def test_plain_int(self):
        assert _parse_ssga_value(100) == pytest.approx(100.0)

    def test_nan_returns_none(self):
        assert _parse_ssga_value(float("nan")) is None

    def test_dollar_string(self):
        assert _parse_ssga_value("$175.23") == pytest.approx(175.23)

    def test_millions_space_m(self):
        assert _parse_ssga_value("653.46 M") == pytest.approx(653_460_000.0)

    def test_millions_no_space(self):
        assert _parse_ssga_value("653.46M") == pytest.approx(653_460_000.0)

    def test_billions_space_b(self):
        assert _parse_ssga_value("1.5 B") == pytest.approx(1_500_000_000.0)

    def test_billions_no_space(self):
        assert _parse_ssga_value("1.5B") == pytest.approx(1_500_000_000.0)

    def test_dollar_comma_millions(self):
        result = _parse_ssga_value("$114,503.03 M")
        assert result == pytest.approx(114_503_030_000.0)

    def test_comma_formatted_plain(self):
        assert _parse_ssga_value("1,234.56") == pytest.approx(1234.56)

    def test_na_string_returns_none(self):
        assert _parse_ssga_value("N/A") is None
        assert _parse_ssga_value("n/a") is None

    def test_dash_returns_none(self):
        assert _parse_ssga_value("-") is None
        assert _parse_ssga_value("--") is None

    def test_empty_string_returns_none(self):
        assert _parse_ssga_value("") is None

    def test_invalid_string_returns_none(self):
        assert _parse_ssga_value("not_a_number") is None

    def test_case_insensitive_suffix(self):
        assert _parse_ssga_value("1.0 m") == pytest.approx(1_000_000.0)
        assert _parse_ssga_value("1.0 b") == pytest.approx(1_000_000_000.0)

    def test_to_float_alias(self):
        assert _to_float("$50.00") == pytest.approx(50.0)
        assert _to_float(None) is None


# ---------------------------------------------------------------------------
# YFinanceCollector
# ---------------------------------------------------------------------------

class TestYFinanceCollector:
    def setup_method(self):
        self.collector = YFinanceCollector()

    def test_source_name(self):
        assert self.collector.source_name == "yfinance"

    def test_fetch_returns_records(self):
        mock_history = _make_yf_history(5)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            records = self.collector.fetch("XLK", _START, _END)

        assert len(records) == 5
        assert all(
            k in records[0]
            for k in ("date", "open", "high", "low", "close", "volume", "adjusted_close")
        )

    def test_fetch_record_values_are_floats(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(1)

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            records = self.collector.fetch("XLK", _START, _END)

        r = records[0]
        assert isinstance(r["close"], float)
        assert isinstance(r["volume"], float)
        assert isinstance(r["adjusted_close"], float)

    def test_fetch_date_is_naive_datetime(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(1)

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            records = self.collector.fetch("XLK", _START, _END)

        assert isinstance(records[0]["date"], datetime)
        assert records[0]["date"].tzinfo is None  # must be naive

    def test_fetch_empty_returns_empty_list(self):
        import pandas as pd
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            records = self.collector.fetch("XLK", _START, _END)

        assert records == []

    def test_fetch_raises_collector_error_on_exception(self):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("network error")

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(CollectorError, match="yfinance fetch failed"):
                self.collector.fetch("XLK", _START, _END)

    def test_fetch_all_returns_dict_keyed_by_ticker(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(2)

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            result = self.collector.fetch_all(["XLK", "XLF"], _START, _END)

        assert set(result.keys()) == {"XLK", "XLF"}
        assert len(result["XLK"]) == 2

    def test_fetch_correct_date_range_passed_to_yfinance(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(0)
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("sector_flow.collectors.yfinance_collector.yf.Ticker", return_value=mock_ticker):
            self.collector.fetch("XLK", _START, _END)

        call_kwargs = mock_ticker.history.call_args[1]
        assert call_kwargs["start"] == _START.isoformat()
        assert call_kwargs["end"] == _END.isoformat()
        assert call_kwargs["auto_adjust"] is False


# ---------------------------------------------------------------------------
# SSGACollector
# ---------------------------------------------------------------------------

class TestSSGACollector:
    def setup_method(self):
        self.collector = SSGACollector()

    def test_source_name(self):
        assert self.collector.source_name == "ssga"

    def _mock_download(self, df: pd.DataFrame | None = None):
        if df is None:
            df = _make_ssga_dataframe()
        return patch.object(self.collector, "_download_snapshot", return_value=df)

    def test_fetch_returns_one_record_per_ticker(self):
        with self._mock_download():
            records = self.collector.fetch("XLK", _START, _END)
        assert len(records) == 1

    def test_fetch_record_has_required_keys(self):
        with self._mock_download():
            records = self.collector.fetch("XLK", _START, _END)
        r = records[0]
        assert "date" in r
        assert "shares_outstanding" in r
        assert "aum_usd" in r
        assert "nav" in r

    def test_fetch_shares_outstanding_parsed(self):
        with self._mock_download():
            records = self.collector.fetch("XLK", _START, _END)
        # "653.46 M" → 653_460_000.0
        assert records[0]["shares_outstanding"] == pytest.approx(653_460_000.0)

    def test_fetch_aum_usd_parsed(self):
        with self._mock_download():
            records = self.collector.fetch("XLK", _START, _END)
        # "$114,503.03 M" → 114_503_030_000.0
        assert records[0]["aum_usd"] == pytest.approx(114_503_030_000.0)

    def test_fetch_raises_for_unknown_ticker(self):
        with self._mock_download():
            with pytest.raises(CollectorError, match="SSGA snapshot missing ticker"):
                self.collector.fetch("FAKE", _START, _END)

    def test_fetch_date_is_today(self):
        with self._mock_download():
            records = self.collector.fetch("XLK", _START, _END)
        today = date.today()
        assert records[0]["date"].date() == today

    def test_fetch_snapshot_returns_dataframe(self):
        df = _make_ssga_dataframe(["XLK", "XLF"])
        with self._mock_download(df):
            result = self.collector.fetch_snapshot()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_download_snapshot_raises_on_network_error(self):
        import requests as req_mod
        with patch("sector_flow.collectors.ssga_collector.requests.get",
                   side_effect=req_mod.RequestException("timeout")):
            with pytest.raises(CollectorError, match="SSGA download failed"):
                self.collector._download_snapshot()

    def test_download_snapshot_raises_on_bad_status(self):
        import requests as req_mod
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_mod.HTTPError("404 Not Found")
        with patch("sector_flow.collectors.ssga_collector.requests.get",
                   return_value=mock_resp):
            with pytest.raises(CollectorError, match="SSGA download failed"):
                self.collector._download_snapshot()

    def test_download_snapshot_raises_when_ticker_column_missing(self):
        df_no_ticker = pd.DataFrame({"Shares Outstanding": ["1 M"], "Total Net Assets": ["$1 M"]})
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = _ssga_excel_bytes(df_no_ticker)

        with patch("sector_flow.collectors.ssga_collector.requests.get", return_value=mock_resp):
            with pytest.raises(CollectorError, match="Ticker.*column not found"):
                self.collector._download_snapshot()

    def test_download_snapshot_raises_when_required_columns_missing(self):
        # Has Ticker but is missing Shares Outstanding and Total Net Assets
        df_missing = pd.DataFrame({"Ticker": ["XLK"], "NAV": ["$175.23"]})
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = _ssga_excel_bytes(df_missing)

        with patch("sector_flow.collectors.ssga_collector.requests.get", return_value=mock_resp):
            with pytest.raises(CollectorError, match="missing required columns"):
                self.collector._download_snapshot()

    def test_download_snapshot_raises_on_corrupt_excel_bytes(self):
        """When resp.content is not valid Excel, CollectorError is raised."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = b"this is not excel content at all"

        with patch("sector_flow.collectors.ssga_collector.requests.get", return_value=mock_resp):
            with pytest.raises(CollectorError, match="SSGA Excel parse failed"):
                self.collector._download_snapshot()

    def test_download_snapshot_success(self):
        df = _make_ssga_dataframe()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.content = _ssga_excel_bytes(df)

        with patch("sector_flow.collectors.ssga_collector.requests.get", return_value=mock_resp):
            result = self.collector._download_snapshot()

        assert "Ticker" in result.columns
        assert "XLK" in result["Ticker"].values
