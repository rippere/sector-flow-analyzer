"""
Tests for the SSGA collector — column validation and format parsing.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sector_flow.collectors.base import CollectorError
from sector_flow.collectors.ssga_collector import SSGACollector, _parse_ssga_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Write a DataFrame as an Excel file in memory matching SSGA's format.

    SSGA Excel uses header=1, meaning pandas skips row 0 (a title row) and
    treats row 1 as the header. We reproduce this by writing a blank first row,
    then the column headers, then data — using openpyxl directly.
    """
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active

    # Row 0: blank title row (SSGA has a product title here)
    ws.append(["SSGA Fund Data"] + [""] * (len(df.columns) - 1))

    # Row 1: column headers
    ws.append(list(df.columns))

    # Rows 2+: data
    for _, row in df.iterrows():
        ws.append(list(row))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_minimal_ssga_df() -> pd.DataFrame:
    """Create a minimal valid SSGA-format DataFrame."""
    return pd.DataFrame(
        {
            "Ticker": ["XLK", "XLF", "XLE"],
            "Name": ["Tech ETF", "Fin ETF", "Energy ETF"],
            "Shares Outstanding": ["653.46 M", "977.10 M", "709.10 M"],
            "Total Net Assets": ["$114,503.03 M", "$50,368.73 M", "$40,827.59 M"],
            "NAV": ["$175.23", "$51.55", "$57.58"],
            "Closing Price": ["$175.20", "$51.58", "$57.57"],
        }
    )


def _mock_http_response(excel_bytes: bytes) -> MagicMock:
    """Build a mock requests.Response returning the given Excel bytes."""
    mock = MagicMock()
    mock.content = excel_bytes
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# _parse_ssga_value unit tests
# ---------------------------------------------------------------------------


class TestParseSSGAValue:
    def test_plain_float(self):
        assert _parse_ssga_value(123.45) == pytest.approx(123.45)

    def test_dollar_amount(self):
        assert _parse_ssga_value("$175.23") == pytest.approx(175.23)

    def test_shares_millions(self):
        assert _parse_ssga_value("653.46 M") == pytest.approx(653_460_000.0)

    def test_aum_dollar_millions(self):
        assert _parse_ssga_value("$114,503.03 M") == pytest.approx(114_503_030_000.0)

    def test_billions(self):
        assert _parse_ssga_value("1.5 B") == pytest.approx(1_500_000_000.0)

    def test_none_returns_none(self):
        assert _parse_ssga_value(None) is None

    def test_dash_returns_none(self):
        assert _parse_ssga_value("-") is None

    def test_na_returns_none(self):
        assert _parse_ssga_value("N/A") is None

    def test_comma_formatted(self):
        assert _parse_ssga_value("1,234.56") == pytest.approx(1234.56)


# ---------------------------------------------------------------------------
# SSGACollector integration tests
# ---------------------------------------------------------------------------


class TestSSGASnapshotRequiredColumns:
    def test_ssga_snapshot_has_required_columns(self):
        """Mock a valid HTTP response; assert required columns are present and parseable."""
        df = _make_minimal_ssga_df()
        # SSGA Excel uses header=1 — we need to prepend one blank row
        excel_bytes = _make_excel_bytes(df)

        with patch("sector_flow.collectors.ssga_collector.requests.get") as mock_get:
            mock_get.return_value = _mock_http_response(excel_bytes)
            collector = SSGACollector()
            result = collector.fetch_snapshot()

        assert "Ticker" in result.columns
        assert "Shares Outstanding" in result.columns
        assert "Total Net Assets" in result.columns

        # Values should parse to non-None floats
        parsed_shares = result["Shares Outstanding"].apply(_parse_ssga_value)
        assert parsed_shares.notna().sum() == len(result), "All Shares Outstanding should be parseable"

    def test_ssga_missing_column_raises(self):
        """Assert CollectorError is raised if required columns are missing."""
        # DataFrame without the required columns
        bad_df = pd.DataFrame(
            {
                "Ticker": ["XLK", "XLF"],
                "Name": ["Tech ETF", "Fin ETF"],
                # Missing: Shares Outstanding, Total Net Assets
            }
        )
        excel_bytes = _make_excel_bytes(bad_df)

        with patch("sector_flow.collectors.ssga_collector.requests.get") as mock_get:
            mock_get.return_value = _mock_http_response(excel_bytes)
            collector = SSGACollector()
            with pytest.raises(CollectorError, match="missing required columns"):
                collector.fetch_snapshot()

    def test_ssga_all_null_column_raises(self):
        """Assert CollectorError if required column exists but all values fail to parse."""
        bad_df = pd.DataFrame(
            {
                "Ticker": ["XLK", "XLF"],
                "Shares Outstanding": ["N/A", "-"],  # all unparseable
                "Total Net Assets": ["N/A", "-"],
            }
        )
        excel_bytes = _make_excel_bytes(bad_df)

        with patch("sector_flow.collectors.ssga_collector.requests.get") as mock_get:
            mock_get.return_value = _mock_http_response(excel_bytes)
            collector = SSGACollector()
            with pytest.raises(CollectorError, match="all values failed to parse"):
                collector.fetch_snapshot()
