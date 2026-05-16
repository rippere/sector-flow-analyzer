"""
Tests for the data ingestion pipeline — gap detection and backfill logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData
from sector_flow.database.repository import ETFRepository, PriceRepository


def _make_in_memory_db():
    """Create a fresh in-memory SQLite engine + session."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def test_gap_detection_uses_latest_db_date():
    """
    When the DB has price data through 5 days ago, run_daily should call the
    yfinance collector with a start date at least 5 days before today.
    """
    engine, Session = _make_in_memory_db()
    session = Session()

    # Seed ETFs
    etf_repo = ETFRepository(session)
    etf_repo.seed_etfs()
    session.commit()

    # Insert price data through 5 days ago for all ETFs
    today = date.today()
    five_days_ago = today - timedelta(days=5)
    five_days_ago_dt = datetime.combine(five_days_ago, datetime.min.time())

    for ticker, _, _ in SECTOR_ETFS:
        etf = session.query(SectorETF).filter_by(ticker=ticker).first()
        session.add(
            PriceData(
                etf_id=etf.id,
                date=five_days_ago_dt,
                open=100.0, high=102.0, low=99.0,
                close=101.0, volume=1e6,
                adjusted_close=101.0,
            )
        )
    session.commit()
    session.close()

    # Track the start date passed to the collector
    collector_calls: list[date] = []

    def fake_fetch(ticker, start, end):
        collector_calls.append(start)
        return []

    import sector_flow.pipeline as pipeline_mod
    from contextlib import contextmanager

    # Patch session to use our in-memory DB
    @contextmanager
    def mock_get_session(db_url=None):
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    with patch.object(pipeline_mod, "get_session", mock_get_session), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF, \
         patch("sector_flow.pipeline.SSGACollector") as MockSSGA:

        mock_yf = MagicMock()
        mock_yf.fetch.side_effect = fake_fetch
        MockYF.return_value = mock_yf

        mock_ssga = MagicMock()
        mock_ssga.fetch_snapshot.return_value = __import__("pandas").DataFrame()
        MockSSGA.return_value = mock_ssga

        pipeline_mod.run_daily(database_url="sqlite://")

    assert len(collector_calls) > 0, "YFinanceCollector.fetch was never called"

    # The start date should be at least 5 days before today (gap >= 5)
    earliest_start = min(collector_calls)
    days_back = (today - earliest_start).days
    assert days_back >= 5, (
        f"Expected collector called with start >= 5 days ago, "
        f"got start={earliest_start} ({days_back} days back)"
    )

    engine.dispose()


# ---------------------------------------------------------------------------
# Helpers shared across new tests
# ---------------------------------------------------------------------------

from contextlib import contextmanager
import sector_flow.pipeline as pipeline_mod


def _mock_session_ctx(Session):
    @contextmanager
    def _ctx(db_url=None):
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
    return _ctx


def _no_ssga():
    """Return a mock SSGACollector whose fetch_snapshot returns an empty DF."""
    import pandas as pd
    mock = MagicMock()
    mock.fetch_snapshot.return_value = pd.DataFrame()
    return mock


# ---------------------------------------------------------------------------
# _compute_gap unit tests
# ---------------------------------------------------------------------------

def test_compute_gap_returns_max_gap_when_no_price_data():
    """`_compute_gap` returns _MAX_GAP_DAYS when ETFs are seeded but have no price rows."""
    from sector_flow.pipeline import _compute_gap, _MAX_GAP_DAYS

    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()

    result = _compute_gap(PriceRepository(s), ETFRepository(s))
    s.close()
    engine.dispose()

    assert result == _MAX_GAP_DAYS


def test_compute_gap_returns_one_when_data_is_current():
    """`_compute_gap` returns 1 when all ETFs have data from today."""
    from sector_flow.pipeline import _compute_gap

    engine, Session = _make_in_memory_db()
    s = Session()
    etf_repo = ETFRepository(s)
    etf_repo.seed_etfs()
    s.commit()

    today_dt = datetime.combine(date.today(), datetime.min.time())
    for ticker, _, _ in SECTOR_ETFS:
        etf = s.query(SectorETF).filter_by(ticker=ticker).first()
        s.add(PriceData(
            etf_id=etf.id, date=today_dt,
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=1e6, adjusted_close=101.0,
        ))
    s.commit()

    result = _compute_gap(PriceRepository(s), ETFRepository(s))
    s.close()
    engine.dispose()

    assert result == 1


def test_compute_gap_handles_date_object_not_datetime():
    """`_compute_gap` works when get_latest_date returns a date (not datetime)."""
    from sector_flow.pipeline import _compute_gap

    mock_price_repo = MagicMock()
    mock_etf_repo = MagicMock()

    # Simulate 10 days old data returned as a plain date object
    mock_etf = MagicMock()
    mock_etf.id = 1
    mock_etf_repo.get_by_ticker.return_value = mock_etf

    target_date = date.today() - timedelta(days=10)
    mock_price_repo.get_latest_date.return_value = target_date  # date, not datetime

    result = _compute_gap(mock_price_repo, mock_etf_repo)
    assert result == 10


def test_compute_gap_handles_etf_not_in_db():
    """`_compute_gap` skips gracefully when a ticker has no ETF record."""
    from sector_flow.pipeline import _compute_gap

    mock_price_repo = MagicMock()
    mock_etf_repo = MagicMock()
    mock_etf_repo.get_by_ticker.return_value = None  # no ETF found

    # Should return 1 since max_gap stays 0
    result = _compute_gap(mock_price_repo, mock_etf_repo)
    assert result == 1
    mock_price_repo.get_latest_date.assert_not_called()


# ---------------------------------------------------------------------------
# run_daily — additional branch coverage
# ---------------------------------------------------------------------------

def test_run_daily_logs_warning_for_large_gap():
    """`run_daily` warns when the detected gap exceeds WARN_GAP_TRADING_DAYS."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()

    # Insert data 10 days ago to produce a gap > 5
    ten_ago = datetime.combine(date.today() - timedelta(days=10), datetime.min.time())
    for ticker, _, _ in SECTOR_ETFS:
        etf = s.query(SectorETF).filter_by(ticker=ticker).first()
        s.add(PriceData(
            etf_id=etf.id, date=ten_ago,
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=1e6, adjusted_close=101.0,
        ))
    s.commit()
    s.close()

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF, \
         patch("sector_flow.pipeline.SSGACollector") as MockSSGA, \
         patch("sector_flow.pipeline.logger") as mock_logger:

        MockYF.return_value.fetch.return_value = []
        MockSSGA.return_value = _no_ssga()

        pipeline_mod.run_daily(database_url="sqlite://")

    # Warning should have been called at least once
    assert mock_logger.warning.called
    engine.dispose()


def test_run_daily_captures_yfinance_error():
    """`run_daily` catches yfinance exceptions and records them in summary['errors']."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF, \
         patch("sector_flow.pipeline.SSGACollector") as MockSSGA:

        MockYF.return_value.fetch.side_effect = RuntimeError("network timeout")
        MockSSGA.return_value = _no_ssga()

        result = pipeline_mod.run_daily(database_url="sqlite://")

    assert len(result["errors"]) > 0
    assert any("yfinance" in e for e in result["errors"])
    engine.dispose()


def test_run_daily_captures_ssga_error():
    """`run_daily` catches SSGA exceptions and records them in summary['errors']."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF, \
         patch("sector_flow.pipeline.SSGACollector") as MockSSGA:

        MockYF.return_value.fetch.return_value = []
        MockSSGA.return_value.fetch_snapshot.side_effect = RuntimeError("ssga down")

        result = pipeline_mod.run_daily(database_url="sqlite://")

    assert any("ssga" in e for e in result["errors"])
    engine.dispose()


def test_run_daily_ssga_updates_flow_fields():
    """`run_daily` updates ssga_rows when fetch_snapshot has matching ticker data."""
    import pandas as pd

    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    # Insert today's price row so update_flow_fields has a row to update
    today_dt = datetime.combine(date.today(), datetime.min.time())
    first_ticker = SECTOR_ETFS[0][0]
    etf = s.query(SectorETF).filter_by(ticker=first_ticker).first()
    s.add(PriceData(
        etf_id=etf.id, date=today_dt,
        open=100.0, high=102.0, low=99.0, close=101.0,
        volume=1e6, adjusted_close=101.0,
    ))
    s.commit()
    s.close()

    # Build a snapshot DataFrame with one matching ticker
    snapshot_df = pd.DataFrame([{
        "Ticker": first_ticker,
        "Shares Outstanding": "653.46 M",
        "Total Net Assets": "$114,503.03 M",
        "NAV": "$175.23",
        "Closing Price": "$175.20",
    }])

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF, \
         patch("sector_flow.pipeline.SSGACollector") as MockSSGA:

        MockYF.return_value.fetch.return_value = []
        MockSSGA.return_value.fetch_snapshot.return_value = snapshot_df

        result = pipeline_mod.run_daily(database_url="sqlite://")

    assert result["ssga_rows"] >= 1
    engine.dispose()


# ---------------------------------------------------------------------------
# backfill tests
# ---------------------------------------------------------------------------

def test_backfill_with_explicit_days():
    """`backfill(days=7)` calls yfinance.fetch with start = today-7."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    starts_seen: list[date] = []

    def fake_fetch(ticker, start, end):
        starts_seen.append(start)
        return []

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF:

        MockYF.return_value.fetch.side_effect = fake_fetch
        result = pipeline_mod.backfill(days=7, database_url="sqlite://")

    assert result["rows_saved"] == 0  # fake_fetch returns []
    assert len(starts_seen) > 0
    expected_start = date.today() - timedelta(days=7)
    assert starts_seen[0] == expected_start, f"Expected {expected_start}, got {starts_seen[0]}"
    engine.dispose()


def test_backfill_without_days_uses_90_on_empty_db():
    """`backfill()` with no days uses 90-day default when DB is empty."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    starts_seen: list[date] = []

    def fake_fetch(ticker, start, end):
        starts_seen.append(start)
        return []

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF:

        MockYF.return_value.fetch.side_effect = fake_fetch
        pipeline_mod.backfill(days=None, database_url="sqlite://")

    # With empty DB, gap = _MAX_GAP_DAYS (30) >= 30, so actual_days = 90
    expected_start = date.today() - timedelta(days=90)
    assert starts_seen[0] == expected_start
    engine.dispose()


def test_backfill_captures_fetch_errors():
    """`backfill()` continues past per-ticker errors and records them."""
    engine, Session = _make_in_memory_db()
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch.object(pipeline_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch("sector_flow.pipeline.YFinanceCollector") as MockYF:

        MockYF.return_value.fetch.side_effect = RuntimeError("connection refused")
        result = pipeline_mod.backfill(days=5, database_url="sqlite://")

    assert len(result["errors"]) > 0
    assert result["rows_saved"] == 0
    engine.dispose()
