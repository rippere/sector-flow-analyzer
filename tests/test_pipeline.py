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
