"""Tests for the intraday refresh path (pipeline.run_intraday) and the
engine's intraday-tagged persistence (engine.run_analysis(intraday=True))."""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import datetime, timedelta, date
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData, FlowMetric, CovarianceMatrix
from sector_flow.database.repository import ETFRepository

import sector_flow.pipeline as pipeline_mod
import sector_flow.analysis.engine as engine_mod


def _make_in_memory_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


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


def _seed_prices(Session, days: int = 90):
    """Seed `days` of gently-trending synthetic prices for every sector ETF."""
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    start = datetime.combine(date.today() - timedelta(days=days), datetime.min.time())
    for i, (ticker, _, _) in enumerate(SECTOR_ETFS):
        etf = s.query(SectorETF).filter_by(ticker=ticker).first()
        for d in range(days):
            # distinct per-ticker waveform so correlations are well-defined
            px = 100.0 + 10.0 * math.sin((d + i * 5) / 9.0) + d * 0.05
            s.add(PriceData(
                etf_id=etf.id,
                date=start + timedelta(days=d),
                open=px, high=px * 1.01, low=px * 0.99,
                close=px, volume=1e6, adjusted_close=px,
            ))
    s.commit()
    s.close()


# --- pipeline.run_intraday gating -----------------------------------------

def test_run_intraday_skips_when_market_closed():
    """When the market is closed and force is False, run_intraday no-ops and
    never touches the data collectors."""
    with patch("sector_flow.market_calendar.is_market_open", return_value=False), \
         patch.object(pipeline_mod, "YFinanceCollector") as MockYF:
        result = pipeline_mod.run_intraday(database_url="sqlite://", force=False)

    assert result["skipped"] == "market_closed"
    MockYF.assert_not_called()


# --- engine intraday tagging ----------------------------------------------

def test_engine_intraday_writes_suffixed_metrics_and_skips_covariance():
    engine, Session = _make_in_memory_db()
    _seed_prices(Session)

    with patch.object(engine_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(engine_mod, "init_db", lambda db_url=None: None):
        result = engine_mod.run_analysis(database_url="sqlite://", intraday=True)

    assert result["intraday"] is True

    s = Session()
    names = {row.metric_name for row in s.query(FlowMetric).all()}
    cov_count = s.query(CovarianceMatrix).count()
    s.close()
    engine.dispose()

    # Intraday metrics are suffixed and the plain EOD names are NOT written.
    assert "momentum_intraday" in names
    assert "regime_label_intraday" in names
    assert "momentum" not in names
    # Intraday runs skip covariance persistence.
    assert cov_count == 0


def test_engine_eod_writes_plain_metrics_and_covariance():
    engine, Session = _make_in_memory_db()
    _seed_prices(Session)

    with patch.object(engine_mod, "get_session", _mock_session_ctx(Session)), \
         patch.object(engine_mod, "init_db", lambda db_url=None: None):
        result = engine_mod.run_analysis(database_url="sqlite://", intraday=False)

    assert result["intraday"] is False

    s = Session()
    names = {row.metric_name for row in s.query(FlowMetric).all()}
    cov_count = s.query(CovarianceMatrix).count()
    s.close()
    engine.dispose()

    assert "momentum" in names
    assert "momentum_intraday" not in names
    assert cov_count > 0
