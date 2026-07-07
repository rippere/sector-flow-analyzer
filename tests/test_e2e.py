"""End-to-end: ingestion → analysis on a shared in-memory DB (collectors mocked).

Exercises the integrated path including the adaptive-threshold wiring, asserting
a valid regime and populated FlowMetric rows — no network required.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData, FlowMetric
from sector_flow.database.repository import ETFRepository
import sector_flow.pipeline as pipeline_mod
import sector_flow.analysis.engine as engine_mod

_VALID_REGIMES = {"crisis", "risk_on", "risk_off", "rotation", "neutral"}


def _make_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def _ctx(Session):
    @contextmanager
    def _c(db_url=None):
        s = Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
    return _c


def _seed_history(Session, days=80):
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    start = datetime.combine(date.today() - timedelta(days=days + 2), datetime.min.time())
    for i, (ticker, _, _) in enumerate(SECTOR_ETFS):
        etf = s.query(SectorETF).filter_by(ticker=ticker).first()
        for d in range(days):
            px = 100.0 + 10.0 * math.sin((d + i * 4) / 8.0) + d * 0.05
            s.add(PriceData(
                etf_id=etf.id, date=start + timedelta(days=d),
                open=px, high=px * 1.01, low=px * 0.99,
                close=px, volume=1e6, adjusted_close=px,
            ))
    s.commit()
    s.close()


def test_ingest_then_analyze_end_to_end():
    engine, Session = _make_db()
    _seed_history(Session)

    # Mock collectors: yfinance returns no new bars, SSGA returns an empty frame.
    mock_yf = MagicMock()
    mock_yf.return_value.fetch.return_value = []
    mock_ssga = MagicMock()
    mock_ssga.return_value.fetch_snapshot.return_value = pd.DataFrame({"Ticker": []})

    with patch.object(pipeline_mod, "get_session", _ctx(Session)), \
         patch.object(pipeline_mod, "init_db", lambda db_url=None: None), \
         patch.object(pipeline_mod, "YFinanceCollector", mock_yf), \
         patch.object(pipeline_mod, "SSGACollector", mock_ssga):
        summary = pipeline_mod.run_daily(database_url="sqlite://")

    assert "errors" in summary

    with patch.object(engine_mod, "get_session", _ctx(Session)), \
         patch.object(engine_mod, "init_db", lambda db_url=None: None):
        result = engine_mod.run_analysis(database_url="sqlite://")

    assert result["market_regime"] in _VALID_REGIMES
    assert 0.0 <= result["cohesion"] <= 1.0
    assert len(result["sector_momentum"]) > 0

    s = Session()
    metric_names = {r.metric_name for r in s.query(FlowMetric).all()}
    s.close()
    engine.dispose()
    assert {"momentum", "regime_label", "cohesion"}.issubset(metric_names)
