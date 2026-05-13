"""
Phase 3 API tests — FastAPI REST + WebSocket layer.

Uses an in-memory SQLite database seeded with synthetic data.
External calls (yfinance, SSGA) are mocked.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seeded_engine():
    """
    Create an in-memory SQLite engine with StaticPool so every connection
    shares the same in-process database, then seed it with all 11 ETFs +
    60 rows of OHLCV each.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    base_date = datetime(2024, 1, 2)
    for ticker, sector_name, sector_code in SECTOR_ETFS:
        etf = SectorETF(ticker=ticker, sector_name=sector_name, sector_code=sector_code)
        session.add(etf)
        session.flush()
        for i in range(60):
            session.add(
                PriceData(
                    etf_id=etf.id,
                    date=base_date + timedelta(days=i),
                    open=100.0 + i * 0.1,
                    high=102.0 + i * 0.1,
                    low=99.0 + i * 0.1,
                    close=101.0 + i * 0.1,
                    volume=1_000_000.0,
                    adjusted_close=101.0 + i * 0.1,
                )
            )

    session.commit()
    session.close()
    return engine


def _make_override(engine):
    """Return a get_db dependency override bound to `engine`."""
    TestSession = sessionmaker(bind=engine)

    def override_get_db():
        session = TestSession()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return override_get_db


def _patch_session_globals(engine):
    """
    Redirect sector_flow.database.session module globals to our test engine
    so that any call to get_session() / init_db() uses it.
    Returns (orig_engine, orig_factory) for later restoration.
    """
    import sector_flow.database.session as sess_mod

    orig_engine = sess_mod._engine
    orig_factory = sess_mod._SessionFactory

    TestSession = sessionmaker(bind=engine)
    sess_mod._engine = engine
    sess_mod._SessionFactory = TestSession

    return orig_engine, orig_factory


def _restore_session_globals(orig_engine, orig_factory):
    import sector_flow.database.session as sess_mod

    sess_mod._engine = orig_engine
    sess_mod._SessionFactory = orig_factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_engine():
    engine = _make_seeded_engine()
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def app_with_db(seeded_engine):
    """FastAPI app with get_db overridden to the seeded engine."""
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db

    app.dependency_overrides[get_db] = _make_override(seeded_engine)
    yield app, seeded_engine
    app.dependency_overrides.clear()


@pytest.fixture()
def app_with_analysis(seeded_engine):
    """
    FastAPI app where the analysis engine has been run against the seeded DB.
    Both dependency override and session globals are patched.
    """
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db
    from sector_flow.analysis.engine import run_analysis

    orig_engine, orig_factory = _patch_session_globals(seeded_engine)
    # Run analysis synchronously before handing app to tests
    run_analysis()

    app.dependency_overrides[get_db] = _make_override(seeded_engine)
    yield app, seeded_engine

    app.dependency_overrides.clear()
    _restore_session_globals(orig_engine, orig_factory)


# ---------------------------------------------------------------------------
# Sector tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_sectors_returns_11(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors")
    assert resp.status_code == 200
    assert len(resp.json()) == 11


@pytest.mark.asyncio
async def test_get_sector_xlk(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/XLK")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "XLK"
    assert data["sector_name"] == "Technology"


@pytest.mark.asyncio
async def test_get_sector_unknown_404(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/FAKE")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_prices_xlk(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/XLK/prices")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for field in ("date", "open", "high", "low", "close", "volume", "adjusted_close"):
        assert field in data[0]


@pytest.mark.asyncio
async def test_get_prices_limit(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/XLK/prices?limit=5")
    assert resp.status_code == 200
    assert len(resp.json()) <= 5


@pytest.mark.asyncio
async def test_get_flows_xlk(app_with_db):
    """Flows endpoint returns list (empty since no SSGA data in seed)."""
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/XLK/flows")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_analysis_regime(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/regime")
    assert resp.status_code == 200
    data = resp.json()
    assert "market_regime" in data
    assert data["market_regime"] in ("trending", "rotation", "risk_off", "neutral")
    assert "cohesion" in data
    assert "sector_regimes" in data
    assert "sector_momentum" in data


@pytest.mark.asyncio
async def test_get_correlations(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/correlations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for pair in data:
        assert pair["ticker_a"] < pair["ticker_b"], "ticker_a must be lexically before ticker_b"
        assert pair["significant"] is True


@pytest.mark.asyncio
async def test_get_matrix_shape(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/matrix")
    assert resp.status_code == 200
    matrix = resp.json()
    assert len(matrix) == 11
    for ticker, row in matrix.items():
        assert len(row) == 11, f"Row for {ticker} should have 11 entries"
        assert row[ticker] == 1.0, f"Diagonal entry for {ticker} should be 1.0"


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_pipeline_analyze(app_with_db, seeded_engine):
    app, engine = app_with_db
    orig_engine, orig_factory = _patch_session_globals(engine)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/pipeline/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "market_regime" in data["detail"]
    finally:
        _restore_session_globals(orig_engine, orig_factory)


@pytest.mark.asyncio
async def test_post_pipeline_backfill_mocked(seeded_engine):
    """POST /pipeline/backfill with yfinance mocked — no network calls."""
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db

    fake_records = [
        {
            "date": datetime(2024, 3, 1),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1_000_000.0,
            "adjusted_close": 101.0,
        }
    ]

    orig_engine, orig_factory = _patch_session_globals(seeded_engine)
    app.dependency_overrides[get_db] = _make_override(seeded_engine)

    try:
        with patch("sector_flow.pipeline.YFinanceCollector.fetch", return_value=fake_records):
            with patch("sector_flow.pipeline.init_db", return_value=None):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    resp = await ac.post("/pipeline/backfill", json={"days": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "rows_saved" in data["detail"]
        assert data["detail"]["rows_saved"] >= 0
    finally:
        app.dependency_overrides.clear()
        _restore_session_globals(orig_engine, orig_factory)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
