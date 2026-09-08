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
async def test_get_sectors_returns_all(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors")
    assert resp.status_code == 200
    assert len(resp.json()) == len(SECTOR_ETFS)


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


@pytest.mark.asyncio
async def test_get_prices_unknown_ticker_404(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/BOGUS/prices")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_flows_unknown_ticker_404(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/sectors/BOGUS/flows")
    assert resp.status_code == 404


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
    assert len(matrix) == len(SECTOR_ETFS)
    for ticker, row in matrix.items():
        assert len(row) == len(SECTOR_ETFS), f"Row for {ticker} should have {len(SECTOR_ETFS)} entries"
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
# Analysis flows tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_analysis_flows_structure(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/flows")
    assert resp.status_code == 200
    data = resp.json()
    assert "flows" in data
    assert "correlations" in data
    assert "computed_at" in data


@pytest.mark.asyncio
async def test_get_analysis_flows_all_sectors(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/flows")
    data = resp.json()
    tickers_returned = {f["ticker"] for f in data["flows"]}
    expected = {t for t, _, _ in SECTOR_ETFS}
    assert tickers_returned == expected


@pytest.mark.asyncio
async def test_get_analysis_flows_momentum_rank_in_range(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/flows")
    data = resp.json()
    for entry in data["flows"]:
        rank = entry["momentum_rank"]
        assert 0.0 <= rank <= 1.0, f"{entry['ticker']} rank {rank} out of [0,1]"


@pytest.mark.asyncio
async def test_get_analysis_flows_absent_momentum_does_not_outrank(app_with_db):
    """A sector with no momentum data must not receive a higher momentum_rank
    than sectors with real (even all-negative) momentum — regression for
    audit:sector-flow-analyzer:bugs:get-flows-momentum-rank-absent-data-outranks.
    """
    from unittest.mock import MagicMock
    from sqlalchemy.orm import sessionmaker
    from sector_flow.database.models import SectorETF

    app, engine = app_with_db

    Session = sessionmaker(bind=engine)
    session = Session()
    etf_ids_by_ticker = {e.ticker: e.id for e in session.query(SectorETF).all()}
    session.close()

    # Every sector has real, negative momentum except XLK, which has none.
    momentum_by_ticker = {t: -0.5 for t in etf_ids_by_ticker}
    momentum_by_ticker["XLF"] = -0.1  # best (highest) of the real values
    del momentum_by_ticker["XLK"]

    def _get_latest(etf_id, metric_name):
        if metric_name != "momentum":
            return None
        ticker = next((t for t, i in etf_ids_by_ticker.items() if i == etf_id), None)
        return momentum_by_ticker.get(ticker)

    mock_flow_inst = MagicMock()
    mock_flow_inst.get_latest.side_effect = _get_latest

    with patch("sector_flow.api.routers.analysis.FlowMetricRepository", return_value=mock_flow_inst):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/analysis/flows")

    data = resp.json()
    by_ticker = {f["ticker"]: f for f in data["flows"]}

    assert by_ticker["XLK"]["momentum"] is None
    assert by_ticker["XLK"]["momentum_rank"] == 0.5
    assert by_ticker["XLF"]["momentum_rank"] == 1.0
    assert by_ticker["XLK"]["momentum_rank"] < by_ticker["XLF"]["momentum_rank"]


@pytest.mark.asyncio
async def test_get_analysis_flows_correlations_filtered_and_sorted(app_with_analysis):
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/analysis/flows")
    data = resp.json()
    pairs = data["correlations"]
    for pair in pairs:
        assert abs(pair["correlation"]) >= 0.3
        assert pair["ticker_a"] < pair["ticker_b"]
    if len(pairs) > 1:
        keys = [(p["ticker_a"], p["ticker_b"]) for p in pairs]
        assert keys == sorted(keys)


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


@pytest.mark.asyncio
async def test_dashboard_redirects(app_with_db):
    app, _ = app_with_db
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        follow_redirects=False,
    ) as ac:
        resp = await ac.get("/dashboard")
    assert resp.status_code in (301, 302, 307, 308)
    assert "/static/dashboard.html" in resp.headers.get("location", "")


def test_app_lifespan_startup_and_shutdown():
    """TestClient triggers the lifespan; verifies init_db + task lifecycle called."""
    from starlette.testclient import TestClient as SyncTestClient
    from sector_flow.api.app import app

    with patch("sector_flow.api.app.init_db") as mock_init_db:
        with patch("sector_flow.api.routers.ws.start_background_tasks") as mock_start:
            with patch("sector_flow.api.routers.ws.stop_background_tasks") as mock_stop:
                with SyncTestClient(app) as client:
                    resp = client.get("/health")
                    assert resp.status_code == 200

    mock_init_db.assert_called_once()
    mock_start.assert_called_once()
    mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline ingest endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_pipeline_ingest_mocked(seeded_engine):
    """POST /pipeline/ingest with run_daily mocked — covers ingest_pipeline body."""
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db

    orig_engine, orig_factory = _patch_session_globals(seeded_engine)
    app.dependency_overrides[get_db] = _make_override(seeded_engine)

    fake_result = {"yfinance_rows": 11, "ssga_rows": 0, "flow_rows_updated": 11, "errors": []}

    try:
        with patch("sector_flow.pipeline.run_daily", return_value=fake_result):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/pipeline/ingest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["detail"]["yfinance_rows"] == 11
    finally:
        app.dependency_overrides.clear()
        _restore_session_globals(orig_engine, orig_factory)


# ---------------------------------------------------------------------------
# get_db dependency
# ---------------------------------------------------------------------------

def test_get_db_yields_session():
    """get_db() body is covered when get_session is not mocked out."""
    from unittest.mock import MagicMock, patch
    from sector_flow.api.deps import get_db

    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_session)
    mock_cm.__exit__ = MagicMock(return_value=None)

    with patch("sector_flow.api.deps.get_session", return_value=mock_cm):
        gen = get_db()
        session = next(gen)

    assert session is mock_session


# ---------------------------------------------------------------------------
# Regime classification branches (lines 72-76 in analysis.py)
# ---------------------------------------------------------------------------

def _make_regime_mocks(cohesion: float, momentum_by_ticker: dict[str, float]):
    """Return (etf_inst, flow_inst, cov_inst) that steer regime classification.

    momentum_by_ticker maps real sector ticker → momentum value so that the
    TICKERS loop in _get_regime_snapshot can find each ETF in etf_map.
    """
    from unittest.mock import MagicMock
    import uuid as _uuid

    etfs = []
    for ticker, mom in momentum_by_ticker.items():
        e = MagicMock()
        e.id = _uuid.uuid4()
        e.ticker = ticker
        etfs.append((e, mom))

    mock_etf_inst = MagicMock()
    mock_etf_inst.get_all.return_value = [e for e, _ in etfs]

    id_to_mom = {str(e.id): mom for e, mom in etfs}

    def _get_latest(etf_id, metric_name):
        if metric_name == "cohesion":
            return cohesion
        if metric_name == "momentum":
            return id_to_mom.get(str(etf_id), 0.0)
        return None  # regime_label

    mock_flow_inst = MagicMock()
    mock_flow_inst.get_latest.side_effect = _get_latest

    mock_cov_inst = MagicMock()
    mock_cov_inst.get_latest_matrix.return_value = []

    return mock_etf_inst, mock_flow_inst, mock_cov_inst


@pytest.mark.asyncio
async def test_regime_trending_when_high_cohesion(app_with_db):
    """avg_cohesion > 0.6 → market_regime = 'trending' (line 72)."""
    app, _ = app_with_db
    etf_inst, flow_inst, cov_inst = _make_regime_mocks(
        cohesion=0.75, momentum_by_ticker={"XLK": 0.1, "XLF": -0.1, "XLE": 0.0}
    )

    with patch("sector_flow.api.routers.analysis.ETFRepository", return_value=etf_inst):
        with patch("sector_flow.api.routers.analysis.FlowMetricRepository", return_value=flow_inst):
            with patch("sector_flow.api.routers.analysis.CovarianceRepository", return_value=cov_inst):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    resp = await ac.get("/analysis/regime")

    assert resp.status_code == 200
    assert resp.json()["market_regime"] == "trending"


@pytest.mark.asyncio
async def test_regime_rotation_when_bullish_majority(app_with_db):
    """bullish > bearish, cohesion <= 0.6 → market_regime = 'rotation' (line 74)."""
    app, _ = app_with_db
    # 2 positive, 1 negative momentum → bullish majority
    etf_inst, flow_inst, cov_inst = _make_regime_mocks(
        cohesion=0.3, momentum_by_ticker={"XLK": 0.5, "XLF": 0.4, "XLE": -0.2}
    )

    with patch("sector_flow.api.routers.analysis.ETFRepository", return_value=etf_inst):
        with patch("sector_flow.api.routers.analysis.FlowMetricRepository", return_value=flow_inst):
            with patch("sector_flow.api.routers.analysis.CovarianceRepository", return_value=cov_inst):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    resp = await ac.get("/analysis/regime")

    assert resp.status_code == 200
    assert resp.json()["market_regime"] == "rotation"


@pytest.mark.asyncio
async def test_regime_risk_off_when_bearish_majority(app_with_db):
    """bearish > bullish, cohesion <= 0.6 → market_regime = 'risk_off' (line 76)."""
    app, _ = app_with_db
    # 2 negative, 1 positive momentum → bearish majority
    etf_inst, flow_inst, cov_inst = _make_regime_mocks(
        cohesion=0.2, momentum_by_ticker={"XLK": -0.5, "XLF": -0.3, "XLE": 0.1}
    )

    with patch("sector_flow.api.routers.analysis.ETFRepository", return_value=etf_inst):
        with patch("sector_flow.api.routers.analysis.FlowMetricRepository", return_value=flow_inst):
            with patch("sector_flow.api.routers.analysis.CovarianceRepository", return_value=cov_inst):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    resp = await ac.get("/analysis/regime")

    assert resp.status_code == 200
    assert resp.json()["market_regime"] == "risk_off"


@pytest.mark.asyncio
async def test_regime_neutral_when_balanced(app_with_db):
    """bullish == bearish, cohesion <= 0.6 → market_regime = 'neutral' (line 78)."""
    app, _ = app_with_db
    # 1 positive, 1 negative, 1 zero momentum → tied → neutral
    etf_inst, flow_inst, cov_inst = _make_regime_mocks(
        cohesion=0.3, momentum_by_ticker={"XLK": 0.5, "XLF": -0.5, "XLE": 0.0}
    )

    with patch("sector_flow.api.routers.analysis.ETFRepository", return_value=etf_inst):
        with patch("sector_flow.api.routers.analysis.FlowMetricRepository", return_value=flow_inst):
            with patch("sector_flow.api.routers.analysis.CovarianceRepository", return_value=cov_inst):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                    resp = await ac.get("/analysis/regime")

    assert resp.status_code == 200
    assert resp.json()["market_regime"] == "neutral"
