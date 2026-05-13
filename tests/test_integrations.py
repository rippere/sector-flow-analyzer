"""
Phase 5 integration tests — OSC bridge, Prometheus metrics, Alfred query CLI.

All external calls are mocked. The API layer tests use the same in-memory
SQLite setup as the Phase 3 test suite.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from click.testing import CliRunner
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_api.py helpers)
# ---------------------------------------------------------------------------


def _make_seeded_engine():
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
def app_with_analysis(seeded_engine):
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db
    from sector_flow.analysis.engine import run_analysis

    orig_engine, orig_factory = _patch_session_globals(seeded_engine)
    run_analysis()

    app.dependency_overrides[get_db] = _make_override(seeded_engine)
    yield app, seeded_engine

    app.dependency_overrides.clear()
    _restore_session_globals(orig_engine, orig_factory)


@pytest.fixture()
def app_with_db(seeded_engine):
    from sector_flow.api.app import app
    from sector_flow.api.deps import get_db

    app.dependency_overrides[get_db] = _make_override(seeded_engine)
    yield app, seeded_engine
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Deliverable 1: OSC Bridge
# ---------------------------------------------------------------------------

_FAKE_REGIME = {
    "market_regime": "rotation",
    "cohesion": 0.56,
    "significant_pairs": 27,
    "total_pairs": 55,
    "sector_regimes": {
        "XLK": "accumulation",
        "XLF": "neutral",
        "XLE": "distribution",
        "XLV": "accumulation",
        "XLY": "distribution",
        "XLP": "breakout",
        "XLI": "neutral",
        "XLB": "neutral",
        "XLRE": "neutral",
        "XLU": "neutral",
        "XLC": "neutral",
    },
    "sector_momentum": {
        "XLK": 1.0,
        "XLF": 0.1,
        "XLE": -0.5,
        "XLV": 1.0,
        "XLY": -0.38,
        "XLP": 1.0,
        "XLI": 0.05,
        "XLB": 0.0,
        "XLRE": 0.0,
        "XLU": -0.98,
        "XLC": -0.63,
    },
    "computed_at": "2026-05-13T10:00:00",
}

_FAKE_CORRELATIONS = [
    {"ticker_a": "XLK", "ticker_b": "XLF", "correlation": 0.82, "covariance": 0.001, "p_value": None, "window_days": 30, "significant": True},
    {"ticker_a": "XLI", "ticker_b": "XLB", "correlation": 0.79, "covariance": 0.001, "p_value": None, "window_days": 30, "significant": True},
]

_FAKE_SECTORS = [
    {"ticker": t, "sector_name": n, "regime": None, "momentum": None, "cohesion": None,
     "latest_close": None, "latest_date": None, "row_count": 0}
    for t, n, _ in SECTOR_ETFS
]


def _make_mock_response(data):
    """Build a minimal mock requests.Response-like object."""
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


def test_osc_bridge_broadcast_builds_messages():
    """OSCBridge.broadcast_snapshot() should call UDP client and return > 0 messages."""
    from sector_flow.integrations.osc_bridge import OSCBridge

    def fake_get(url, timeout=5):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        if "/sectors" in url and "/sectors/" not in url:
            return _make_mock_response(_FAKE_SECTORS)
        return _make_mock_response({})

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=fake_get):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient") as mock_client_cls:
            mock_sock = MagicMock()
            mock_client_cls.return_value = mock_sock

            bridge = OSCBridge(api_url="http://localhost:8000", osc_host="127.0.0.1", osc_port=9000)
            count = bridge.broadcast_snapshot()

    # Should send at least 1 message per sector (5 each) + meta messages
    assert count > 0, f"Expected > 0 messages, got {count}"
    assert count >= 11 * 5, f"Expected at least {11 * 5} sector messages, got {count}"


def test_osc_bridge_returns_zero_on_api_failure():
    """When the API is unreachable, broadcast_snapshot() returns 0."""
    from sector_flow.integrations.osc_bridge import OSCBridge
    import requests as req_mod

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=req_mod.ConnectionError("down")):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient"):
            bridge = OSCBridge()
            count = bridge.broadcast_snapshot()

    assert count == 0


def test_osc_bridge_sends_expected_addresses():
    """Verify that sector weight and meta regime addresses are sent."""
    from sector_flow.integrations.osc_bridge import OSCBridge

    sent_addresses: list[str] = []

    def fake_get(url, timeout=5):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response(_FAKE_SECTORS)

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=fake_get):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient"):
            bridge = OSCBridge()
            # Intercept _send to capture addresses
            original_send = bridge._send

            def capturing_send(address, *args):
                sent_addresses.append(address)
                return original_send(address, *args)

            bridge._send = capturing_send
            bridge.broadcast_snapshot()

    assert "/sector/XLK/weight" in sent_addresses
    assert "/meta/regime" in sent_addresses
    assert "/meta/cohesion" in sent_addresses
    assert "/meta/dominant_sector" in sent_addresses
    # Pair messages
    assert "/sector/pair/XLK/XLF/correlation" in sent_addresses


# ---------------------------------------------------------------------------
# Deliverable 2: Prometheus /metrics endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint_status_and_content_type(app_with_db):
    """GET /metrics returns 200 with text/plain content-type."""
    app, _ = app_with_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_prometheus_metrics_contains_expected_keys(app_with_analysis):
    """GET /metrics body contains all required metric names."""
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/metrics")

    body = resp.text
    assert "sector_flow_momentum" in body
    assert "sector_flow_regime_code" in body
    assert "sector_flow_cohesion" in body
    assert "sector_flow_significant_pairs" in body
    assert "sector_flow_data_freshness_seconds" in body


@pytest.mark.asyncio
async def test_prometheus_metrics_contains_ticker_labels(app_with_analysis):
    """Metric lines include ticker labels for known sector ETFs."""
    app, _ = app_with_analysis
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/metrics")

    body = resp.text
    for ticker in ("XLK", "XLF", "XLE"):
        assert f'ticker="{ticker}"' in body, f"Expected ticker label for {ticker}"


# ---------------------------------------------------------------------------
# Deliverable 3: Alfred query CLI command
# ---------------------------------------------------------------------------


def test_query_command_output_rotation_thesis():
    """sector-flow query should output 'Rotation Thesis' section."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query", "what is the current sector rotation thesis"])

    assert result.exit_code == 0, f"CLI exited with {result.exit_code}: {result.output}"
    assert "Rotation Thesis" in result.output


def test_query_command_shows_market_regime():
    """Output should mention the market regime from the API."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query"])

    assert "ROTATION" in result.output


def test_query_command_api_unreachable():
    """When API is down, output should contain an error message, not crash."""
    from sector_flow.cli import main
    import requests as req_mod

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=req_mod.ConnectionError("refused")):
        result = runner.invoke(main, ["query"])

    assert result.exit_code == 0
    assert "ERROR" in result.output or "error" in result.output.lower()


def test_query_command_shows_momentum_leaders():
    """Output should list at least one momentum leader."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query"])

    assert "Momentum Leaders" in result.output
    assert "XLK" in result.output
