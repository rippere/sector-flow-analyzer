"""
Tests for sector_flow.cli — all Click commands and _build_query_output.

Each command imports its dependencies inside the function body, so patches
target the source module (e.g. sector_flow.pipeline.backfill), not the cli
module's local bindings.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sector_flow.cli import main, _build_query_output
from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData
from sector_flow.database.repository import ETFRepository, PriceRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def _session_ctx(Session):
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


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------

def test_cli_backfill_reports_rows_saved(runner):
    with patch("sector_flow.pipeline.backfill", return_value={"rows_saved": 42, "errors": []}):
        result = runner.invoke(main, ["backfill"])
    assert result.exit_code == 0
    assert "42" in result.output


def test_cli_backfill_prints_errors_to_stderr(runner):
    with patch("sector_flow.pipeline.backfill", return_value={"rows_saved": 5, "errors": ["timeout"]}):
        result = runner.invoke(main, ["backfill"])
    assert "timeout" in result.output


def test_cli_backfill_passes_days_arg(runner):
    received = []
    def fake(days, database_url):
        received.append(days)
        return {"rows_saved": 0, "errors": []}
    with patch("sector_flow.pipeline.backfill", side_effect=fake):
        runner.invoke(main, ["backfill", "--days", "14"])
    assert received == [14]


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def test_cli_ingest_reports_summary(runner):
    with patch("sector_flow.pipeline.run_daily", return_value={
        "yfinance_rows": 110, "ssga_rows": 15, "flow_rows_updated": 3, "errors": []
    }):
        result = runner.invoke(main, ["ingest"])
    assert result.exit_code == 0
    assert "110" in result.output
    assert "15" in result.output


def test_cli_ingest_prints_errors(runner):
    with patch("sector_flow.pipeline.run_daily", return_value={
        "yfinance_rows": 0, "ssga_rows": 0, "flow_rows_updated": 0, "errors": ["ssga down"]
    }):
        result = runner.invoke(main, ["ingest"])
    assert "ssga down" in result.output


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

_FAKE_ANALYSIS = {
    "market_regime": "rotation",
    "cohesion": 0.56,
    "significant_pairs": 27,
    "total_pairs": 55,
    "sector_regimes": {"XLK": "accumulation", "XLF": "neutral", "XLE": "distribution"},
    "sector_momentum": {"XLK": 1.0, "XLF": 0.05, "XLE": -0.5},
    "computed_at": "2026-05-15T10:00:00",
}


def test_cli_analyze_shows_regime(runner):
    with patch("sector_flow.analysis.engine.run_analysis", return_value=_FAKE_ANALYSIS):
        result = runner.invoke(main, ["analyze"])
    assert result.exit_code == 0
    assert "ROTATION" in result.output


def test_cli_analyze_shows_cohesion(runner):
    with patch("sector_flow.analysis.engine.run_analysis", return_value=_FAKE_ANALYSIS):
        result = runner.invoke(main, ["analyze"])
    assert "0.560" in result.output


def test_cli_analyze_shows_significant_pairs(runner):
    with patch("sector_flow.analysis.engine.run_analysis", return_value=_FAKE_ANALYSIS):
        result = runner.invoke(main, ["analyze"])
    assert "27" in result.output and "55" in result.output


def test_cli_analyze_shows_all_tickers(runner):
    with patch("sector_flow.analysis.engine.run_analysis", return_value=_FAKE_ANALYSIS):
        result = runner.invoke(main, ["analyze"])
    assert "XLK" in result.output
    assert "XLE" in result.output


def test_cli_analyze_window_passed(runner):
    received = []
    def fake(database_url, window):
        received.append(window)
        return _FAKE_ANALYSIS
    with patch("sector_flow.analysis.engine.run_analysis", side_effect=fake):
        runner.invoke(main, ["analyze", "--window", "60"])
    assert received == [60]


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_cli_status_shows_header(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Ticker" in result.output
    engine.dispose()


def test_cli_status_shows_all_etf_tickers(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["status"])

    for ticker, _, _ in SECTOR_ETFS:
        assert ticker in result.output
    engine.dispose()


def test_cli_status_shows_flow_ratio(runner):
    """With a price row that has net_inflow_usd, the flow ratio should show 1/1."""
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    first_ticker = SECTOR_ETFS[0][0]
    etf = s.query(SectorETF).filter_by(ticker=first_ticker).first()
    today_dt = datetime.combine(date.today(), datetime.min.time())
    s.add(PriceData(
        etf_id=etf.id, date=today_dt,
        open=100.0, high=102.0, low=99.0, close=101.0,
        volume=1e6, adjusted_close=101.0, net_inflow_usd=500_000.0,
    ))
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["status"])

    assert "1/1" in result.output
    engine.dispose()


# ---------------------------------------------------------------------------
# show-regime
# ---------------------------------------------------------------------------

def test_cli_show_regime_no_etfs_shows_hint(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-regime"])

    assert result.exit_code == 0
    assert "backfill" in result.output.lower() or "No ETFs" in result.output
    engine.dispose()


def test_cli_show_regime_with_etfs_shows_table(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-regime"])

    assert result.exit_code == 0
    assert "Ticker" in result.output
    assert "XLK" in result.output
    engine.dispose()


def test_cli_show_regime_na_when_no_metrics(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-regime"])

    assert "n/a" in result.output
    engine.dispose()


# ---------------------------------------------------------------------------
# show-flows
# ---------------------------------------------------------------------------

def test_cli_show_flows_no_etfs_shows_hint(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-flows"])

    assert result.exit_code == 0
    assert "backfill" in result.output.lower() or "No ETFs" in result.output
    engine.dispose()


def test_cli_show_flows_no_inflow_data_shows_message(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-flows"])

    assert "No net_inflow_usd" in result.output or "ingest" in result.output.lower()
    engine.dispose()


def test_cli_show_flows_ranks_tickers_by_inflow(runner):
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    today_dt = datetime.combine(date.today(), datetime.min.time())
    for i, (ticker, _, _) in enumerate(SECTOR_ETFS[:3]):
        etf = s.query(SectorETF).filter_by(ticker=ticker).first()
        s.add(PriceData(
            etf_id=etf.id, date=today_dt,
            open=100.0, high=102.0, low=99.0, close=101.0,
            volume=1e6, adjusted_close=101.0,
            net_inflow_usd=float((3 - i) * 1_000_000),
        ))
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-flows"])

    assert result.exit_code == 0
    assert "Rank" in result.output
    assert "Net Inflow" in result.output
    engine.dispose()


def test_cli_show_flows_lists_no_data_tickers(runner):
    """Tickers without flow data are listed at the bottom."""
    engine, Session = _make_db()
    ctx = _session_ctx(Session)
    s = Session()
    ETFRepository(s).seed_etfs()
    today_dt = datetime.combine(date.today(), datetime.min.time())
    # Only give flow to first ticker
    first_ticker = SECTOR_ETFS[0][0]
    etf = s.query(SectorETF).filter_by(ticker=first_ticker).first()
    s.add(PriceData(
        etf_id=etf.id, date=today_dt,
        open=100.0, high=102.0, low=99.0, close=101.0,
        volume=1e6, adjusted_close=101.0, net_inflow_usd=1_000_000.0,
    ))
    s.commit()
    s.close()

    with patch("sector_flow.database.session.get_session", ctx), \
         patch("sector_flow.database.session.init_db", lambda db_url=None: None):
        result = runner.invoke(main, ["show-flows"])

    assert "No flow data" in result.output
    engine.dispose()


# ---------------------------------------------------------------------------
# dashboard / serve / osc-bridge (blocking — verify routing only)
# ---------------------------------------------------------------------------

def test_cli_dashboard_invokes_run_dashboard(runner):
    with patch("sector_flow.visualizations.dashboard.run_dashboard") as mock_run:
        runner.invoke(main, ["dashboard"])
    mock_run.assert_called_once()


def test_cli_dashboard_passes_api_url(runner):
    with patch("sector_flow.visualizations.dashboard.run_dashboard") as mock_run:
        runner.invoke(main, ["dashboard", "--api-url", "http://myserver:9000"])
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["api_url"] == "http://myserver:9000"


def test_cli_serve_invokes_uvicorn(runner):
    with patch("uvicorn.run") as mock_run, \
         patch("sector_flow.config.settings") as s:
        s.api_host = "0.0.0.0"
        s.api_port = 8000
        s.log_level = "INFO"
        runner.invoke(main, ["serve"])
    mock_run.assert_called_once()


def test_cli_osc_bridge_runs_forever(runner):
    with patch("sector_flow.integrations.osc_bridge.OSCBridge") as MockBridge:
        mock = MagicMock()
        MockBridge.return_value = mock
        runner.invoke(main, ["osc-bridge"])
    mock.run_forever.assert_called_once()


def test_cli_osc_bridge_passes_port(runner):
    with patch("sector_flow.integrations.osc_bridge.OSCBridge") as MockBridge:
        mock = MagicMock()
        MockBridge.return_value = mock
        runner.invoke(main, ["osc-bridge", "--osc-port", "9001"])
    call_kwargs = MockBridge.call_args[1]
    assert call_kwargs["osc_port"] == 9001


# ---------------------------------------------------------------------------
# _build_query_output — unit-level edge cases
# ---------------------------------------------------------------------------

_REGIME_NO_LEADERS = {
    "market_regime": "neutral",
    "cohesion": 0.3,
    "significant_pairs": 2,
    "total_pairs": 10,
    "sector_regimes": {"XLK": "neutral", "XLF": "neutral"},
    "sector_momentum": {"XLK": 0.0, "XLF": 0.0},
}

_REGIME_NO_LAGGARDS = {
    "market_regime": "accumulation",
    "cohesion": 0.85,
    "significant_pairs": 8,
    "total_pairs": 10,
    "sector_regimes": {"XLK": "accumulation", "XLF": "breakout"},
    "sector_momentum": {"XLK": 1.0, "XLF": 0.5},
}


def _make_mock_response(data):
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = data
    return mock


def test_build_query_output_no_leaders():
    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response(_REGIME_NO_LEADERS)
        return _make_mock_response([])

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")
    assert "Momentum Leaders: none" in output


def test_build_query_output_no_laggards():
    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response(_REGIME_NO_LAGGARDS)
        return _make_mock_response([])

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")
    assert "Momentum Laggards: none" in output


def test_build_query_output_correlations_failure_silently_omitted():
    """When correlation endpoint raises, output is returned without a correlations section."""
    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response(_REGIME_NO_LEADERS)
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("504 timeout")
        return bad

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")

    assert "Sector Flow Analysis" in output
    assert "Strongest Correlations" not in output


def test_build_query_output_api_down_returns_error():
    import requests as req_mod
    with patch("sector_flow.cli.requests.get", side_effect=req_mod.ConnectionError("refused")):
        output = _build_query_output("http://fake")
    assert "ERROR" in output


def test_build_query_output_includes_rotation_thesis():
    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response({
                "market_regime": "rotation",
                "cohesion": 0.6,
                "significant_pairs": 5,
                "total_pairs": 10,
                "sector_regimes": {"XLK": "accumulation", "XLE": "distribution"},
                "sector_momentum": {"XLK": 1.0, "XLE": -0.5},
            })
        return _make_mock_response([])

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")
    assert "Rotation Thesis" in output


def test_build_query_output_shows_top_correlations():
    """When correlations are returned, the Strongest Correlations section is rendered."""
    fake_corrs = [
        {"ticker_a": "XLK", "ticker_b": "XLF", "correlation": 0.82},
        {"ticker_a": "XLI", "ticker_b": "XLB", "correlation": 0.79},
    ]

    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response({
                "market_regime": "rotation",
                "cohesion": 0.6,
                "significant_pairs": 5,
                "total_pairs": 10,
                "sector_regimes": {"XLK": "accumulation"},
                "sector_momentum": {"XLK": 1.0},
            })
        return _make_mock_response(fake_corrs)

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")
    assert "Strongest Correlations" in output
    assert "XLK ↔ XLF" in output


def test_build_query_output_distribution_sectors_mentioned():
    def fake_get(url, timeout=8):
        if "/regime" in url:
            return _make_mock_response({
                "market_regime": "rotation",
                "cohesion": 0.6,
                "significant_pairs": 5,
                "total_pairs": 10,
                "sector_regimes": {"XLK": "accumulation", "XLE": "distribution"},
                "sector_momentum": {"XLK": 1.0, "XLE": -0.5},
            })
        return _make_mock_response([])

    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        output = _build_query_output("http://fake")
    assert "distribution" in output.lower()
