"""Tests for the WebSocket router: _build_flow_message, loops, start/stop, endpoint."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

import sector_flow.api.routers.ws as ws_mod
from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData, FlowMetric


from sector_flow.database.models import CovarianceMatrix

FAKE_FLOW_MSG = {
    "type": "flow_update",
    "timestamp": "2024-01-01T00:00:00",
    "data": {
        "sectors": {"XLK": {"net_inflow_usd": None, "aum_usd": None, "momentum_rank": 0.5}},
        "correlations": [],
        "meta": {"avg_cohesion": 0.0},
    },
}


def _make_seeded_engine_with_covariance():
    """Seeded engine that also has one CovarianceMatrix pair."""
    engine = _make_seeded_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    etfs = session.query(SectorETF).order_by(SectorETF.ticker).all()
    etf_a, etf_b = etfs[0], etfs[1]
    session.add(
        CovarianceMatrix(
            etf_a_id=etf_a.id,
            etf_b_id=etf_b.id,
            window_days=30,
            computed_at=datetime(2024, 3, 1),
            covariance=0.001,
            correlation=0.85,
        )
    )
    # Add a pair where the ticker with higher id comes first (exercises ta>tb swap)
    etf_z, etf_y = etfs[-1], etfs[-2]
    session.add(
        CovarianceMatrix(
            etf_a_id=etf_z.id,
            etf_b_id=etf_y.id,
            window_days=30,
            computed_at=datetime(2024, 3, 1),
            covariance=0.0005,
            correlation=0.15,  # below 0.3 threshold → skipped from correlations list
        )
    )
    session.commit()
    session.close()
    return engine


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
        for i in range(30):
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


# ---------------------------------------------------------------------------
# _build_flow_message
# ---------------------------------------------------------------------------


class TestBuildFlowMessage:
    def test_returns_flow_update_type(self):
        engine = _make_seeded_engine()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        assert result["type"] == "flow_update"
        assert "timestamp" in result
        assert "data" in result

    def test_data_has_expected_keys(self):
        engine = _make_seeded_engine()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        assert "sectors" in result["data"]
        assert "correlations" in result["data"]
        assert "meta" in result["data"]

    def test_all_tickers_in_sectors(self):
        engine = _make_seeded_engine()
        expected_tickers = {t for t, _, _ in SECTOR_ETFS}
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        assert set(result["data"]["sectors"].keys()) == expected_tickers

    def test_sectors_have_expected_fields(self):
        engine = _make_seeded_engine()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        for sector_data in result["data"]["sectors"].values():
            assert "net_inflow_usd" in sector_data
            assert "aum_usd" in sector_data
            assert "momentum_rank" in sector_data

    def test_no_flow_metrics_gives_half_rank(self):
        """No FlowMetric rows → all momentum == 0.0 → guard returns 0.5."""
        engine = _make_seeded_engine()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        for sector_data in result["data"]["sectors"].values():
            assert sector_data["momentum_rank"] == 0.5

    def test_non_uniform_momentum_computes_real_rank(self):
        """Non-zero momentum range hits the actual rank formula (line 75 new)."""
        engine = _make_seeded_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        etfs = session.query(SectorETF).all()
        analysis_date = datetime(2024, 3, 1)
        for i, etf in enumerate(etfs):
            session.add(FlowMetric(
                etf_id=etf.id,
                date=analysis_date,
                metric_name="momentum",
                value=float(i),
            ))
        session.commit()
        session.close()

        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()

        ranks = [d["momentum_rank"] for d in result["data"]["sectors"].values()]
        assert not all(r == 0.5 for r in ranks), "expected non-uniform ranks"

    def test_significant_correlation_included(self):
        """Covers lines 89-98: correlation loop, ta>tb swap, |corr|>=0.3 filter."""
        engine = _make_seeded_engine_with_covariance()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        assert len(result["data"]["correlations"]) >= 1
        assert result["data"]["meta"]["avg_cohesion"] > 0.0
        for pair in result["data"]["correlations"]:
            assert abs(pair["correlation"]) >= 0.3

    def test_empty_db_gives_zero_cohesion(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        for ticker, sector_name, sector_code in SECTOR_ETFS:
            session.add(SectorETF(ticker=ticker, sector_name=sector_name, sector_code=sector_code))
        session.commit()
        session.close()
        with patch("sqlalchemy.create_engine", return_value=engine):
            result = ws_mod._build_flow_message()
        assert result["data"]["meta"]["avg_cohesion"] == 0.0


# ---------------------------------------------------------------------------
# start / stop background tasks
# ---------------------------------------------------------------------------


class TestBackgroundTasks:
    def test_stop_cancels_both_tasks(self):
        mock_b = MagicMock()
        mock_h = MagicMock()
        orig_b, orig_h = ws_mod._broadcast_task, ws_mod._heartbeat_task
        ws_mod._broadcast_task = mock_b
        ws_mod._heartbeat_task = mock_h
        try:
            ws_mod.stop_background_tasks()
            mock_b.cancel.assert_called_once()
            mock_h.cancel.assert_called_once()
        finally:
            ws_mod._broadcast_task = orig_b
            ws_mod._heartbeat_task = orig_h

    def test_stop_is_noop_when_tasks_are_none(self):
        orig_b, orig_h = ws_mod._broadcast_task, ws_mod._heartbeat_task
        ws_mod._broadcast_task = None
        ws_mod._heartbeat_task = None
        try:
            ws_mod.stop_background_tasks()  # must not raise
        finally:
            ws_mod._broadcast_task = orig_b
            ws_mod._heartbeat_task = orig_h

    @pytest.mark.asyncio
    async def test_start_creates_two_tasks(self):
        with patch("asyncio.create_task", return_value=MagicMock()) as mock_create:
            ws_mod.start_background_tasks()
        assert mock_create.call_count == 2
        ws_mod._broadcast_task = None
        ws_mod._heartbeat_task = None


# ---------------------------------------------------------------------------
# _broadcast_loop
# ---------------------------------------------------------------------------


class TestBroadcastLoop:
    @pytest.mark.asyncio
    async def test_no_clients_skips_build(self):
        ws_mod._connected.clear()
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch("sector_flow.api.routers.ws._build_flow_message") as mock_build:
                with pytest.raises(asyncio.CancelledError):
                    await ws_mod._broadcast_loop()

        mock_build.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_message_to_connected_client(self):
        ws_mod._connected.clear()
        mock_ws = AsyncMock()
        ws_mod._connected.add(mock_ws)
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch("sector_flow.api.routers.ws._build_flow_message", return_value=FAKE_FLOW_MSG):
                with pytest.raises(asyncio.CancelledError):
                    await ws_mod._broadcast_loop()

        mock_ws.send_json.assert_called_once_with(FAKE_FLOW_MSG)
        ws_mod._connected.discard(mock_ws)

    @pytest.mark.asyncio
    async def test_dead_client_removed_on_send_error(self):
        ws_mod._connected.clear()
        dead_ws = AsyncMock()
        dead_ws.send_json.side_effect = Exception("gone")
        ws_mod._connected.add(dead_ws)
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch("sector_flow.api.routers.ws._build_flow_message", return_value=FAKE_FLOW_MSG):
                with pytest.raises(asyncio.CancelledError):
                    await ws_mod._broadcast_loop()

        assert dead_ws not in ws_mod._connected

    @pytest.mark.asyncio
    async def test_build_exception_caught_loop_continues(self):
        ws_mod._connected.clear()
        mock_ws = AsyncMock()
        ws_mod._connected.add(mock_ws)
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with patch(
                "sector_flow.api.routers.ws._build_flow_message",
                side_effect=RuntimeError("DB error"),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await ws_mod._broadcast_loop()

        ws_mod._connected.discard(mock_ws)


# ---------------------------------------------------------------------------
# _heartbeat_loop
# ---------------------------------------------------------------------------


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_no_clients_skips_send(self):
        ws_mod._connected.clear()
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(asyncio.CancelledError):
                await ws_mod._heartbeat_loop()

    @pytest.mark.asyncio
    async def test_sends_heartbeat_message(self):
        ws_mod._connected.clear()
        mock_ws = AsyncMock()
        ws_mod._connected.add(mock_ws)
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(asyncio.CancelledError):
                await ws_mod._heartbeat_loop()

        mock_ws.send_json.assert_called_once()
        msg = mock_ws.send_json.call_args[0][0]
        assert msg["type"] == "heartbeat"
        assert "uptime_seconds" in msg["data"]
        ws_mod._connected.discard(mock_ws)

    @pytest.mark.asyncio
    async def test_dead_client_removed(self):
        ws_mod._connected.clear()
        dead_ws = AsyncMock()
        dead_ws.send_json.side_effect = Exception("gone")
        ws_mod._connected.add(dead_ws)
        calls = [0]

        async def fast_sleep(delay):
            calls[0] += 1
            if calls[0] >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            with pytest.raises(asyncio.CancelledError):
                await ws_mod._heartbeat_loop()

        assert dead_ws not in ws_mod._connected


# ---------------------------------------------------------------------------
# websocket_live endpoint
# ---------------------------------------------------------------------------


class TestWebSocketLive:
    @pytest.fixture()
    def test_app(self):
        from sector_flow.api.app import app
        yield app

    def test_connect_receives_initial_snapshot(self, test_app):
        ws_mod._connected.clear()
        with patch("sector_flow.api.routers.ws._build_flow_message", return_value=FAKE_FLOW_MSG):
            with TestClient(test_app).websocket_connect("/ws/live") as ws:
                msg = ws.receive_json()
        assert msg["type"] == "flow_update"
        ws_mod._connected.clear()

    def test_snapshot_error_sends_error_message(self, test_app):
        ws_mod._connected.clear()
        with patch(
            "sector_flow.api.routers.ws._build_flow_message",
            side_effect=RuntimeError("DB unavailable"),
        ):
            with TestClient(test_app).websocket_connect("/ws/live") as ws:
                msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "DB unavailable" in msg["data"]["message"]
        ws_mod._connected.clear()

    def test_client_added_then_removed_on_disconnect(self, test_app):
        ws_mod._connected.clear()
        with patch("sector_flow.api.routers.ws._build_flow_message", return_value=FAKE_FLOW_MSG):
            with TestClient(test_app).websocket_connect("/ws/live") as ws:
                ws.receive_json()
                connected_during = len(ws_mod._connected)
        assert connected_during == 1
        assert len(ws_mod._connected) == 0
