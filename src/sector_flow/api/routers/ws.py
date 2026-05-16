"""WebSocket endpoint — live regime broadcast."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Set

from datetime import timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])

# Module-level connected client set
_connected: Set[WebSocket] = set()

# Server start time for uptime calculation
_start_time: float = time.monotonic()


def _build_flow_message() -> dict:
    """Read latest objective flow data from DB and build a flow_update message dict."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sector_flow.config import settings
    from sector_flow.database.models import SECTOR_ETFS
    from sector_flow.database.repository import ETFRepository, FlowMetricRepository, PriceRepository, CovarianceRepository

    TICKERS = [t for t, _, _ in SECTOR_ETFS]

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        etf_repo = ETFRepository(session)
        flow_repo = FlowMetricRepository(session)
        price_repo = PriceRepository(session)
        cov_repo = CovarianceRepository(session)

        etfs = etf_repo.get_all()
        etf_map = {etf.ticker: etf for etf in etfs}
        id_to_ticker = {etf.id: etf.ticker for etf in etfs}

        raw_momentum: dict[str, float] = {}
        raw_flow: dict[str, float | None] = {}
        raw_aum: dict[str, float | None] = {}

        for ticker in TICKERS:
            etf = etf_map.get(ticker)
            if etf is None:
                continue
            mom = flow_repo.get_latest(etf.id, "momentum")
            raw_momentum[ticker] = mom if mom is not None else 0.0
            rows = price_repo.get_price_data(etf.id)
            if rows:
                latest = rows[-1]
                raw_flow[ticker] = latest.net_inflow_usd
                raw_aum[ticker] = latest.aum_usd
            else:
                raw_flow[ticker] = None
                raw_aum[ticker] = None

        # Cross-sectional min-max momentum rank [0.0, 1.0]
        mom_values = list(raw_momentum.values())
        mom_min = min(mom_values) if mom_values else 0.0
        mom_max = max(mom_values) if mom_values else 0.0
        mom_range = mom_max - mom_min

        def _rank(ticker: str) -> float:
            if mom_range == 0.0:
                return 0.5
            return round((raw_momentum.get(ticker, 0.0) - mom_min) / mom_range, 4)

        sectors = {
            ticker: {
                "net_inflow_usd": raw_flow.get(ticker),
                "aum_usd": raw_aum.get(ticker),
                "momentum_rank": _rank(ticker),
            }
            for ticker in TICKERS
            if ticker in etf_map
        }

        latest_pairs = cov_repo.get_latest_matrix(window=30)
        correlations = []
        corr_vals = []
        for p in latest_pairs:
            ta = id_to_ticker.get(p.etf_a_id, "")
            tb = id_to_ticker.get(p.etf_b_id, "")
            if not ta or not tb:
                continue
            if ta > tb:
                ta, tb = tb, ta
            corr = p.correlation or 0.0
            corr_vals.append(abs(corr))
            if abs(corr) >= 0.3:
                correlations.append({"ticker_a": ta, "ticker_b": tb, "correlation": corr})

        avg_cohesion = round(sum(corr_vals) / len(corr_vals), 4) if corr_vals else 0.0

        return {
            "type": "flow_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "sectors": sectors,
                "correlations": correlations,
                "meta": {"avg_cohesion": avg_cohesion},
            },
        }
    finally:
        session.close()
        engine.dispose()


async def _broadcast_loop() -> None:
    """Send regime updates every 60 s to all connected clients."""
    while True:
        await asyncio.sleep(60)
        if not _connected:
            continue
        try:
            msg = await asyncio.get_event_loop().run_in_executor(None, _build_flow_message)
            dead = set()
            for ws in list(_connected):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.add(ws)
            _connected.difference_update(dead)
        except Exception as exc:
            logger.error(f"[ws] broadcast error: {exc}")


async def _heartbeat_loop() -> None:
    """Send heartbeat every 10 s to all connected clients."""
    while True:
        await asyncio.sleep(10)
        if not _connected:
            continue
        uptime = int(time.monotonic() - _start_time)
        msg = {
            "type": "heartbeat",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"uptime_seconds": uptime},
        }
        dead = set()
        for ws in list(_connected):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        _connected.difference_update(dead)


# Background task references (stored to avoid GC)
_broadcast_task: asyncio.Task | None = None
_heartbeat_task: asyncio.Task | None = None


def start_background_tasks() -> None:
    """Start broadcast + heartbeat loops. Called from app lifespan."""
    global _broadcast_task, _heartbeat_task
    _broadcast_task = asyncio.create_task(_broadcast_loop())
    _heartbeat_task = asyncio.create_task(_heartbeat_loop())


def stop_background_tasks() -> None:
    """Cancel background tasks on shutdown."""
    if _broadcast_task:
        _broadcast_task.cancel()
    if _heartbeat_task:
        _heartbeat_task.cancel()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live regime updates.

    On connect: sends the current regime snapshot immediately.
    Then broadcasts updates every 60 s and heartbeats every 10 s.
    """
    await websocket.accept()
    _connected.add(websocket)
    logger.info(f"[ws] client connected — total: {len(_connected)}")

    try:
        # Immediate snapshot
        try:
            snap = await asyncio.get_event_loop().run_in_executor(None, _build_flow_message)
            await websocket.send_json(snap)
        except Exception as exc:
            logger.error(f"[ws] initial snapshot error: {exc}")
            await websocket.send_json(
                {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"message": str(exc)},
                }
            )

        # Keep connection alive by receiving (client may send pings)
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info("[ws] client disconnected")
    finally:
        _connected.discard(websocket)
