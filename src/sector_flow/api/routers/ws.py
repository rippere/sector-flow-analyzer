"""WebSocket endpoint — live regime broadcast."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])

# Module-level connected client set
_connected: Set[WebSocket] = set()

# Server start time for uptime calculation
_start_time: float = time.monotonic()


def _build_regime_message() -> dict:
    """Read latest regime from DB and build a regime_update message dict."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sector_flow.config import settings
    from sector_flow.database.models import SECTOR_ETFS
    from sector_flow.database.repository import ETFRepository, FlowMetricRepository, CovarianceRepository

    _REGIME_INV = {
        1.0: "accumulation",
        2.0: "breakout",
        0.0: "neutral",
        -1.0: "distribution",
        -2.0: "breakdown",
    }
    TICKERS = [t for t, _, _ in SECTOR_ETFS]

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        etf_repo = ETFRepository(session)
        flow_repo = FlowMetricRepository(session)
        cov_repo = CovarianceRepository(session)

        etfs = etf_repo.get_all()
        etf_map = {etf.ticker: etf for etf in etfs}

        sector_regimes: dict[str, str] = {}
        sector_momentum: dict[str, float] = {}
        cohesion_vals = []

        for ticker in TICKERS:
            etf = etf_map.get(ticker)
            if etf is None:
                continue
            regime_val = flow_repo.get_latest(etf.id, "regime_label")
            momentum = flow_repo.get_latest(etf.id, "momentum")
            cohesion = flow_repo.get_latest(etf.id, "cohesion")

            sector_regimes[ticker] = (
                _REGIME_INV.get(regime_val, "neutral") if regime_val is not None else "neutral"
            )
            sector_momentum[ticker] = momentum if momentum is not None else 0.0
            if cohesion is not None:
                cohesion_vals.append(cohesion)

        avg_cohesion = sum(cohesion_vals) / len(cohesion_vals) if cohesion_vals else 0.0

        latest_pairs = cov_repo.get_latest_matrix(window=30)
        total_pairs = len(latest_pairs)
        sig_pairs = sum(
            1 for p in latest_pairs if p.correlation is not None and abs(p.correlation) >= 0.3
        )

        bullish = sum(1 for m in sector_momentum.values() if m > 0)
        bearish = sum(1 for m in sector_momentum.values() if m < 0)
        if avg_cohesion > 0.6:
            market_regime = "trending"
        elif bullish > bearish:
            market_regime = "rotation"
        elif bearish > bullish:
            market_regime = "risk_off"
        else:
            market_regime = "neutral"

        return {
            "type": "regime_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "market_regime": market_regime,
                "cohesion": avg_cohesion,
                "significant_pairs": sig_pairs,
                "total_pairs": total_pairs,
                "sector_regimes": sector_regimes,
                "sector_momentum": sector_momentum,
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
            msg = await asyncio.get_event_loop().run_in_executor(None, _build_regime_message)
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
            "timestamp": datetime.utcnow().isoformat(),
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
            snap = await asyncio.get_event_loop().run_in_executor(None, _build_regime_message)
            await websocket.send_json(snap)
        except Exception as exc:
            logger.error(f"[ws] initial snapshot error: {exc}")
            await websocket.send_json(
                {
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
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
