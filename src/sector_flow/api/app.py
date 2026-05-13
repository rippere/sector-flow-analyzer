"""FastAPI application factory for the Sector Flow Analyzer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from fastapi.responses import Response
from sqlalchemy.orm import Session

from sector_flow.database.session import init_db
from sector_flow.api.deps import get_db
from sector_flow.api.routers import sectors, analysis, pipeline, ws as ws_router
from sector_flow.api.routers.analysis import _get_regime_snapshot
from sector_flow.database.models import SECTOR_ETFS
from sector_flow.database.repository import ETFRepository, PriceRepository


_REGIME_ENCODING: dict[str, int] = {
    "accumulation": 2,
    "breakout": 1,
    "neutral": 0,
    "distribution": -1,
    "breakdown": -2,
}

TICKERS = [t for t, _, _ in SECTOR_ETFS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic."""
    # Initialise database tables
    init_db()
    # Start background WS broadcast/heartbeat tasks
    ws_router.start_background_tasks()
    yield
    # Shutdown
    ws_router.stop_background_tasks()


app = FastAPI(
    title="Sector Flow Analyzer",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(sectors.router)
app.include_router(analysis.router)
app.include_router(pipeline.router)
app.include_router(ws_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics(db: Session = Depends(get_db)) -> Response:
    """Prometheus text-format metrics endpoint (no external library required)."""
    snap = _get_regime_snapshot(db)

    sector_momentum: dict[str, float] = snap.get("sector_momentum", {})
    sector_regimes: dict[str, str] = snap.get("sector_regimes", {})
    cohesion: float = snap.get("cohesion", 0.0)
    sig_pairs: int = snap.get("significant_pairs", 0)

    # Data freshness: seconds since the latest OHLCV row across all sectors
    etf_repo = ETFRepository(db)
    price_repo = PriceRepository(db)
    etfs = etf_repo.get_all()
    latest_dates = []
    for etf in etfs:
        d = price_repo.get_latest_date(etf.id)
        if d is not None:
            latest_dates.append(d)

    if latest_dates:
        most_recent = max(latest_dates)
        # most_recent may be naive (UTC) — treat as UTC
        if most_recent.tzinfo is None:
            most_recent = most_recent.replace(tzinfo=timezone.utc)
        freshness_seconds = int((datetime.now(tz=timezone.utc) - most_recent).total_seconds())
    else:
        freshness_seconds = -1

    lines: list[str] = []

    # --- sector_flow_momentum ---
    lines.append("# HELP sector_flow_momentum Sector momentum score [-1, 1]")
    lines.append("# TYPE sector_flow_momentum gauge")
    for ticker in sorted(TICKERS):
        mom = sector_momentum.get(ticker, 0.0)
        lines.append(f'sector_flow_momentum{{ticker="{ticker}"}} {mom}')

    # --- sector_flow_regime_code ---
    lines.append("# HELP sector_flow_regime_code Sector regime encoded as int (accumulation=2,breakout=1,neutral=0,distribution=-1,breakdown=-2)")
    lines.append("# TYPE sector_flow_regime_code gauge")
    for ticker in sorted(TICKERS):
        regime = sector_regimes.get(ticker, "neutral")
        code = _REGIME_ENCODING.get(regime, 0)
        lines.append(f'sector_flow_regime_code{{ticker="{ticker}"}} {code}')

    # --- sector_flow_cohesion ---
    lines.append("# HELP sector_flow_cohesion Market cohesion score [0, 1]")
    lines.append("# TYPE sector_flow_cohesion gauge")
    lines.append(f"sector_flow_cohesion {cohesion}")

    # --- sector_flow_significant_pairs ---
    lines.append("# HELP sector_flow_significant_pairs Number of statistically significant correlation pairs")
    lines.append("# TYPE sector_flow_significant_pairs gauge")
    lines.append(f"sector_flow_significant_pairs {sig_pairs}")

    # --- sector_flow_data_freshness_seconds ---
    lines.append("# HELP sector_flow_data_freshness_seconds Seconds since last OHLCV data point")
    lines.append("# TYPE sector_flow_data_freshness_seconds gauge")
    lines.append(f"sector_flow_data_freshness_seconds {freshness_seconds}")

    content = "\n".join(lines) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4")
