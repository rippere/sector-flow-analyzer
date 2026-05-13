"""FastAPI application factory for the Sector Flow Analyzer."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from sector_flow.database.session import init_db
from sector_flow.api.routers import sectors, analysis, pipeline, ws as ws_router


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
