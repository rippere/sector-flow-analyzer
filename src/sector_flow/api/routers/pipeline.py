"""Pipeline endpoints — backfill, ingest, analyze."""

from __future__ import annotations

from fastapi import APIRouter, Body

from sector_flow.api.schemas import PipelineResult

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/backfill", response_model=PipelineResult)
def backfill_pipeline(days: int = Body(90, embed=True)) -> PipelineResult:
    """Pull historical OHLCV data for all sectors."""
    from sector_flow.pipeline import backfill as run_backfill

    result = run_backfill(days=days)
    return PipelineResult(status="ok", detail=result)


@router.post("/ingest", response_model=PipelineResult)
def ingest_pipeline() -> PipelineResult:
    """Run the daily ingestion pipeline (OHLCV + SSGA flow snapshot)."""
    from sector_flow.pipeline import run_daily

    result = run_daily()
    return PipelineResult(status="ok", detail=result)


@router.post("/analyze", response_model=PipelineResult)
def analyze_pipeline() -> PipelineResult:
    """Run the full analysis engine and persist results."""
    from sector_flow.analysis.engine import run_analysis

    result = run_analysis()
    return PipelineResult(status="ok", detail=result)
