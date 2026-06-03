"""Pydantic schemas for the Sector Flow Analyzer REST API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class SectorSummary(BaseModel):
    ticker: str
    sector_name: str
    regime: Optional[str]
    momentum: Optional[float]
    cohesion: Optional[float]
    latest_close: Optional[float]
    latest_date: Optional[str]
    row_count: int


class SectorDetail(SectorSummary):
    sector_code: Optional[str]


class PriceRecord(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float
    net_inflow_usd: Optional[float]
    shares_outstanding: Optional[float]
    aum_usd: Optional[float]


class CorrelationPair(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation: float
    covariance: float
    p_value: Optional[float]
    window_days: int
    significant: bool


class MarketRegimeResponse(BaseModel):
    market_regime: str
    cohesion: float
    significant_pairs: int
    total_pairs: int
    sector_regimes: dict[str, str]
    sector_momentum: dict[str, float]
    computed_at: str


class PipelineResult(BaseModel):
    status: str
    detail: dict


class SectorFlowEntry(BaseModel):
    ticker: str
    net_inflow_usd: Optional[float]   # None if SSGA data not yet ingested
    aum_usd: Optional[float]
    momentum_rank: float               # [0.0, 1.0] cross-sectional min-max of raw momentum
    momentum: Optional[float] = None   # raw engine momentum, ~[-1, 1] — fixed dial scale


class FlowAnalysisResponse(BaseModel):
    flows: list[SectorFlowEntry]
    correlations: list[CorrelationPair]
    computed_at: str


class WebSocketMessage(BaseModel):
    type: str  # "flow_update" | "heartbeat" | "error"
    timestamp: str
    data: dict
