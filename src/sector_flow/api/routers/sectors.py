"""Sector endpoints — list, detail, price history, flows."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from sector_flow.api.deps import get_db
from sector_flow.api.schemas import SectorDetail, SectorSummary, PriceRecord
from sector_flow.database.repository import ETFRepository, FlowMetricRepository, PriceRepository

router = APIRouter(prefix="/sectors", tags=["sectors"])

_REGIME_INV = {
    1.0: "accumulation",
    2.0: "breakout",
    0.0: "neutral",
    -1.0: "distribution",
    -2.0: "breakdown",
}


def _build_summary(etf, price_repo: PriceRepository, flow_repo: FlowMetricRepository) -> dict:
    rows = price_repo.get_price_data(etf.id)
    latest_row = rows[-1] if rows else None

    regime_val = flow_repo.get_latest(etf.id, "regime_label")
    momentum = flow_repo.get_latest(etf.id, "momentum")
    cohesion = flow_repo.get_latest(etf.id, "cohesion")

    regime_str = _REGIME_INV.get(regime_val) if regime_val is not None else None

    return {
        "ticker": etf.ticker,
        "sector_name": etf.sector_name,
        "regime": regime_str,
        "momentum": momentum,
        "cohesion": cohesion,
        "latest_close": latest_row.close if latest_row else None,
        "latest_date": latest_row.date.isoformat() if latest_row else None,
        "row_count": len(rows),
    }


@router.get("", response_model=list[SectorSummary])
def list_sectors(db: Session = Depends(get_db)) -> list[SectorSummary]:
    """Return summary data for all 11 sector ETFs."""
    etf_repo = ETFRepository(db)
    price_repo = PriceRepository(db)
    flow_repo = FlowMetricRepository(db)

    etfs = sorted(etf_repo.get_all(), key=lambda e: e.ticker)
    return [SectorSummary(**_build_summary(etf, price_repo, flow_repo)) for etf in etfs]


@router.get("/{ticker}", response_model=SectorDetail)
def get_sector(ticker: str, db: Session = Depends(get_db)) -> SectorDetail:
    """Return detail for a single sector ETF."""
    etf_repo = ETFRepository(db)
    price_repo = PriceRepository(db)
    flow_repo = FlowMetricRepository(db)

    etf = etf_repo.get_by_ticker(ticker.upper())
    if etf is None:
        raise HTTPException(status_code=404, detail=f"Sector '{ticker}' not found")

    data = _build_summary(etf, price_repo, flow_repo)
    data["sector_code"] = etf.sector_code
    return SectorDetail(**data)


@router.get("/{ticker}/prices", response_model=list[PriceRecord])
def get_prices(
    ticker: str,
    start: Optional[str] = Query(None, description="ISO date, e.g. 2024-01-01"),
    end: Optional[str] = Query(None, description="ISO date, e.g. 2024-12-31"),
    limit: int = Query(100, ge=1, le=10000),
    db: Session = Depends(get_db),
) -> list[PriceRecord]:
    """Return OHLCV price history for a sector ETF."""
    etf_repo = ETFRepository(db)
    price_repo = PriceRepository(db)

    etf = etf_repo.get_by_ticker(ticker.upper())
    if etf is None:
        raise HTTPException(status_code=404, detail=f"Sector '{ticker}' not found")

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None

    rows = price_repo.get_price_data(etf.id, start=start_dt, end=end_dt)
    rows = rows[-limit:]  # apply limit from tail

    return [
        PriceRecord(
            date=r.date.isoformat(),
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            adjusted_close=r.adjusted_close,
            net_inflow_usd=r.net_inflow_usd,
            shares_outstanding=r.shares_outstanding,
            aum_usd=r.aum_usd,
        )
        for r in rows
    ]


@router.get("/{ticker}/flows", response_model=list[PriceRecord])
def get_flows(
    ticker: str,
    limit: int = Query(100, ge=1, le=10000),
    db: Session = Depends(get_db),
) -> list[PriceRecord]:
    """Return only rows that have net_inflow_usd populated."""
    etf_repo = ETFRepository(db)
    price_repo = PriceRepository(db)

    etf = etf_repo.get_by_ticker(ticker.upper())
    if etf is None:
        raise HTTPException(status_code=404, detail=f"Sector '{ticker}' not found")

    rows = price_repo.get_price_data(etf.id)
    flow_rows = [r for r in rows if r.net_inflow_usd is not None]
    flow_rows = flow_rows[-limit:]

    return [
        PriceRecord(
            date=r.date.isoformat(),
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            adjusted_close=r.adjusted_close,
            net_inflow_usd=r.net_inflow_usd,
            shares_outstanding=r.shares_outstanding,
            aum_usd=r.aum_usd,
        )
        for r in flow_rows
    ]
