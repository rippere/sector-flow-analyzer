"""Analysis endpoints — regime, correlations, and correlation matrix."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sector_flow.api.deps import get_db
from sector_flow.api.schemas import CorrelationPair, MarketRegimeResponse
from sector_flow.database.models import SECTOR_ETFS
from sector_flow.database.repository import (
    CovarianceRepository,
    ETFRepository,
    FlowMetricRepository,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])

TICKERS = [t for t, _, _ in SECTOR_ETFS]

_REGIME_INV = {
    1.0: "accumulation",
    2.0: "breakout",
    0.0: "neutral",
    -1.0: "distribution",
    -2.0: "breakdown",
}


def _get_regime_snapshot(db: Session) -> dict:
    """Read the latest per-sector metrics and assemble a regime snapshot dict."""
    etf_repo = ETFRepository(db)
    flow_repo = FlowMetricRepository(db)
    cov_repo = CovarianceRepository(db)

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

        sector_regimes[ticker] = _REGIME_INV.get(regime_val, "neutral") if regime_val is not None else "neutral"
        sector_momentum[ticker] = momentum if momentum is not None else 0.0
        if cohesion is not None:
            cohesion_vals.append(cohesion)

    avg_cohesion = sum(cohesion_vals) / len(cohesion_vals) if cohesion_vals else 0.0

    # Latest covariance pairs to derive significant_pairs / total_pairs
    latest_pairs = cov_repo.get_latest_matrix(window=30)
    total_pairs = len(latest_pairs)

    # A pair is significant if |correlation| >= 0.3 (mirrors significance.py default)
    sig_pairs = sum(1 for p in latest_pairs if p.correlation is not None and abs(p.correlation) >= 0.3)

    # Derive market regime from cohesion + momentum
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
        "market_regime": market_regime,
        "cohesion": avg_cohesion,
        "significant_pairs": sig_pairs,
        "total_pairs": total_pairs,
        "sector_regimes": sector_regimes,
        "sector_momentum": sector_momentum,
        "computed_at": datetime.utcnow().isoformat(),
    }


@router.get("/regime", response_model=MarketRegimeResponse)
def get_regime(db: Session = Depends(get_db)) -> MarketRegimeResponse:
    """Return the latest market regime snapshot from stored DB data (no recompute)."""
    snap = _get_regime_snapshot(db)
    return MarketRegimeResponse(**snap)


@router.get("/correlations", response_model=list[CorrelationPair])
def get_correlations(db: Session = Depends(get_db)) -> list[CorrelationPair]:
    """Return significant correlation pairs from the latest CovarianceMatrix snapshot."""
    etf_repo = ETFRepository(db)
    cov_repo = CovarianceRepository(db)

    etfs = etf_repo.get_all()
    id_to_ticker = {etf.id: etf.ticker for etf in etfs}

    latest_pairs = cov_repo.get_latest_matrix(window=30)

    result = []
    for p in latest_pairs:
        ta = id_to_ticker.get(p.etf_a_id, "")
        tb = id_to_ticker.get(p.etf_b_id, "")
        if not ta or not tb:
            continue
        # Enforce ticker_a < ticker_b
        if ta > tb:
            ta, tb = tb, ta

        corr = p.correlation or 0.0
        cov = p.covariance or 0.0
        significant = abs(corr) >= 0.3

        result.append(
            CorrelationPair(
                ticker_a=ta,
                ticker_b=tb,
                correlation=corr,
                covariance=cov,
                p_value=None,  # not stored in DB directly
                window_days=p.window_days,
                significant=significant,
            )
        )

    # Return only significant pairs, sorted
    result = [r for r in result if r.significant]
    result.sort(key=lambda r: (r.ticker_a, r.ticker_b))
    return result


@router.get("/matrix")
def get_matrix(db: Session = Depends(get_db)) -> dict:
    """Return the full 11x11 correlation matrix (ticker → ticker → correlation float)."""
    etf_repo = ETFRepository(db)
    cov_repo = CovarianceRepository(db)

    etfs = etf_repo.get_all()
    id_to_ticker = {etf.id: etf.ticker for etf in etfs}
    all_tickers = sorted([etf.ticker for etf in etfs])

    # Initialise with 1.0 on diagonal, 0.0 elsewhere
    matrix: dict[str, dict[str, float]] = {
        t: {other: (1.0 if other == t else 0.0) for other in all_tickers}
        for t in all_tickers
    }

    latest_pairs = cov_repo.get_latest_matrix(window=30)
    for p in latest_pairs:
        ta = id_to_ticker.get(p.etf_a_id)
        tb = id_to_ticker.get(p.etf_b_id)
        if ta and tb and p.correlation is not None:
            matrix[ta][tb] = p.correlation
            matrix[tb][ta] = p.correlation

    return matrix
