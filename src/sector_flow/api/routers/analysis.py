"""Analysis endpoints — regime, correlations, and correlation matrix."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sector_flow.api.deps import get_db
from sector_flow.api.schemas import CorrelationPair, FlowAnalysisResponse, MarketRegimeResponse, SectorFlowEntry
from sector_flow.database.models import SECTOR_ETFS
from sector_flow.database.repository import (
    CovarianceRepository,
    ETFRepository,
    FlowMetricRepository,
    PriceRepository,
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
        "computed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }


@router.get("/flows", response_model=FlowAnalysisResponse)
def get_flows(db: Session = Depends(get_db)) -> FlowAnalysisResponse:
    """
    Primary objective data endpoint.

    Returns per-sector capital flows, AUM, and momentum rank (cross-sectional),
    plus significant correlation pairs. No regime interpretation.

    momentum_rank is min-max normalized across all 11 sectors: the sector with
    the highest raw momentum gets 1.0, lowest gets 0.0. If all momentum values
    are equal or missing, all sectors return 0.5.
    """
    etf_repo = ETFRepository(db)
    flow_repo = FlowMetricRepository(db)
    price_repo = PriceRepository(db)
    cov_repo = CovarianceRepository(db)

    etfs = etf_repo.get_all()
    etf_map = {etf.ticker: etf for etf in etfs}
    id_to_ticker = {etf.id: etf.ticker for etf in etfs}

    # Collect raw momentum values for cross-sectional normalization
    raw_momentum: dict[str, float | None] = {}
    raw_flow: dict[str, float | None] = {}
    raw_aum: dict[str, float | None] = {}

    for ticker in TICKERS:
        etf = etf_map.get(ticker)
        if etf is None:
            continue

        # momentum lives in FlowMetric (stored by analysis engine). Keep
        # missing values as None here — defaulting to 0.0 would fold "no
        # data" into the cross-sectional min/max and could let an absent
        # sector outrank sectors with real (e.g. all-negative) momentum.
        mom = flow_repo.get_latest(etf.id, "momentum")
        raw_momentum[ticker] = mom

        # net_inflow_usd and aum_usd live in PriceData (from SSGA collector).
        # The market bar lands hours before each day's SSGA snapshot (and the
        # snapshot file itself lags a day), so the newest row is often bar-only —
        # serve the latest SNAPSHOT-BEARING row rather than None.
        rows = price_repo.get_price_data(etf.id)
        snap = next((r for r in reversed(rows) if r.aum_usd is not None), None)
        if snap is not None:
            raw_flow[ticker] = snap.net_inflow_usd
            raw_aum[ticker] = snap.aum_usd
        else:
            raw_flow[ticker] = None
            raw_aum[ticker] = None

    # Min-max normalize momentum across all 11 sectors to [0.0, 1.0].
    # Sectors with no momentum data are excluded from the min/max so an
    # absent value can't masquerade as the top (or bottom) performer.
    mom_values = [v for v in raw_momentum.values() if v is not None]
    mom_min = min(mom_values) if mom_values else 0.0
    mom_max = max(mom_values) if mom_values else 0.0
    mom_range = mom_max - mom_min

    def _momentum_rank(ticker: str) -> float:
        raw = raw_momentum.get(ticker)
        if raw is None or mom_range == 0.0:
            return 0.5
        return (raw - mom_min) / mom_range

    def _momentum(ticker: str) -> float | None:
        raw = raw_momentum.get(ticker)
        return round(raw, 4) if raw is not None else None

    flows = [
        SectorFlowEntry(
            ticker=ticker,
            net_inflow_usd=raw_flow.get(ticker),
            aum_usd=raw_aum.get(ticker),
            momentum_rank=round(_momentum_rank(ticker), 4),
            momentum=_momentum(ticker),
        )
        for ticker in TICKERS
        if ticker in etf_map
    ]

    # Significant correlation pairs (reuse same query as /analysis/correlations)
    latest_pairs = cov_repo.get_latest_matrix(window=30)
    correlations = []
    for p in latest_pairs:
        ta = id_to_ticker.get(p.etf_a_id, "")
        tb = id_to_ticker.get(p.etf_b_id, "")
        if not ta or not tb:
            continue
        if ta > tb:
            ta, tb = tb, ta
        corr = p.correlation or 0.0
        if abs(corr) < 0.3:
            continue
        correlations.append(
            CorrelationPair(
                ticker_a=ta,
                ticker_b=tb,
                correlation=corr,
                covariance=p.covariance or 0.0,
                p_value=None,
                window_days=p.window_days,
                significant=True,
            )
        )
    correlations.sort(key=lambda r: (r.ticker_a, r.ticker_b))

    return FlowAnalysisResponse(
        flows=flows,
        correlations=correlations,
        computed_at=datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    )


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
