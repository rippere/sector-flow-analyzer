"""
Analysis engine orchestrator.

Ties together covariance computation, significance filtering, regime
classification, and database persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
from typing import Optional

import pandas as pd
from loguru import logger

from sector_flow.database.models import SECTOR_ETFS, PriceData
from sector_flow.database.repository import ETFRepository, PriceRepository, FlowMetricRepository, CovarianceRepository
from sector_flow.database.session import get_session, init_db

from sector_flow.analysis.covariance import build_price_matrix, compute_rolling_correlation, compute_pairwise_latest
from sector_flow.analysis.significance import filter_significant_correlations
from sector_flow.analysis.regime import (
    classify_market_regime,
    classify_sector_regime,
    compute_adaptive_thresholds,
    compute_cohesion,
    compute_momentum,
)

TICKERS = [t for t, _, _ in SECTOR_ETFS]


def run_analysis(
    database_url: Optional[str] = None,
    window: int = 30,
    intraday: bool = False,
) -> dict:
    """
    Run the full analysis pipeline and persist results.

    Steps
    -----
    1. Load all PriceData from DB into a price matrix.
    2. Compute rolling correlation (all pairs).
    3. Filter for statistical significance.
    4. Classify market regime.
    5. Classify per-sector regimes.
    6. Compute cohesion score.
    7. Store latest covariance pairs into CovarianceMatrix table.
    8. Store per-sector regime + momentum into FlowMetric table.

    Returns
    -------
    dict with keys:
        market_regime, cohesion, sector_regimes, sector_momentum,
        significant_pairs, total_pairs
    """
    init_db(database_url)

    with get_session(database_url) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        flow_repo = FlowMetricRepository(session)
        cov_repo = CovarianceRepository(session)

        etf_repo.seed_etfs()

        # 1. Load price data for all tickers
        all_etfs = etf_repo.get_all()
        etf_map = {etf.ticker: etf for etf in all_etfs}

        # Collect all PriceData records across all sectors
        all_price_records: list[PriceData] = []
        for ticker in TICKERS:
            etf = etf_map.get(ticker)
            if etf is None:
                continue
            rows = price_repo.get_price_data(etf.id)
            # Eagerly attach ticker info for build_price_matrix
            for row in rows:
                row.etf = etf
            all_price_records.extend(rows)

        # 2. Build wide-format price matrix
        price_df = build_price_matrix(all_price_records, tickers=TICKERS)

        if price_df.empty:
            logger.warning("No price data found — skipping analysis.")
            return {
                "market_regime": "neutral",
                "cohesion": 0.0,
                "sector_regimes": {},
                "sector_momentum": {},
                "significant_pairs": 0,
                "total_pairs": 0,
            }

        # 3. Compute rolling correlation
        corr_df = compute_rolling_correlation(price_df, window=window)
        total_pairs = len(corr_df[corr_df["date"] == corr_df["date"].max()]) if not corr_df.empty else 0

        # 4. Filter for significance
        if not corr_df.empty:
            latest_date = corr_df["date"].max()
            latest_corr = corr_df[corr_df["date"] == latest_date].copy()
            sig_df = filter_significant_correlations(latest_corr)
        else:
            sig_df = pd.DataFrame()

        significant_pairs = len(sig_df)

        # 5. Classify market regime
        # Build optional flow_df from net_inflow_usd
        flow_rows = []
        for ticker in TICKERS:
            etf = etf_map.get(ticker)
            if etf is None:
                continue
            rows = price_repo.get_price_data(etf.id)
            flow_vals = [r.net_inflow_usd for r in rows if r.net_inflow_usd is not None]
            if flow_vals:
                flow_rows.append({"ticker": ticker, "net_inflow_usd": flow_vals[-1]})

        flow_df = pd.DataFrame(flow_rows) if flow_rows else None

        # Adaptive crisis/rotation thresholds from the rolling-correlation
        # history (per-date mean|corr| and std), so regime calls are judged
        # against this market's own recent behaviour rather than fixed cutoffs.
        crisis_thr = rotation_thr = None
        if not corr_df.empty and "date" in corr_df.columns:
            by_date = corr_df.groupby("date")["correlation"]
            cohesion_hist = by_date.apply(lambda s: s.abs().mean())
            corr_std_hist = by_date.std()
            crisis_thr, rotation_thr = compute_adaptive_thresholds(cohesion_hist, corr_std_hist)

        market_regime = classify_market_regime(
            sig_df if not sig_df.empty else corr_df,
            flow_df=flow_df,
            crisis_threshold=crisis_thr,
            rotation_threshold=rotation_thr,
        )

        # 6. Compute cohesion
        cohesion = compute_cohesion(sig_df if not sig_df.empty else corr_df)

        # 7. Classify per-sector regimes and momentum
        sector_regimes: dict[str, str] = {}
        sector_momentum: dict[str, float] = {}

        analysis_date = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        # Intraday snapshots are written under distinct metric names so they
        # refresh in place (one row/day) without clobbering the authoritative
        # EOD rows that `analyze` writes under the plain names.
        metric_sfx = "_intraday" if intraday else ""

        for ticker in TICKERS:
            etf = etf_map.get(ticker)
            if etf is None:
                continue

            price_col = price_df.get(ticker) if ticker in price_df.columns else None
            if price_col is None or price_col.dropna().empty:
                sector_regimes[ticker] = "neutral"
                sector_momentum[ticker] = 0.0
                continue

            # Flow and volume series from price_df are not available separately;
            # load raw records for flow/volume
            price_records = price_repo.get_price_data(etf.id)
            if price_records:
                flow_ser = pd.Series(
                    [r.net_inflow_usd for r in price_records],
                    index=pd.to_datetime([r.date for r in price_records]),
                )
                vol_ser = pd.Series(
                    [r.volume for r in price_records],
                    index=pd.to_datetime([r.date for r in price_records]),
                )
            else:
                flow_ser = None
                vol_ser = None

            regime = classify_sector_regime(
                ticker=ticker,
                price_series=price_col.dropna(),
                flow_series=flow_ser,
                volume_series=vol_ser,
            )
            mom = compute_momentum(price_col.dropna())

            sector_regimes[ticker] = regime
            sector_momentum[ticker] = mom

            # 8. Persist to FlowMetric table (intraday snapshots use suffixed names)
            flow_repo.save_metric(etf.id, analysis_date, f"regime_label{metric_sfx}", _regime_to_float(regime))
            flow_repo.save_metric(etf.id, analysis_date, f"momentum{metric_sfx}", mom)
            flow_repo.save_metric(etf.id, analysis_date, f"cohesion{metric_sfx}", cohesion)

        # 9. Persist covariance pairs to CovarianceMatrix (EOD only — intraday
        #    runs skip this to avoid bloating the matrix with per-interval rows).
        if not intraday:
            latest_pairs = compute_pairwise_latest(price_df, window=window)
            enriched_pairs = []
            computed_at = _utcnow()
            for pair in latest_pairs:
                etf_a = etf_map.get(pair["ticker_a"])
                etf_b = etf_map.get(pair["ticker_b"])
                if etf_a is None or etf_b is None:
                    continue
                enriched_pairs.append(
                    {
                        "etf_a_id": etf_a.id,
                        "etf_b_id": etf_b.id,
                        "window_days": pair["window_days"],
                        "computed_at": computed_at,
                        "covariance": pair["covariance"],
                        "correlation": pair["correlation"],
                    }
                )

            saved_cov = cov_repo.save_pairs(enriched_pairs)
            logger.info(f"Saved {saved_cov} covariance pairs")

        result = {
            "market_regime": market_regime,
            "cohesion": cohesion,
            "sector_regimes": sector_regimes,
            "sector_momentum": sector_momentum,
            "significant_pairs": significant_pairs,
            "total_pairs": total_pairs,
            "intraday": intraday,
        }
        logger.info(f"Analysis complete: {result}")
        return result


_REGIME_MAP = {
    "accumulation": 1.0,
    "breakout": 2.0,
    "neutral": 0.0,
    "distribution": -1.0,
    "breakdown": -2.0,
}


def _regime_to_float(regime: str) -> float:
    return _REGIME_MAP.get(regime, 0.0)
