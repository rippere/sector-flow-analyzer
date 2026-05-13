"""
Data ingestion pipeline — orchestrates collectors and writes to the database.

Daily run order:
  1. YFinanceCollector  → OHLCV for all 11 sectors
  2. SSGACollector      → shares_outstanding + AUM for all 11 sectors
  3. compute_flows()    → net_inflow_usd from consecutive-day deltas
"""

from datetime import date, timedelta, datetime
from typing import Optional

from loguru import logger

from sector_flow.collectors.yfinance_collector import YFinanceCollector
from sector_flow.collectors.ssga_collector import SSGACollector
from sector_flow.database.models import SECTOR_ETFS
from sector_flow.database.repository import ETFRepository, PriceRepository
from sector_flow.database.session import get_session, init_db


TICKERS = [t for t, _, _ in SECTOR_ETFS]


def run_daily(lookback_days: int = 1, database_url: Optional[str] = None) -> dict:
    """
    Fetch yesterday's (or more) OHLCV + today's SSGA snapshot, merge, persist.
    Returns a summary dict with row counts and any errors.
    """
    init_db(database_url)
    yf_collector = YFinanceCollector()
    ssga_collector = SSGACollector()

    today = date.today()
    start = today - timedelta(days=lookback_days + 1)
    end = today

    summary = {"yfinance_rows": 0, "ssga_rows": 0, "flow_rows_updated": 0, "errors": []}

    with get_session(database_url) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)

        etf_repo.seed_etfs()

        # --- Phase 1: OHLCV via yfinance ---
        for ticker, _, _ in SECTOR_ETFS:
            etf = etf_repo.get_by_ticker(ticker)
            if etf is None:
                continue
            try:
                records = yf_collector.fetch(ticker, start, end)
                saved = price_repo.save_price_data(etf.id, records)
                summary["yfinance_rows"] += saved
            except Exception as exc:
                logger.error(f"yfinance error for {ticker}: {exc}")
                summary["errors"].append(f"yfinance:{ticker}:{exc}")

        # --- Phase 2: SSGA snapshot (today's shares_outstanding + AUM) ---
        try:
            snapshot = ssga_collector.fetch_snapshot()
            for ticker, _, _ in SECTOR_ETFS:
                etf = etf_repo.get_by_ticker(ticker)
                if etf is None:
                    continue
                row = snapshot[snapshot["Ticker"] == ticker]
                if row.empty:
                    logger.warning(f"[ssga] {ticker} not found in snapshot")
                    continue

                r = row.iloc[0]
                today_dt = datetime.combine(today, datetime.min.time())
                updated = price_repo.update_flow_fields(
                    etf_id=etf.id,
                    as_of=today_dt,
                    shares_outstanding=_to_float(r.get("Shares Outstanding")),
                    aum_usd=_to_float(r.get("Total Net Assets")),
                )
                if updated:
                    summary["ssga_rows"] += 1

        except Exception as exc:
            logger.error(f"SSGA snapshot error: {exc}")
            summary["errors"].append(f"ssga:{exc}")

        # --- Phase 3: Compute net_inflow_usd from consecutive day deltas ---
        for ticker, _, _ in SECTOR_ETFS:
            etf = etf_repo.get_by_ticker(ticker)
            if etf is None:
                continue
            updated = price_repo.compute_and_store_flows(etf.id)
            summary["flow_rows_updated"] += updated

    logger.info(f"Pipeline complete: {summary}")
    return summary


def _to_float(val) -> float | None:
    import math
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def backfill(days: int = 90, database_url: Optional[str] = None) -> dict:
    """
    Pull up to `days` of OHLCV history for all sectors via yfinance.
    SSGA flow data cannot be backfilled — starts from today forward.
    """
    init_db(database_url)
    yf_collector = YFinanceCollector()
    today = date.today()
    start = today - timedelta(days=days)
    summary = {"rows_saved": 0, "errors": []}

    with get_session(database_url) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        etf_repo.seed_etfs()

        for ticker, _, _ in SECTOR_ETFS:
            etf = etf_repo.get_by_ticker(ticker)
            if etf is None:
                continue
            try:
                records = yf_collector.fetch(ticker, start, today)
                saved = price_repo.save_price_data(etf.id, records)
                summary["rows_saved"] += saved
                logger.info(f"[backfill] {ticker}: {saved} rows")
            except Exception as exc:
                logger.error(f"[backfill] {ticker}: {exc}")
                summary["errors"].append(f"{ticker}:{exc}")

    return summary
