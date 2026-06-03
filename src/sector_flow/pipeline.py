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

_MAX_GAP_DAYS = 30
_WARN_GAP_TRADING_DAYS = 5


def _compute_gap(price_repo: PriceRepository, etf_repo: ETFRepository) -> int:
    """
    Determine the calendar-day gap from the most recent PriceData row to today.

    Queries each ETF's latest date and returns the maximum gap across all ETFs,
    capped at _MAX_GAP_DAYS. Returns 1 if no data exists yet (fresh DB).
    """
    today = date.today()
    max_gap = 0
    for ticker, _, _ in SECTOR_ETFS:
        etf = etf_repo.get_by_ticker(ticker)
        if etf is None:
            continue
        latest = price_repo.get_latest_date(etf.id)
        if latest is None:
            # No data at all — treat as needing a full fetch (capped)
            return _MAX_GAP_DAYS
        if isinstance(latest, datetime):
            latest_date = latest.date()
        else:
            latest_date = latest
        gap = (today - latest_date).days
        if gap > max_gap:
            max_gap = gap

    if max_gap == 0:
        return 1  # already up to date
    return min(max_gap, _MAX_GAP_DAYS)


def run_daily(lookback_days: int = 1, database_url: Optional[str] = None) -> dict:
    """
    Fetch recent OHLCV + today's SSGA snapshot, merge, persist.

    The lookback window is auto-detected from the latest PriceData date in the
    DB: gap = today - most_recent_row_date, capped at 30 calendar days.
    A WARNING is logged if the gap exceeds 5 trading days.

    The `lookback_days` parameter is kept for backward compatibility but is
    ignored when the DB already has data — the gap takes precedence.

    Returns a summary dict with row counts and any errors.
    """
    init_db(database_url)
    yf_collector = YFinanceCollector()
    ssga_collector = SSGACollector()

    today = date.today()
    summary = {"yfinance_rows": 0, "ssga_rows": 0, "flow_rows_updated": 0, "errors": []}

    with get_session(database_url) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)

        etf_repo.seed_etfs()

        # Auto-detect gap from DB instead of using hardcoded lookback_days=1
        gap = _compute_gap(price_repo, etf_repo)
        if gap > _WARN_GAP_TRADING_DAYS:
            logger.warning(
                f"[pipeline] Data gap is {gap} calendar days — "
                f"exceeds {_WARN_GAP_TRADING_DAYS} trading-day threshold. "
                f"Consider running backfill."
            )
        logger.info(f"[pipeline] Detected gap: {gap} calendar days → fetching from today-{gap}")

        start = today - timedelta(days=gap + 1)
        end = today

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
            # SSGA prints registered-trademark glyphs on some tickers ("GLD®") — strip for matching
            snapshot["Ticker"] = (
                snapshot["Ticker"].astype(str)
                .str.replace("®", "", regex=False)
                .str.replace("™", "", regex=False)
                .str.strip()
            )
            for ticker, _, _ in SECTOR_ETFS:
                etf = etf_repo.get_by_ticker(ticker)
                if etf is None:
                    continue
                row = snapshot[snapshot["Ticker"] == ticker]
                if row.empty:
                    logger.warning(f"[ssga] {ticker} not found in snapshot")
                    continue

                r = row.iloc[0]
                # Stamp the snapshot with the FILE's own as-of date (it lags ~1 day),
                # not the run date — re-downloading the same file then upserts the same
                # row instead of minting a duplicate "today" row whose zero deltas
                # freeze net_inflow at 0 (the 2026-06-03 incident).
                as_of_dt = _parse_snapshot_as_of(r) or datetime.combine(today, datetime.min.time())
                updated = price_repo.update_flow_fields(
                    etf_id=etf.id,
                    as_of=as_of_dt,
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
    from sector_flow.collectors.ssga_collector import _to_float as _ssga_to_float
    return _ssga_to_float(val)


def _parse_snapshot_as_of(row) -> Optional[datetime]:
    """Parse the SSGA 'As of**' cell ('Jun 02 2026' or a Timestamp) to a midnight datetime."""
    raw = None
    for key in ("As of**", "As of", "As Of**", "As Of"):
        raw = row.get(key)
        if raw is not None:
            break
    if raw is None:
        return None
    try:
        import pandas as pd

        ts = pd.to_datetime(str(raw), errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        return datetime.combine(ts.date(), datetime.min.time())
    except Exception:
        return None


def backfill(days: Optional[int] = None, database_url: Optional[str] = None) -> dict:
    """
    Pull OHLCV history for all sectors via yfinance.

    If `days` is None and the DB already has data, fetches only the calendar
    gap between the most recent row and today (capped at 30 days) rather than
    defaulting to a full 90-day pull. If the DB is empty, defaults to 90 days.

    SSGA flow data cannot be backfilled — starts from today forward.
    """
    init_db(database_url)
    yf_collector = YFinanceCollector()
    today = date.today()
    summary = {"rows_saved": 0, "errors": []}

    with get_session(database_url) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        etf_repo.seed_etfs()

        # If days not specified, use gap detection; fall back to 90 for fresh DBs
        if days is None:
            detected_gap = _compute_gap(price_repo, etf_repo)
            # _compute_gap returns _MAX_GAP_DAYS (30) when DB is empty
            # For backfill with no args, use a 90-day default on empty DB
            if detected_gap >= _MAX_GAP_DAYS:
                actual_days = 90
            else:
                actual_days = detected_gap
            logger.info(f"[backfill] No days specified; auto-detected gap = {detected_gap} → fetching {actual_days} days")
        else:
            actual_days = days

        start = today - timedelta(days=actual_days)

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
