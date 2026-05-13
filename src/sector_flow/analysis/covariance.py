"""
Rolling covariance and correlation engine for sector ETF pairs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sector_flow.database.models import PriceData


def build_price_matrix(records: list[PriceData], tickers: list[str]) -> pd.DataFrame:
    """
    Pivot PriceData ORM records into a wide-format DataFrame.

    Parameters
    ----------
    records : list[PriceData]
        ORM rows from the price_data table (mixed tickers).
    tickers : list[str]
        Ordered list of tickers to include as columns.

    Returns
    -------
    pd.DataFrame
        index = date (DatetimeIndex), columns = ticker, values = adjusted_close.
        Forward-filled for weekends/holidays; no backfill.
    """
    rows = [
        {"date": r.date, "ticker": r.etf.ticker, "adjusted_close": r.adjusted_close}
        for r in records
        if r.etf is not None and r.etf.ticker in tickers
    ]
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="ticker", values="adjusted_close", aggfunc="last")
    pivot = pivot.reindex(columns=tickers)

    # Build complete calendar index between first and last date
    if not pivot.empty:
        full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
        pivot = pivot.reindex(full_idx)
        # Forward-fill only (no backfill)
        pivot = pivot.ffill()

    pivot.index.name = "date"
    return pivot


def compute_rolling_correlation(
    price_df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlation and covariance for all ticker pairs.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide-format price matrix (index=date, columns=tickers, values=adjusted_close).
    window : int
        Rolling window in days.

    Returns
    -------
    pd.DataFrame
        Long-form with columns: date, ticker_a, ticker_b, correlation, covariance, window_days.
        Only pairs where ticker_a < ticker_b (alphabetical). NaN rows dropped.
    """
    tickers = sorted(price_df.columns.tolist())
    records = []

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            ser_a = price_df[a]
            ser_b = price_df[b]

            rolling_corr = ser_a.rolling(window=window, min_periods=window).corr(ser_b)
            rolling_cov = ser_a.rolling(window=window, min_periods=window).cov(ser_b)

            for date, corr, cov in zip(rolling_corr.index, rolling_corr.values, rolling_cov.values):
                if np.isnan(corr) or np.isnan(cov):
                    continue
                records.append(
                    {
                        "date": date,
                        "ticker_a": a,
                        "ticker_b": b,
                        "correlation": float(corr),
                        "covariance": float(cov),
                        "window_days": window,
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=["date", "ticker_a", "ticker_b", "correlation", "covariance", "window_days"]
        )

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["date", "ticker_a", "ticker_b"]).reset_index(drop=True)
    return result


def compute_pairwise_latest(
    price_df: pd.DataFrame,
    window: int = 30,
) -> list[dict]:
    """
    Compute rolling correlations and return the most recent date's rows as dicts.

    Returns
    -------
    list[dict]
        Each dict: ticker_a, ticker_b, correlation, covariance, window_days, date.
        Ready to insert into CovarianceMatrix after etf_a_id / etf_b_id are resolved.
    """
    corr_df = compute_rolling_correlation(price_df, window=window)
    if corr_df.empty:
        return []

    latest_date = corr_df["date"].max()
    latest = corr_df[corr_df["date"] == latest_date]
    return latest.to_dict(orient="records")
