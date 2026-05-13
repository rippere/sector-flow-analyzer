"""
Statistical significance filtering for sector correlation pairs.

Prevents noise in sector flow analysis by retaining only pairs whose
Pearson correlation is both large enough (min_abs_correlation) and
statistically significant (p_value_threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_p_values(
    price_df: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    For each ticker pair, compute the Pearson correlation and p-value over
    the most recent `window` trading rows available.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide-format price matrix (index=date, columns=tickers).
    window : int
        Number of most-recent rows to use for the correlation test.

    Returns
    -------
    pd.DataFrame
        Columns: ticker_a, ticker_b, correlation, p_value.
        ticker_a < ticker_b always.
    """
    tickers = sorted(price_df.columns.tolist())
    # Use the tail of the price matrix (drop rows where ALL values are NaN)
    recent = price_df.dropna(how="all").tail(window)

    records = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            ser_a = recent[a].dropna()
            ser_b = recent[b].dropna()
            # Align on shared index
            common = ser_a.index.intersection(ser_b.index)
            if len(common) < 3:
                continue
            xa = ser_a.loc[common].values
            xb = ser_b.loc[common].values
            corr, pval = stats.pearsonr(xa, xb)
            records.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "correlation": float(corr),
                    "p_value": float(pval),
                }
            )

    if not records:
        return pd.DataFrame(columns=["ticker_a", "ticker_b", "correlation", "p_value"])

    return pd.DataFrame(records)


def filter_significant_correlations(
    corr_df: pd.DataFrame,
    min_abs_correlation: float = 0.3,
    p_value_threshold: float = 0.05,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Filter a rolling-correlation DataFrame to statistically significant pairs.

    If `corr_df` does not already have a 'p_value' column, p-values are added
    using scipy.stats.pearsonr on a per-row basis (single-pair approximation).
    This is suitable for already-windowed data where each row represents a
    completed rolling window.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Output from compute_rolling_correlation or a subset thereof.
        Must have columns: ticker_a, ticker_b, correlation, window_days.
    min_abs_correlation : float
        Absolute correlation threshold below which pairs are dropped.
    p_value_threshold : float
        Maximum p-value to retain (two-tailed Pearson test).
    min_periods : int
        Minimum number of periods used to compute the correlation; used to
        derive a t-statistic when p_value is not already present.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with added 'p_value' and 'significant' columns.
    """
    if corr_df.empty:
        out = corr_df.copy()
        out["p_value"] = pd.Series(dtype=float)
        out["significant"] = pd.Series(dtype=bool)
        return out

    df = corr_df.copy()

    if "p_value" not in df.columns:
        # Derive p-value from the t-distribution given n and r
        # n is taken from window_days (available in the rolling-correlation output)
        n = df["window_days"].astype(int) if "window_days" in df.columns else min_periods

        r = df["correlation"].clip(-0.9999999, 0.9999999)
        t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
        # Two-tailed p-value
        df["p_value"] = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    df["significant"] = (
        (df["correlation"].abs() >= min_abs_correlation)
        & (df["p_value"] <= p_value_threshold)
    )

    return df[df["significant"]].reset_index(drop=True)
