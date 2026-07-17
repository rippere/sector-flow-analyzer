"""
Statistical significance filtering for sector correlation pairs.

Prevents noise in sector flow analysis by retaining only pairs whose
Pearson correlation is both large enough (min_abs_correlation) and
statistically significant (p_value_threshold).

Also provides a permutation-based validation harness: `compute_p_values`'
parametric p-values assume normal, independent returns, and testing many
pairs at once means the single best-looking pair is expected to look
significant by chance alone (a "best-of-N" selection problem). The
`permutation_correlation_test` and `best_of_n_significance` functions below
build empirical null distributions instead of trusting the parametric
assumption or the nominal per-pair p-value.
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


def permutation_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Nonparametric permutation test for the Pearson correlation between two series.

    Repeatedly shuffles `y` to break any association with `x` while preserving
    each series' own marginal distribution, recomputing the correlation each
    time to build an empirical null distribution. Avoids relying on
    `scipy.stats.pearsonr`'s normality assumption, which daily return series
    routinely violate.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length paired samples.
    n_permutations : int
        Number of label shuffles used to build the null distribution.
    rng : np.random.Generator | None
        Source of randomness; a fresh default_rng() is used if not supplied.

    Returns
    -------
    tuple[float, float]
        (observed_correlation, empirical_p_value). The p-value is the
        two-tailed fraction of permuted |correlation| >= observed
        |correlation|, with add-one smoothing so it is never exactly zero.
        (nan, nan) if there are fewer than 3 paired observations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return float("nan"), float("nan")

    rng = rng if rng is not None else np.random.default_rng()

    observed = float(np.corrcoef(x, y)[0, 1])
    observed_abs = abs(observed)

    exceed_count = 0
    for _ in range(n_permutations):
        shuffled_y = rng.permutation(y)
        perm_corr = np.corrcoef(x, shuffled_y)[0, 1]
        if abs(perm_corr) >= observed_abs:
            exceed_count += 1

    p_value = (exceed_count + 1) / (n_permutations + 1)
    return observed, float(p_value)


def best_of_n_significance(
    price_df: pd.DataFrame,
    window: int = 30,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Selection-aware significance for the single most-correlated pair among N.

    `compute_p_values` tests every ticker pair; reporting the nominal p-value
    of whichever pair happens to look best overstates significance, since the
    more pairs tested the more likely one clears a fixed threshold by chance
    alone. This shuffles every ticker's own return series independently
    (destroying all cross-sectional correlation while preserving each
    series' marginal distribution), recomputes every pair's correlation
    under that null, and tracks the *max* |correlation| across all pairs per
    permutation — the empirical null distribution of "the best of N pairs".
    The corrected p-value is the fraction of permutations whose best-of-N
    correlation is at least as extreme as the one actually observed.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide-format price matrix (index=date, columns=tickers).
    window : int
        Number of most-recent rows to use, matching `compute_p_values`.
    n_permutations : int
        Number of independent full-panel shuffles used to build the null.
    rng : np.random.Generator | None
        Source of randomness; a fresh default_rng() is used if not supplied.

    Returns
    -------
    dict
        best_pair, best_correlation, nominal_p_value (from compute_p_values),
        corrected_p_value (family-wise, permutation-based), n_pairs_tested.
    """
    pvals = compute_p_values(price_df, window=window)
    if pvals.empty:
        return {
            "best_pair": None,
            "best_correlation": float("nan"),
            "nominal_p_value": float("nan"),
            "corrected_p_value": float("nan"),
            "n_pairs_tested": 0,
        }

    best_idx = pvals["correlation"].abs().idxmax()
    best_row = pvals.loc[best_idx]
    observed_max = float(pvals["correlation"].abs().max())

    rng = rng if rng is not None else np.random.default_rng()
    recent = price_df.dropna(how="all").tail(window)
    tickers = sorted(recent.columns.tolist())

    exceed_count = 0
    for _ in range(n_permutations):
        null_max = 0.0
        shuffled = {t: rng.permutation(recent[t].dropna().values) for t in tickers}
        for i, a in enumerate(tickers):
            for b in tickers[i + 1:]:
                xa, xb = shuffled[a], shuffled[b]
                n = min(len(xa), len(xb))
                if n < 3:
                    continue
                c = abs(np.corrcoef(xa[:n], xb[:n])[0, 1])
                if c > null_max:
                    null_max = c
        if null_max >= observed_max:
            exceed_count += 1

    corrected_p = (exceed_count + 1) / (n_permutations + 1)

    return {
        "best_pair": (str(best_row["ticker_a"]), str(best_row["ticker_b"])),
        "best_correlation": float(best_row["correlation"]),
        "nominal_p_value": float(best_row["p_value"]),
        "corrected_p_value": float(corrected_p),
        "n_pairs_tested": int(len(pvals)),
    }
