"""
Regime classification for market-level and per-sector analysis.

Market regimes: crisis | risk_on | risk_off | rotation | neutral
Sector regimes: accumulation | distribution | breakout | breakdown | neutral

These regime labels drive particle behavior in the nw_wrld visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tickers for each regime bucket
_RISK_ON_TICKERS = {"XLK", "XLY", "XLF"}
_RISK_OFF_TICKERS = {"XLU", "XLP", "XLV"}

_CRISIS_THRESHOLD = 0.80          # avg pairwise correlation
_ROTATION_STD_THRESHOLD = 0.25    # std of all pair correlations
_MOMENTUM_WINDOW = 20             # days for momentum / sector-regime signals


def compute_cohesion(corr_df: pd.DataFrame) -> float:
    """
    Average absolute pairwise correlation across all sector pairs.

    Range [0, 1]. High = crisis/macro-driven. Low = idiosyncratic rotation.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Must have a 'correlation' column.

    Returns
    -------
    float
        Cohesion score in [0, 1], or 0.0 if no data.
    """
    if corr_df.empty or "correlation" not in corr_df.columns:
        return 0.0
    val = float(corr_df["correlation"].abs().mean())
    return float(np.clip(val, 0.0, 1.0))


def compute_momentum(
    price_series: pd.Series,
    window: int = _MOMENTUM_WINDOW,
) -> float:
    """
    Signed normalized momentum: (current - rolling_mean) / rolling_std.
    Clamped to [-1, 1].

    Parameters
    ----------
    price_series : pd.Series
        Time series of adjusted_close values.
    window : int
        Rolling window for mean and std.

    Returns
    -------
    float
        Momentum in [-1, 1], or 0.0 if insufficient data.
    """
    if len(price_series) < 2:
        return 0.0

    clean = price_series.dropna()
    if len(clean) < 2:
        return 0.0

    tail = clean.tail(window)
    rolling_mean = tail.mean()
    rolling_std = tail.std()

    if rolling_std == 0 or np.isnan(rolling_std):
        return 0.0

    current = float(clean.iloc[-1])
    mom = (current - rolling_mean) / rolling_std
    return float(np.clip(mom, -1.0, 1.0))


def classify_market_regime(
    corr_df: pd.DataFrame,
    flow_df: pd.DataFrame | None = None,
) -> str:
    """
    Classify the overall market regime based on inter-sector correlations
    and optional flow data.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Significant correlations for the latest date.
        Must have 'ticker_a', 'ticker_b', 'correlation' columns.
    flow_df : pd.DataFrame | None
        Optional. Must have 'ticker' and 'net_inflow_usd' columns.

    Returns
    -------
    str
        One of: 'crisis', 'risk_on', 'risk_off', 'rotation', 'neutral'.
    """
    if corr_df.empty:
        return "neutral"

    correlations = corr_df["correlation"].values if "correlation" in corr_df.columns else np.array([])

    # --- Crisis: all sectors moving together ---
    if len(correlations) > 0:
        avg_corr = float(np.abs(correlations).mean())
        if avg_corr > _CRISIS_THRESHOLD:
            return "crisis"

    # --- Flow-based regime detection ---
    if flow_df is not None and not flow_df.empty and "net_inflow_usd" in flow_df.columns:
        ranked = (
            flow_df.dropna(subset=["net_inflow_usd"])
            .sort_values("net_inflow_usd", ascending=False)
        )
        if len(ranked) >= 3:
            top3 = set(ranked["ticker"].head(3).tolist())
            if _RISK_ON_TICKERS.issubset(top3):
                return "risk_on"
            if _RISK_OFF_TICKERS.issubset(top3):
                return "risk_off"

    # --- Rotation: high dispersion in correlations ---
    if len(correlations) > 1:
        corr_std = float(np.std(correlations))
        if corr_std > _ROTATION_STD_THRESHOLD:
            return "rotation"

    return "neutral"


def classify_sector_regime(
    ticker: str,
    price_series: pd.Series,
    flow_series: pd.Series | None = None,
    volume_series: pd.Series | None = None,
) -> str:
    """
    Classify the regime for a single sector ETF.

    Parameters
    ----------
    ticker : str
        ETF symbol (informational only; not used in logic).
    price_series : pd.Series
        Adjusted close prices (sorted chronologically).
    flow_series : pd.Series | None
        Net inflow in USD for the same dates.
    volume_series : pd.Series | None
        Trading volume for the same dates.

    Returns
    -------
    str
        One of: 'accumulation', 'distribution', 'breakout', 'breakdown', 'neutral'.
    """
    if len(price_series) < 2:
        return "neutral"

    clean_price = price_series.dropna()
    if len(clean_price) < 2:
        return "neutral"

    current_price = float(clean_price.iloc[-1])
    window_prices = clean_price.tail(_MOMENTUM_WINDOW)

    rolling_high = float(window_prices.max())
    rolling_low = float(window_prices.min())
    rolling_mean = float(window_prices.mean())

    # Trend: positive if current > rolling mean
    trend_up = current_price > rolling_mean
    trend_down = current_price < rolling_mean

    # Determine flow signal
    has_positive_flow = None
    if flow_series is not None and len(flow_series.dropna()) > 0:
        latest_flow = flow_series.dropna().iloc[-1]
        has_positive_flow = bool(latest_flow > 0)
    elif volume_series is not None and len(volume_series.dropna()) >= 5:
        # Proxy: compare recent avg volume to prior window
        clean_vol = volume_series.dropna()
        recent_vol = float(clean_vol.tail(5).mean())
        prior_vol = float(clean_vol.iloc[:-5].tail(_MOMENTUM_WINDOW).mean()) if len(clean_vol) > 5 else recent_vol
        has_positive_flow = bool(recent_vol > prior_vol) if prior_vol > 0 else None

    # --- Breakout: crosses 20-day high ---
    if current_price >= rolling_high and trend_up:
        if has_positive_flow is True or has_positive_flow is None:
            return "breakout"

    # --- Breakdown: crosses 20-day low ---
    if current_price <= rolling_low and trend_down:
        if has_positive_flow is False or has_positive_flow is None:
            return "breakdown"

    # --- Accumulation: uptrend + positive flow ---
    if trend_up and has_positive_flow is True:
        return "accumulation"

    # --- Distribution: downtrend + negative flow ---
    if trend_down and has_positive_flow is False:
        return "distribution"

    return "neutral"
