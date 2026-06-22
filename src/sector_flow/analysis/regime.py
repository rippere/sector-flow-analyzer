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

_CRISIS_THRESHOLD = 0.80          # avg pairwise correlation (static fallback)
_ROTATION_STD_THRESHOLD = 0.25    # std of all pair correlations (static fallback)
_MOMENTUM_WINDOW = 20             # days for momentum / sector-regime signals

# Bounds keep adaptive thresholds sane even on pathological history.
_CRISIS_BOUNDS = (0.55, 0.95)
_ROTATION_BOUNDS = (0.10, 0.50)
# Flow-tilt z-score magnitude required to call a flow-driven risk_on/off regime.
_FLOW_Z_THRESHOLD = 1.0


def compute_adaptive_thresholds(
    cohesion_history=None,
    corr_std_history=None,
    *,
    crisis_pct: float = 0.90,
    rotation_pct: float = 0.75,
    min_obs: int = 30,
) -> tuple[float, float]:
    """Derive crisis / rotation thresholds from recent market history.

    Rather than fixed cutoffs (0.80 / 0.25) calibrated to one regime, set the
    crisis threshold at the ``crisis_pct`` quantile of trailing cohesion and the
    rotation threshold at the ``rotation_pct`` quantile of trailing correlation
    dispersion — so "unusually correlated/dispersed" is judged against what this
    market has actually been doing. Falls back to the static constants when there
    is too little history (< ``min_obs``).

    Parameters
    ----------
    cohesion_history, corr_std_history : pd.Series | None
        Trailing per-date mean(|corr|) and std(corr).

    Returns
    -------
    (crisis_threshold, rotation_threshold)
    """
    crisis = _CRISIS_THRESHOLD
    rotation = _ROTATION_STD_THRESHOLD

    if cohesion_history is not None:
        clean = cohesion_history.dropna()
        if len(clean) >= min_obs:
            crisis = float(np.clip(np.quantile(clean, crisis_pct), *_CRISIS_BOUNDS))
    if corr_std_history is not None:
        clean = corr_std_history.dropna()
        if len(clean) >= min_obs:
            rotation = float(np.clip(np.quantile(clean, rotation_pct), *_ROTATION_BOUNDS))

    return crisis, rotation


def risk_tilt(flow_df) -> float:
    """Net risk-on minus risk-off inflow share, in [-1, 1].

    ``(sum risk_on inflows - sum risk_off inflows) / sum |all inflows|`` — a
    scale-free measure of how strongly capital is tilting toward cyclical
    (risk-on) vs defensive (risk-off) sectors. 0.0 when there is no flow data.
    """
    if flow_df is None or flow_df.empty or "net_inflow_usd" not in flow_df.columns:
        return 0.0
    f = flow_df.dropna(subset=["net_inflow_usd"])
    if f.empty:
        return 0.0
    denom = float(f["net_inflow_usd"].abs().sum())
    if denom == 0.0:
        return 0.0
    on = float(f[f["ticker"].isin(_RISK_ON_TICKERS)]["net_inflow_usd"].sum())
    off = float(f[f["ticker"].isin(_RISK_OFF_TICKERS)]["net_inflow_usd"].sum())
    return float(np.clip((on - off) / denom, -1.0, 1.0))


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


_RANK_WINDOW = 252  # 1 trading year for percentile rank lookback


def compute_momentum(
    price_series: pd.Series,
    window: int = _MOMENTUM_WINDOW,
    rank_window: int = _RANK_WINDOW,
) -> float:
    """
    Time-series percentile rank of the current N-day return within its own
    historical distribution, rescaled from [0, 1] to [-1, 1].

    A score of +0.8 means the current 20-day return is in the 90th percentile
    of this sector's own historical distribution — a genuine relative signal
    rather than an absolute level comparison. Unlike z-score clamping,
    multiple sectors cannot all be pinned to ±1.0 unless they are all
    simultaneously at their all-time extreme 20-day return, which is
    itself a meaningful (crisis) signal.

    Parameters
    ----------
    price_series : pd.Series
        Time series of adjusted_close values (chronological order).
    window : int
        N-day return window for the momentum signal (default: 20).
    rank_window : int
        Number of historical days to rank against (default: 252 = 1 trading year).

    Returns
    -------
    float
        Momentum in [-1, 1], or 0.0 if insufficient data.
        0.0 = current return is at the median of its own history.
    """
    if len(price_series) < 2:
        return 0.0

    clean = price_series.dropna()
    if len(clean) < 2:
        return 0.0

    # Compute rolling N-day returns
    returns = clean.pct_change(window)
    valid_returns = returns.dropna()

    if len(valid_returns) < 2:
        return 0.0

    current = valid_returns.iloc[-1]
    if pd.isna(current):
        return 0.0

    # Rank current return within the last rank_window observations
    historical = valid_returns.iloc[-rank_window:]
    if len(historical) < 10:
        return 0.0

    rank = (historical < current).sum() / len(historical)  # 0.0 to 1.0
    return round(float((rank - 0.5) * 2), 4)  # rescale to [-1, 1], 0 = median


def classify_market_regime(
    corr_df: pd.DataFrame,
    flow_df: pd.DataFrame | None = None,
    crisis_threshold: float | None = None,
    rotation_threshold: float | None = None,
    flow_baseline: tuple[float, float] | None = None,
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
    crisis_threshold, rotation_threshold : float | None
        Override the static crisis/rotation cutoffs (e.g. with values from
        :func:`compute_adaptive_thresholds`). ``None`` uses the module defaults.
    flow_baseline : (mean, std) | None
        Trailing mean/std of :func:`risk_tilt`. When provided, the flow-driven
        regime is decided by whether today's tilt is a >|1σ| outlier vs its own
        history — a statistical test rather than the old arbitrary "top-3 inflows
        must include all of {XLK,XLF,XLY}" rule. Falls back to that subset rule
        when no baseline is supplied (preserves prior behaviour).

    Returns
    -------
    str
        One of: 'crisis', 'risk_on', 'risk_off', 'rotation', 'neutral'.
    """
    if corr_df.empty:
        return "neutral"

    crisis_thr = _CRISIS_THRESHOLD if crisis_threshold is None else crisis_threshold
    rotation_thr = _ROTATION_STD_THRESHOLD if rotation_threshold is None else rotation_threshold

    correlations = corr_df["correlation"].values if "correlation" in corr_df.columns else np.array([])

    # --- Crisis: all sectors moving together ---
    if len(correlations) > 0:
        avg_corr = float(np.abs(correlations).mean())
        if avg_corr > crisis_thr:
            return "crisis"

    # --- Flow-based regime detection ---
    if flow_df is not None and not flow_df.empty and "net_inflow_usd" in flow_df.columns:
        ranked = (
            flow_df.dropna(subset=["net_inflow_usd"])
            .sort_values("net_inflow_usd", ascending=False)
        )
        if len(ranked) >= 3:
            if flow_baseline is not None:
                mu, sigma = flow_baseline
                tilt = risk_tilt(flow_df)
                z = (tilt - mu) / sigma if sigma else 0.0
                if z >= _FLOW_Z_THRESHOLD:
                    return "risk_on"
                if z <= -_FLOW_Z_THRESHOLD:
                    return "risk_off"
            else:
                top3 = set(ranked["ticker"].head(3).tolist())
                if _RISK_ON_TICKERS.issubset(top3):
                    return "risk_on"
                if _RISK_OFF_TICKERS.issubset(top3):
                    return "risk_off"

    # --- Rotation: high dispersion in correlations ---
    if len(correlations) > 1:
        corr_std = float(np.std(correlations))
        if corr_std > rotation_thr:
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
