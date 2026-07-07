"""Tests for adaptive thresholds and flow-tilt regime logic in analysis.regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sector_flow.analysis.regime import (
    _CRISIS_THRESHOLD,
    _ROTATION_STD_THRESHOLD,
    classify_market_regime,
    compute_adaptive_thresholds,
    risk_tilt,
)


def _corr_df(values):
    """Build a single-date significant-correlation frame from a list of corrs."""
    n = len(values)
    return pd.DataFrame({
        "ticker_a": [f"A{i}" for i in range(n)],
        "ticker_b": [f"B{i}" for i in range(n)],
        "correlation": values,
        "date": ["2026-06-18"] * n,
    })


# --- compute_adaptive_thresholds ------------------------------------------

def test_adaptive_falls_back_to_static_when_history_short():
    crisis, rotation = compute_adaptive_thresholds(
        pd.Series([0.4, 0.5]), pd.Series([0.2, 0.3]), min_obs=30
    )
    assert crisis == _CRISIS_THRESHOLD
    assert rotation == _ROTATION_STD_THRESHOLD


def test_adaptive_uses_quantiles_with_enough_history():
    coh = pd.Series(np.linspace(0.30, 0.90, 100))   # 90th pct ≈ 0.84
    std = pd.Series(np.linspace(0.10, 0.40, 100))   # 75th pct ≈ 0.325
    crisis, rotation = compute_adaptive_thresholds(coh, std)
    assert 0.80 <= crisis <= 0.88
    assert 0.30 <= rotation <= 0.35


def test_adaptive_clips_to_bounds():
    # Degenerate history of all-1.0 cohesion would push crisis above the cap.
    crisis, _ = compute_adaptive_thresholds(
        pd.Series([1.0] * 100), pd.Series([0.0] * 100)
    )
    assert crisis <= 0.95


# --- risk_tilt ------------------------------------------------------------

def test_risk_tilt_positive_when_riskon_dominates():
    flow = pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLY", "XLU", "XLP", "XLV"],
        "net_inflow_usd": [500, 400, 300, -100, -50, -50],
    })
    assert risk_tilt(flow) > 0.5


def test_risk_tilt_negative_when_riskoff_dominates():
    flow = pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLY", "XLU", "XLP", "XLV"],
        "net_inflow_usd": [-100, -50, -50, 500, 400, 300],
    })
    assert risk_tilt(flow) < -0.5


def test_risk_tilt_zero_without_flow():
    assert risk_tilt(None) == 0.0
    assert risk_tilt(pd.DataFrame()) == 0.0


# --- classify_market_regime with overrides --------------------------------

def test_crisis_threshold_override_changes_call():
    corr = _corr_df([0.7, 0.72, 0.68])  # mean ≈ 0.70
    # Default crisis threshold 0.80 → not crisis
    assert classify_market_regime(corr) != "crisis"
    # Lowered adaptive threshold 0.65 → crisis
    assert classify_market_regime(corr, crisis_threshold=0.65) == "crisis"


def test_flow_baseline_zscore_calls_risk_on():
    corr = _corr_df([0.1, 0.15, 0.05])  # low cohesion, low dispersion → neutral-ish
    flow = pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLY", "XLU", "XLP", "XLV"],
        "net_inflow_usd": [500, 400, 300, -100, -50, -50],
    })
    # tilt ≈ +0.78; baseline mean 0, std 0.2 → z ≈ +3.9 → risk_on
    assert classify_market_regime(corr, flow_df=flow, flow_baseline=(0.0, 0.2)) == "risk_on"


def test_flow_baseline_zscore_within_band_is_not_flow_regime():
    corr = _corr_df([0.1, 0.15, 0.05])
    flow = pd.DataFrame({
        "ticker": ["XLK", "XLF", "XLY", "XLU", "XLP", "XLV"],
        "net_inflow_usd": [120, 100, 80, 100, 90, 110],
    })
    # small tilt vs wide baseline → |z| < 1 → not risk_on/off
    assert classify_market_regime(corr, flow_df=flow, flow_baseline=(0.0, 0.9)) not in ("risk_on", "risk_off")
