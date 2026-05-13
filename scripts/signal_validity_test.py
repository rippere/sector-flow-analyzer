"""
Signal Validity Test — standalone script.

Tests whether a naive momentum-based regime rule (rolling N-day return)
produces statistically different forward returns across market states.

Usage:
    python scripts/signal_validity_test.py

Output:
    - Formatted results table per window size
    - Final verdict: SIGNAL VALID / SIGNAL WEAK
    - Results saved to ~/.sector_flow/signal_validity_results.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]
WINDOWS = [10, 20, 30, 45]
HORIZONS = [5, 20]

REGIME_THRESHOLD_PCT = 0.5  # 0.5% threshold for risk_on / risk_off


def fetch_data(years: int = 5) -> pd.DataFrame:
    """Fetch daily adjusted close prices for all sector ETFs."""
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)
    print(f"Fetching {years}y of daily OHLCV for {len(TICKERS)} sector ETFs...")
    raw = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw.columns else raw["close"]
    close = close[TICKERS].dropna(how="all")
    print(f"  Loaded {len(close)} trading days ({close.index[0].date()} to {close.index[-1].date()})")
    return close


def compute_regime(close: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute market regime for each date using rolling N-day returns.

    - rolling return = pct_change(window) for each sector
    - regime = median cross-sector rolling return:
        > 0.5%  → risk_on
        < -0.5% → risk_off
        else    → neutral
    """
    returns = close.pct_change(window) * 100  # in percent
    median_return = returns.median(axis=1)

    regime = pd.Series("neutral", index=close.index)
    regime[median_return > REGIME_THRESHOLD_PCT] = "risk_on"
    regime[median_return < -REGIME_THRESHOLD_PCT] = "risk_off"
    return regime


def compute_equal_weight_returns(close: pd.DataFrame, horizon: int) -> pd.Series:
    """Equal-weight forward return over `horizon` days."""
    daily_returns = close.pct_change()
    ew_returns = daily_returns.mean(axis=1)  # equal-weight daily return
    forward = ew_returns.rolling(horizon).sum().shift(-horizon)  # forward sum
    return forward


def run_window_analysis(close: pd.DataFrame, window: int) -> dict:
    """Run the full analysis for a single window size."""
    regime = compute_regime(close, window)
    results_by_horizon = {}

    for horizon in HORIZONS:
        fwd = compute_equal_weight_returns(close, horizon)

        combined = pd.DataFrame({"regime": regime, "fwd": fwd}).dropna()

        risk_on = combined[combined["regime"] == "risk_on"]["fwd"].values
        risk_off = combined[combined["regime"] == "risk_off"]["fwd"].values
        neutral = combined[combined["regime"] == "neutral"]["fwd"].values

        if len(risk_on) < 5 or len(risk_off) < 5:
            t_stat, p_value = float("nan"), float("nan")
        else:
            t_stat, p_value = stats.ttest_ind(risk_on, risk_off, equal_var=False)

        results_by_horizon[horizon] = {
            "mean_risk_on": float(np.mean(risk_on)) if len(risk_on) > 0 else float("nan"),
            "mean_risk_off": float(np.mean(risk_off)) if len(risk_off) > 0 else float("nan"),
            "mean_neutral": float(np.mean(neutral)) if len(neutral) > 0 else float("nan"),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n_risk_on": int(len(risk_on)),
            "n_risk_off": int(len(risk_off)),
            "n_neutral": int(len(neutral)),
        }

    return results_by_horizon


def print_results_table(window: int, results: dict) -> None:
    """Print formatted results table for a single window."""
    print(f"\n{'='*70}")
    print(f"  Window: {window}-day momentum | Regime threshold: ±{REGIME_THRESHOLD_PCT}%")
    print(f"{'='*70}")
    header = f"{'Horizon':>8} | {'risk_on %':>10} | {'risk_off %':>10} | {'neutral %':>10} | {'t-stat':>8} | {'p-value':>8} | {'n_on':>6} | {'n_off':>6}"
    print(header)
    print("-" * len(header))
    for horizon, r in results.items():
        sig = " *" if (not np.isnan(r["p_value"])) and r["p_value"] < 0.05 else "  "
        print(
            f"{horizon:>7}d |"
            f" {r['mean_risk_on']:>9.3f}% |"
            f" {r['mean_risk_off']:>9.3f}% |"
            f" {r['mean_neutral']:>9.3f}% |"
            f" {r['t_stat']:>8.3f} |"
            f" {r['p_value']:>8.4f} |"
            f" {r['n_risk_on']:>6} |"
            f" {r['n_risk_off']:>6}{sig}"
        )
    print("  (* = p < 0.05)")


def main() -> None:
    close = fetch_data(years=5)

    all_results = {}
    best_p = 1.0
    best_window = None
    best_horizon = None

    for window in WINDOWS:
        results = run_window_analysis(close, window)
        all_results[window] = results
        print_results_table(window, results)

        for horizon, r in results.items():
            p = r["p_value"]
            if not np.isnan(p) and p < best_p:
                best_p = p
                best_window = window
                best_horizon = horizon

    # --- Verdict ---
    print(f"\n{'='*70}")
    if best_p < 0.05:
        verdict = "SIGNAL VALID"
        print(f"  VERDICT: {verdict} (p < 0.05 for at least one window)")
        print(f"  Best result: window={best_window}d, horizon={best_horizon}d, p={best_p:.4f}")
    else:
        verdict = "SIGNAL WEAK"
        print(f"  VERDICT: {verdict} (no window achieves p < 0.05)")
        print(f"  Best p-value: {best_p:.4f} (window={best_window}d, horizon={best_horizon}d)")
    print(f"{'='*70}")

    # --- Save results ---
    output_path = Path.home() / ".sector_flow" / "signal_validity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_at": datetime.now().isoformat(),
        "verdict": verdict,
        "best_p_value": float(best_p),
        "best_window": best_window,
        "best_horizon": best_horizon,
        "windows_tested": WINDOWS,
        "results": {
            str(w): {str(h): r for h, r in hr.items()}
            for w, hr in all_results.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
