"""
Walk-Forward Regime Backtest Harness — standalone script.

Uses the 45-day momentum window (most significant in signal_validity_test.py).

Walk-forward methodology:
  - Years 1-2: in-sample (no parameter fitting needed — regime rules are fixed)
  - Years 3-5: walk-forward out-of-sample evaluation
    For each date, regime is computed using ONLY data available up to that date.
    Forward returns at 5d and 20d horizons are recorded.

Outputs:
  - Regime transition table
  - Mean forward return (5d/20d) conditional on regime
  - Hit rate: % of risk_on signals followed by positive 5d return
  - Brier score for regime forecasts
  - Confusion matrix vs SPY momentum benchmark
  - Results saved to ~/.sector_flow/backtest_results.json

Usage:
    python scripts/backtest_regime.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Configuration
TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]
BEST_WINDOW = 45          # most significant window from signal_validity_test.py
REGIME_THRESHOLD_PCT = 0.5   # 0.5% median cross-sector return threshold
HORIZONS = [5, 20]
YEARS_TOTAL = 5
YEARS_INSAMPLE = 2


def fetch_data(years: int = YEARS_TOTAL) -> pd.DataFrame:
    """Fetch daily adjusted close prices."""
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)
    print(f"Fetching {years}y of daily OHLCV ({start.date()} to {end.date()})...")
    raw = yf.download(
        TICKERS + ["SPY"],
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].dropna(how="all")
    print(f"  Loaded {len(close)} trading days")
    return close


def compute_regime_at(close_up_to: pd.DataFrame, window: int) -> str:
    """
    Compute regime using ONLY data available up to close_up_to (no lookahead).
    Returns 'risk_on', 'risk_off', or 'neutral'.
    """
    sector_close = close_up_to[TICKERS]
    if len(sector_close) < window + 1:
        return "neutral"

    returns = sector_close.pct_change(window).iloc[-1] * 100
    median_return = float(returns.median())

    if median_return > REGIME_THRESHOLD_PCT:
        return "risk_on"
    elif median_return < -REGIME_THRESHOLD_PCT:
        return "risk_off"
    else:
        return "neutral"


def compute_spy_regime(spy_returns: pd.Series, window: int, idx: int) -> str:
    """SPY momentum benchmark: risk_on if SPY N-day return > 0, else risk_off."""
    if idx < window:
        return "neutral"
    spy_window_return = float(spy_returns.iloc[idx - window:idx].sum())
    if spy_window_return > REGIME_THRESHOLD_PCT:
        return "risk_on"
    elif spy_window_return < -REGIME_THRESHOLD_PCT:
        return "risk_off"
    return "neutral"


def regime_to_int(regime: str) -> int:
    """Map regime to integer for Brier score computation."""
    return {"risk_on": 1, "neutral": 0, "risk_off": -1}[regime]


def run_walk_forward(close: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward regime labeling from year 3 to year 5.

    For each date in the out-of-sample period:
    - Compute regime using all data up to (but not including) that date
    - Record subsequent equal-weight sector forward returns

    Returns a DataFrame with columns:
        date, regime, spy_regime, fwd_5d, fwd_20d
    """
    sector_close = close[TICKERS]
    spy_close = close["SPY"] if "SPY" in close.columns else None

    # Split in-sample / out-of-sample
    split_idx = int(len(close) * YEARS_INSAMPLE / YEARS_TOTAL)
    oos_close = close.iloc[split_idx:]

    print(f"\nIn-sample:  {close.index[0].date()} to {close.index[split_idx-1].date()} ({split_idx} days)")
    print(f"Out-of-sample: {close.index[split_idx].date()} to {close.index[-1].date()} ({len(oos_close)} days)")

    # Equal-weight daily returns for forward computation
    ew_daily = sector_close.pct_change().mean(axis=1)
    spy_daily = spy_close.pct_change() if spy_close is not None else None

    records = []
    total_oos = len(oos_close)

    for i, (dt, _) in enumerate(oos_close.iterrows()):
        global_idx = split_idx + i

        # Compute regime using ONLY data up to (but not including) this date
        close_up_to = close.iloc[:global_idx]
        regime = compute_regime_at(close_up_to, BEST_WINDOW)

        # SPY benchmark regime
        spy_regime = "neutral"
        if spy_daily is not None:
            spy_regime = compute_spy_regime(spy_daily.iloc[:global_idx], BEST_WINDOW, global_idx)

        # Forward returns (may be NaN near end of series)
        fwd_5d = float(ew_daily.iloc[global_idx: global_idx + 5].sum()) if global_idx + 5 < len(ew_daily) else float("nan")
        fwd_20d = float(ew_daily.iloc[global_idx: global_idx + 20].sum()) if global_idx + 20 < len(ew_daily) else float("nan")

        records.append({
            "date": dt,
            "regime": regime,
            "spy_regime": spy_regime,
            "fwd_5d": fwd_5d,
            "fwd_20d": fwd_20d,
        })

        if i % 100 == 0:
            print(f"  Processing {i}/{total_oos}...", end="\r")

    print(f"  Processed {total_oos}/{total_oos} days")
    return pd.DataFrame(records).set_index("date")


def compute_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime-to-regime transition counts."""
    regimes = ["risk_on", "neutral", "risk_off"]
    matrix = pd.DataFrame(0, index=regimes, columns=regimes)
    prev_regime = None
    for regime in df["regime"]:
        if prev_regime is not None:
            matrix.loc[prev_regime, regime] += 1
        prev_regime = regime
    # Normalize to probabilities
    row_sums = matrix.sum(axis=1)
    prob_matrix = matrix.div(row_sums.replace(0, 1), axis=0)
    return prob_matrix


def brier_score(df: pd.DataFrame, horizon: int) -> float:
    """
    Compute Brier score treating regime as a binary risk_on (1) vs other (0) signal
    and forward return sign as the outcome.

    Lower = better. 0.25 = random.
    """
    fwd_col = f"fwd_{horizon}d"
    valid = df[[fwd_col, "regime"]].dropna()
    if len(valid) == 0:
        return float("nan")

    # Forecast: probability of positive return given regime
    # risk_on → forecast=0.67, neutral → 0.5, risk_off → 0.33
    forecast_map = {"risk_on": 0.67, "neutral": 0.5, "risk_off": 0.33}
    forecasts = valid["regime"].map(forecast_map).values
    outcomes = (valid[fwd_col] > 0).astype(float).values
    return float(np.mean((forecasts - outcomes) ** 2))


def confusion_matrix_vs_spy(df: pd.DataFrame) -> dict:
    """Compare walk-forward regime labels to SPY momentum benchmark."""
    regimes = ["risk_on", "neutral", "risk_off"]
    result = {}
    for regime in regimes:
        wf_mask = df["regime"] == regime
        spy_mask = df["spy_regime"] == regime
        tp = int((wf_mask & spy_mask).sum())
        fp = int((wf_mask & ~spy_mask).sum())
        fn = int((~wf_mask & spy_mask).sum())
        tn = int((~wf_mask & ~spy_mask).sum())
        result[regime] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return result


def print_results(df: pd.DataFrame) -> dict:
    """Print all results and return a dict for JSON serialization."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD BACKTEST — {BEST_WINDOW}-day momentum window")
    print(f"{'='*70}")

    # --- Regime counts ---
    regime_counts = df["regime"].value_counts()
    print(f"\nRegime distribution:")
    for r, n in regime_counts.items():
        pct = n / len(df) * 100
        print(f"  {r:>10}: {n:>4} days ({pct:.1f}%)")

    # --- Mean forward returns by regime ---
    print(f"\nMean forward returns by regime:")
    print(f"  {'Regime':>10} | {'5d return':>10} | {'20d return':>10} | {'n':>6}")
    print(f"  {'-'*48}")

    regime_stats = {}
    for regime in ["risk_on", "neutral", "risk_off"]:
        subset = df[df["regime"] == regime]
        mean_5d = float(subset["fwd_5d"].mean()) if len(subset) > 0 else float("nan")
        mean_20d = float(subset["fwd_20d"].mean()) if len(subset) > 0 else float("nan")
        n = len(subset)
        print(f"  {regime:>10} | {mean_5d*100:>9.3f}% | {mean_20d*100:>9.3f}% | {n:>6}")
        regime_stats[regime] = {"mean_fwd_5d": mean_5d, "mean_fwd_20d": mean_20d, "n": n}

    # --- Hit rate (risk_on → positive 5d return) ---
    risk_on = df[df["regime"] == "risk_on"]["fwd_5d"].dropna()
    hit_rate = float((risk_on > 0).mean()) if len(risk_on) > 0 else float("nan")
    print(f"\nHit rate (risk_on → positive 5d return): {hit_rate*100:.1f}% (n={len(risk_on)})")

    # --- Brier scores ---
    brier_5d = brier_score(df, 5)
    brier_20d = brier_score(df, 20)
    print(f"\nBrier scores (lower=better, 0.25=random):")
    print(f"  5d horizon: {brier_5d:.4f}")
    print(f"  20d horizon: {brier_20d:.4f}")

    # --- Transition table ---
    trans = compute_transition_table(df)
    print(f"\nRegime transition probabilities (rows=from, cols=to):")
    print(trans.round(3).to_string())

    # --- Confusion matrix vs SPY ---
    cm = confusion_matrix_vs_spy(df)
    print(f"\nConfusion matrix vs SPY momentum benchmark:")
    for regime, counts in cm.items():
        print(f"  {regime}: TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, TN={counts['tn']}")

    print(f"\n{'='*70}")

    return {
        "run_at": datetime.now().isoformat(),
        "config": {
            "window": BEST_WINDOW,
            "regime_threshold_pct": REGIME_THRESHOLD_PCT,
            "years_total": YEARS_TOTAL,
            "years_insample": YEARS_INSAMPLE,
        },
        "regime_stats": regime_stats,
        "hit_rate_risk_on_5d": hit_rate,
        "brier_scores": {"5d": brier_5d, "20d": brier_20d},
        "transition_table": {r: trans.loc[r].to_dict() for r in trans.index},
        "confusion_matrix_vs_spy": cm,
    }


def main() -> None:
    close = fetch_data(YEARS_TOTAL)
    df = run_walk_forward(close)
    results = print_results(df)

    output_path = Path.home() / ".sector_flow" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
