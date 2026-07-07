# Model Card — Sector Flow Regime Signal

## What it does

Classifies the current market into one of `crisis | risk_on | risk_off | rotation | neutral`
from two independent inputs:

- **Correlation structure** — average pairwise inter-sector correlation (cohesion) and its
  dispersion, computed daily from SPDR sector ETF returns.
- **Flow tilt** — net risk-on (`XLK`/`XLY`/`XLF`) minus risk-off (`XLU`/`XLP`/`XLV`) share of
  daily fund inflows (SSGA), a scale-free measure in `[-1, 1]`.

## Adaptive thresholds (`analysis/regime.py`)

Earlier versions used fixed cutoffs (crisis at avg corr > 0.80, rotation at corr std > 0.25)
calibrated to one historical regime. `compute_adaptive_thresholds()` instead sets:

- **crisis threshold** = 90th percentile of trailing cohesion
- **rotation threshold** = 75th percentile of trailing correlation dispersion

against the market's own recent history (≥30 observations), clipped to sane bounds
(`crisis ∈ [0.55, 0.95]`, `rotation ∈ [0.10, 0.50]`), and falls back to the static constants
on short history. `risk_tilt()` plus a trailing `(mean, std)` baseline replaces the old
"top-3 inflows must be exactly XLK/XLF/XLY" rule with a `>|1σ|` outlier test on flow tilt —
falling back to the legacy subset rule when no baseline is supplied. `classify_market_regime()`
is backward-compatible: all new arguments are optional.

## Validated performance (observed, real yfinance data)

> **Methodology scope — read before citing these numbers.** The walk-forward and
> cross-validation results below were produced by `scripts/backtest_regime.py`, whose
> `compute_regime_at()` is a deliberately simple *baseline* classifier (median 45-day
> cross-sector momentum vs a fixed ±0.5% threshold). It does **not** call
> `classify_market_regime()`, `compute_adaptive_thresholds()`, or `risk_tilt()` — so these
> numbers validate the baseline momentum regime signal, **not** the adaptive-threshold /
> flow-tilt model described above. Additionally, the z-score flow-tilt path is not yet
> exercised in production: `run_analysis()` supplies no `flow_baseline`, so
> `classify_market_regime()` currently takes the legacy top-3-inflows branch in the live
> pipeline. Wiring the backtest to the adaptive classifier (and the baseline into the
> engine) is future work; treat the adaptive-threshold model as unit-tested but not yet
> backtest-validated.

**5-year walk-forward, 45-day momentum window** (`scripts/backtest_regime.py`, in-sample
years 1-2, OOS years 3-5): p=3.26e-20 at the 45d/20d horizon; 61.1% hit rate; risk_off periods
show +2.63% mean 20-day forward return (mean-reversion).

**15-year out-of-sample extension + cross-validation** (3,283 OOS trading days, 2013-06 to
2026-06): hit rate risk_on→5d **60.7%**; 20-day mean forward return risk_off **+2.59%** >
neutral **+0.87%** > risk_on **+0.55%**; regime persistence 95% (risk_on) / 88% (risk_off).

Cross-validated against independent benchmarks the model does **not** use as inputs:
- **VIX** (≥25 risk_off / ≤16 risk_on, prior-close, no look-ahead): **52.9%** agreement.
- **SPY momentum**: 6.4% agreement.

The low SPY agreement and well-above-chance-but-partial VIX agreement indicate the signal
tracks broad fear/greed sentiment but is **distinct from "just the VIX"** rather than a
relabeling of price momentum.

Extended history required trimming: `XLRE` (listed 2015) and `XLC` (listed 2018) post-date
the original nine SPDR sectors (1998), so the backtest only starts once at least
`MIN_SECTORS=9` ETFs have data — avoiding a degenerate 1-2 sector cross-section in the
earliest years.

## Known limitations

- **Not intraday.** Flow data is EOD-only (SSGA limitation). Price-derived signals
  (momentum/correlation/cohesion/regime) can refresh intraday via `run_intraday()`, but
  flow tilt cannot — it is computed from the last EOD flow snapshot.
- **Decision-support only**, not a standalone trading signal — see the mean-reversion edge
  size (+2.59% / 20d) against realistic transaction costs and slippage before sizing.
- Put/call ratio cross-validation is optional and requires a local CSV
  (`~/.sector_flow/put_call.csv`); it is skipped silently when absent.
