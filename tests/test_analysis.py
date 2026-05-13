"""
Phase 2 tests — Analysis engine: covariance, significance, regime, engine.

All tests use synthetic price data (no yfinance calls).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sector_flow.database.models import Base, SECTOR_ETFS, SectorETF, PriceData
from sector_flow.database.repository import (
    ETFRepository,
    PriceRepository,
    FlowMetricRepository,
    CovarianceRepository,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = [t for t, _, _ in SECTOR_ETFS]  # 11 tickers


def _make_price_df(
    n_days: int = 60,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic wide-format price DataFrame."""
    if tickers is None:
        tickers = TICKERS
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    data = {}
    for ticker in tickers:
        price = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
        data[ticker] = price
    return pd.DataFrame(data, index=dates)


def _make_engine_and_session():
    """Create an in-memory SQLite engine + session for integration tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, Session()


# ---------------------------------------------------------------------------
# Module 1 — covariance.py
# ---------------------------------------------------------------------------


class TestCovariance:
    def test_rolling_correlation_shape(self):
        """Output has the correct columns and no NaN after warmup period."""
        from sector_flow.analysis.covariance import compute_rolling_correlation

        price_df = _make_price_df(n_days=60, tickers=["XLK", "XLF", "XLE"])
        result = compute_rolling_correlation(price_df, window=20)

        assert not result.empty
        for col in ["date", "ticker_a", "ticker_b", "correlation", "covariance", "window_days"]:
            assert col in result.columns, f"Missing column: {col}"

        # No NaN values (they are dropped)
        assert result["correlation"].isna().sum() == 0
        assert result["covariance"].isna().sum() == 0

    def test_no_duplicate_pairs(self):
        """ticker_a < ticker_b always holds — no mirror or self-pairs."""
        from sector_flow.analysis.covariance import compute_rolling_correlation

        price_df = _make_price_df(n_days=60)
        result = compute_rolling_correlation(price_df, window=20)

        assert not result.empty
        assert (result["ticker_a"] < result["ticker_b"]).all(), (
            "Found rows where ticker_a >= ticker_b"
        )

    def test_number_of_pairs(self):
        """For N tickers, there should be N*(N-1)/2 unique pairs per date."""
        from sector_flow.analysis.covariance import compute_rolling_correlation

        tickers = ["XLK", "XLF", "XLE"]
        price_df = _make_price_df(n_days=60, tickers=tickers)
        result = compute_rolling_correlation(price_df, window=20)

        latest = result[result["date"] == result["date"].max()]
        expected_pairs = len(tickers) * (len(tickers) - 1) // 2
        assert len(latest) == expected_pairs

    def test_correlation_range(self):
        """All correlations are in [-1, 1]."""
        from sector_flow.analysis.covariance import compute_rolling_correlation

        price_df = _make_price_df(n_days=60)
        result = compute_rolling_correlation(price_df, window=20)
        assert (result["correlation"].abs() <= 1.0 + 1e-9).all()

    def test_pairwise_latest_returns_list_of_dicts(self):
        """compute_pairwise_latest returns the most recent date as a list of dicts."""
        from sector_flow.analysis.covariance import compute_pairwise_latest

        price_df = _make_price_df(n_days=60, tickers=["XLK", "XLF", "XLE"])
        pairs = compute_pairwise_latest(price_df, window=20)

        assert isinstance(pairs, list)
        assert len(pairs) > 0
        for p in pairs:
            assert "ticker_a" in p
            assert "ticker_b" in p
            assert "correlation" in p
            assert "covariance" in p

    def test_build_price_matrix_forward_fills(self):
        """build_price_matrix forward-fills gaps without backfilling."""
        from sector_flow.analysis.covariance import build_price_matrix

        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf = session.query(SectorETF).filter_by(ticker="XLK").first()
        # Insert two rows with a gap
        base = datetime(2024, 1, 2)
        for i in [0, 3]:  # gap on day 1, 2
            session.add(
                PriceData(
                    etf_id=etf.id,
                    date=base + timedelta(days=i),
                    open=100.0, high=102.0, low=99.0,
                    close=101.0 + i, volume=1e6,
                    adjusted_close=101.0 + i,
                )
            )
        session.commit()

        rows = session.query(PriceData).all()
        for row in rows:
            row.etf = etf

        matrix = build_price_matrix(rows, tickers=["XLK"])
        # The gap days should be forward-filled with day 0 value
        assert not matrix.empty
        gap_val = matrix["XLK"].iloc[1]  # day after first row
        assert gap_val == pytest.approx(101.0, abs=1e-3)

        session.close()
        engine.dispose()

    def test_compute_rolling_correlation_empty_on_short_data(self):
        """If data length < window, result should be empty (no complete windows)."""
        from sector_flow.analysis.covariance import compute_rolling_correlation

        price_df = _make_price_df(n_days=5, tickers=["XLK", "XLF"])
        result = compute_rolling_correlation(price_df, window=20)
        assert result.empty


# ---------------------------------------------------------------------------
# Module 2 — significance.py
# ---------------------------------------------------------------------------


class TestSignificance:
    def test_significance_filter_removes_weak(self):
        """Correlations below min_abs_correlation are removed."""
        from sector_flow.analysis.significance import filter_significant_correlations

        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK", "XLK", "XLF"],
                "ticker_b": ["XLF", "XLE", "XLE"],
                "correlation": [0.85, 0.05, -0.90],
                "covariance": [0.01, 0.001, -0.009],
                "window_days": [30, 30, 30],
            }
        )
        result = filter_significant_correlations(corr_df, min_abs_correlation=0.3, p_value_threshold=0.99)

        # 0.05 is below threshold
        kept_corrs = result["correlation"].abs().tolist()
        assert all(c >= 0.3 for c in kept_corrs), f"Weak pair survived: {kept_corrs}"

    def test_significance_filter_has_p_value_column(self):
        """Output always has 'p_value' and 'significant' columns."""
        from sector_flow.analysis.significance import filter_significant_correlations

        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK"],
                "ticker_b": ["XLF"],
                "correlation": [0.80],
                "covariance": [0.01],
                "window_days": [30],
            }
        )
        result = filter_significant_correlations(corr_df)
        assert "p_value" in result.columns
        assert "significant" in result.columns

    def test_compute_p_values_shape(self):
        """compute_p_values returns a row per pair with correct columns."""
        from sector_flow.analysis.significance import compute_p_values

        price_df = _make_price_df(n_days=60, tickers=["XLK", "XLF", "XLE"])
        result = compute_p_values(price_df, window=30)

        assert not result.empty
        for col in ["ticker_a", "ticker_b", "correlation", "p_value"]:
            assert col in result.columns
        # p-values in [0, 1]
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_significance_filter_empty_input(self):
        """Empty input returns empty DataFrame with correct columns."""
        from sector_flow.analysis.significance import filter_significant_correlations

        empty_df = pd.DataFrame(
            columns=["ticker_a", "ticker_b", "correlation", "covariance", "window_days"]
        )
        result = filter_significant_correlations(empty_df)
        assert result.empty
        assert "p_value" in result.columns
        assert "significant" in result.columns


# ---------------------------------------------------------------------------
# Module 3 — regime.py
# ---------------------------------------------------------------------------


class TestRegime:
    def test_regime_crisis_high_correlation(self):
        """Average abs correlation > 0.80 → 'crisis'."""
        from sector_flow.analysis.regime import classify_market_regime

        # Build corr_df where avg |corr| > 0.80
        pairs = [("XLK", "XLF"), ("XLK", "XLE"), ("XLF", "XLE")]
        corr_df = pd.DataFrame(
            {
                "ticker_a": [a for a, _ in pairs],
                "ticker_b": [b for _, b in pairs],
                "correlation": [0.85, 0.90, 0.88],
            }
        )
        regime = classify_market_regime(corr_df)
        assert regime == "crisis"

    def test_regime_risk_on_flow_signal(self):
        """XLK, XLY, XLF top 3 by inflow → 'risk_on'."""
        from sector_flow.analysis.regime import classify_market_regime

        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK"],
                "ticker_b": ["XLF"],
                "correlation": [0.40],
            }
        )
        flow_df = pd.DataFrame(
            {
                "ticker": ["XLK", "XLY", "XLF", "XLU", "XLP", "XLV", "XLE", "XLI", "XLB", "XLRE", "XLC"],
                "net_inflow_usd": [1e9, 9e8, 8e8, 1e6, 1e6, 1e6, 1e5, 1e5, 1e5, 1e5, 1e5],
            }
        )
        regime = classify_market_regime(corr_df, flow_df=flow_df)
        assert regime == "risk_on"

    def test_regime_risk_off_flow_signal(self):
        """XLU, XLP, XLV top 3 by inflow → 'risk_off'."""
        from sector_flow.analysis.regime import classify_market_regime

        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK"],
                "ticker_b": ["XLF"],
                "correlation": [0.30],
            }
        )
        flow_df = pd.DataFrame(
            {
                "ticker": ["XLU", "XLP", "XLV", "XLK", "XLY", "XLF", "XLE", "XLI", "XLB", "XLRE", "XLC"],
                "net_inflow_usd": [1e9, 9e8, 8e8, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4],
            }
        )
        regime = classify_market_regime(corr_df, flow_df=flow_df)
        assert regime == "risk_off"

    def test_regime_rotation_high_dispersion(self):
        """High std of correlations → 'rotation'."""
        from sector_flow.analysis.regime import classify_market_regime

        # Wide spread: some high, some low, some negative
        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLU"],
                "ticker_b": ["XLF", "XLE", "XLV", "XLY", "XLU", "XLP"],
                "correlation": [0.85, -0.80, 0.10, 0.90, -0.75, 0.05],
            }
        )
        regime = classify_market_regime(corr_df)
        assert regime == "rotation"

    def test_regime_neutral_default(self):
        """Moderate, uniform correlations → 'neutral'."""
        from sector_flow.analysis.regime import classify_market_regime

        corr_df = pd.DataFrame(
            {
                "ticker_a": ["XLK", "XLF"],
                "ticker_b": ["XLF", "XLE"],
                "correlation": [0.45, 0.50],
            }
        )
        regime = classify_market_regime(corr_df)
        assert regime == "neutral"

    def test_cohesion_range(self):
        """Cohesion is always in [0, 1]."""
        from sector_flow.analysis.regime import compute_cohesion

        corr_df = _make_price_df(n_days=60)  # wrong type — use DataFrame with correlation col
        # Correct input
        corr_df2 = pd.DataFrame({"correlation": np.random.uniform(-1, 1, 50)})
        val = compute_cohesion(corr_df2)
        assert 0.0 <= val <= 1.0

    def test_cohesion_empty(self):
        """Empty input returns 0.0."""
        from sector_flow.analysis.regime import compute_cohesion

        assert compute_cohesion(pd.DataFrame()) == 0.0
        assert compute_cohesion(pd.DataFrame({"correlation": []})) == 0.0

    def test_sector_regime_accumulation(self):
        """Uptrend + positive flow → 'accumulation'.

        To avoid triggering 'breakout', the last price must be above the
        rolling mean but NOT at the rolling high. We achieve this by having
        a price spike mid-series that exceeds the final value.
        """
        from sector_flow.analysis.regime import classify_sector_regime

        # 40-day series: rises from 90 to ~108, spikes to 115 at day 35, then
        # settles at 110 — final price is above the 20-day mean (~108) but the
        # rolling high (115) is higher, so no breakout fires.
        base = np.linspace(90, 108, 35).tolist()
        spike = [115.0]
        tail = [110.0, 110.0, 110.0, 110.0]
        prices = pd.Series(base + spike + tail)
        # Positive flow throughout
        flows = pd.Series([1e6] * len(prices))
        regime = classify_sector_regime("XLK", prices, flow_series=flows)
        assert regime == "accumulation"

    def test_sector_regime_distribution(self):
        """Downtrend + negative flow → 'distribution'.

        The last price must be below the rolling mean but NOT at the rolling
        low (to avoid 'breakdown'). We achieve this with a mid-series dip.
        """
        from sector_flow.analysis.regime import classify_sector_regime

        # 40-day series: falls from 110 to ~92, dips to 85 at day 35, then
        # settles at 90 — final price is below the 20-day mean (~92) but the
        # rolling low (85) is lower, so no breakdown fires.
        base = np.linspace(110, 92, 35).tolist()
        dip = [85.0]
        tail = [90.0, 90.0, 90.0, 90.0]
        prices = pd.Series(base + dip + tail)
        flows = pd.Series([-1e6] * len(prices))
        regime = classify_sector_regime("XLK", prices, flow_series=flows)
        assert regime == "distribution"

    def test_sector_regime_breakout(self):
        """Price at new 20-day high with positive momentum → 'breakout'."""
        from sector_flow.analysis.regime import classify_sector_regime

        # Make price spike at the end to a new high
        prices = pd.Series([100.0] * 19 + [130.0])
        regime = classify_sector_regime("XLK", prices, flow_series=None)
        assert regime == "breakout"

    def test_sector_regime_breakdown(self):
        """Price at new 20-day low with downward momentum → 'breakdown'."""
        from sector_flow.analysis.regime import classify_sector_regime

        prices = pd.Series([100.0] * 19 + [70.0])
        regime = classify_sector_regime("XLK", prices, flow_series=None)
        assert regime == "breakdown"

    def test_sector_regime_neutral_short_series(self):
        """Single data point → 'neutral'."""
        from sector_flow.analysis.regime import classify_sector_regime

        prices = pd.Series([100.0])
        regime = classify_sector_regime("XLK", prices)
        assert regime == "neutral"

    def test_momentum_clamped(self):
        """compute_momentum is always in [-1, 1]."""
        from sector_flow.analysis.regime import compute_momentum

        rng = np.random.default_rng(0)
        for _ in range(20):
            series = pd.Series(rng.normal(100, 50, 50))
            mom = compute_momentum(series)
            assert -1.0 <= mom <= 1.0, f"Momentum out of range: {mom}"

    def test_momentum_empty(self):
        """Empty series returns 0.0."""
        from sector_flow.analysis.regime import compute_momentum

        assert compute_momentum(pd.Series([], dtype=float)) == 0.0

    def test_momentum_no_all_ones(self):
        """
        With realistic 90-day price data for all 11 sectors, fewer than 4
        sectors should return exactly ±1.0.

        The percentile-rank approach ensures sectors cannot all be pinned to
        ±1.0 unless every sector is simultaneously at its all-time return extreme.
        """
        from sector_flow.analysis.regime import compute_momentum

        rng = np.random.default_rng(123)
        extreme_count = 0
        for _ in range(11):
            # Realistic price series: random walk over 90 days
            prices = 100.0 + np.cumsum(rng.normal(0, 1, 90))
            series = pd.Series(prices)
            mom = compute_momentum(series)
            if abs(mom) == 1.0:
                extreme_count += 1

        assert extreme_count < 4, (
            f"Expected fewer than 4 sectors at ±1.0, got {extreme_count}. "
            f"Percentile rank should differentiate sectors."
        )

    def test_momentum_median_return_near_zero(self):
        """
        A stationary price series (mean-reverting random walk) should produce
        momentum near 0.0 because the current 20-day return will typically
        be close to the median of its own historical distribution.
        """
        from sector_flow.analysis.regime import compute_momentum

        # Mean-reverting series: at each step, price moves toward 100
        # This keeps 20-day returns clustered tightly around 0
        rng = np.random.default_rng(7)
        n = 300
        prices = [100.0]
        for _ in range(n - 1):
            # Mean-reversion: move 10% back toward 100 + small noise
            prev = prices[-1]
            noise = rng.normal(0, 0.3)
            prices.append(prev + 0.1 * (100.0 - prev) + noise)

        series = pd.Series(prices)
        mom = compute_momentum(series)
        # A stationary series: current 20d return should land near the middle
        # of its own distribution — allow ±0.7 tolerance
        assert abs(mom) < 0.7, (
            f"Expected stationary series to have |momentum| < 0.7, got {mom}"
        )

    def test_classify_market_regime_empty_df(self):
        """Empty corr_df → 'neutral'."""
        from sector_flow.analysis.regime import classify_market_regime

        assert classify_market_regime(pd.DataFrame()) == "neutral"


# ---------------------------------------------------------------------------
# Module 4 — engine.py (integration test with in-memory DB)
# ---------------------------------------------------------------------------


class TestEngine:
    def _seed_db_with_prices(self, session, n_days=60):
        """Seed all 11 ETFs + synthetic price data into an in-memory session."""
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        rng = np.random.default_rng(99)
        base = datetime(2024, 1, 2)

        for ticker, _, _ in SECTOR_ETFS:
            etf = session.query(SectorETF).filter_by(ticker=ticker).first()
            prices = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
            for i in range(n_days):
                p = float(prices[i])
                session.add(
                    PriceData(
                        etf_id=etf.id,
                        date=base + timedelta(days=i),
                        open=p, high=p + 1, low=p - 1,
                        close=p, volume=1e6,
                        adjusted_close=p,
                    )
                )
        session.commit()

    def test_engine_run_analysis_returns_all_keys(self, monkeypatch):
        """Integration: run_analysis returns a dict with all required keys."""
        from sector_flow.analysis import engine as engine_module

        mem_engine, session = _make_engine_and_session()
        self._seed_db_with_prices(session, n_days=60)

        # Patch get_session and init_db to use our in-memory session
        from contextlib import contextmanager

        @contextmanager
        def _mock_get_session(db_url=None):
            yield session

        monkeypatch.setattr(engine_module, "get_session", _mock_get_session)
        monkeypatch.setattr(engine_module, "init_db", lambda db_url=None: None)

        result = engine_module.run_analysis(database_url="sqlite:///:memory:", window=20)

        required_keys = {
            "market_regime",
            "cohesion",
            "sector_regimes",
            "sector_momentum",
            "significant_pairs",
            "total_pairs",
        }
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

        assert result["market_regime"] in {"crisis", "risk_on", "risk_off", "rotation", "neutral"}
        assert 0.0 <= result["cohesion"] <= 1.0
        assert isinstance(result["sector_regimes"], dict)
        assert isinstance(result["sector_momentum"], dict)
        assert isinstance(result["significant_pairs"], int)
        assert isinstance(result["total_pairs"], int)
        assert result["total_pairs"] >= result["significant_pairs"]

        session.close()
        mem_engine.dispose()

    def test_engine_sector_regimes_covers_all_tickers(self, monkeypatch):
        """All 11 tickers appear in sector_regimes output."""
        from sector_flow.analysis import engine as engine_module

        mem_engine, session = _make_engine_and_session()
        self._seed_db_with_prices(session, n_days=60)

        from contextlib import contextmanager

        @contextmanager
        def _mock_get_session(db_url=None):
            yield session

        monkeypatch.setattr(engine_module, "get_session", _mock_get_session)
        monkeypatch.setattr(engine_module, "init_db", lambda db_url=None: None)

        result = engine_module.run_analysis(database_url="sqlite:///:memory:", window=20)
        for ticker in TICKERS:
            assert ticker in result["sector_regimes"], f"{ticker} missing from sector_regimes"

        session.close()
        mem_engine.dispose()


# ---------------------------------------------------------------------------
# Module 5 — FlowMetricRepository + CovarianceRepository
# ---------------------------------------------------------------------------


class TestFlowMetricRepository:
    def test_save_and_get_latest(self):
        """save_metric + get_latest round-trip."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf = session.query(SectorETF).filter_by(ticker="XLK").first()
        repo = FlowMetricRepository(session)

        date = datetime(2024, 1, 15)
        repo.save_metric(etf.id, date, "momentum", 0.75)
        session.commit()

        val = repo.get_latest(etf.id, "momentum")
        assert val == pytest.approx(0.75)

        session.close()
        engine.dispose()

    def test_save_metric_upsert(self):
        """Saving twice for same (etf, date, metric) updates in place."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf = session.query(SectorETF).filter_by(ticker="XLK").first()
        repo = FlowMetricRepository(session)

        date = datetime(2024, 1, 15)
        repo.save_metric(etf.id, date, "momentum", 0.5)
        session.commit()
        repo.save_metric(etf.id, date, "momentum", 0.9)
        session.commit()

        val = repo.get_latest(etf.id, "momentum")
        assert val == pytest.approx(0.9)

        from sector_flow.database.models import FlowMetric
        count = session.query(FlowMetric).filter_by(etf_id=etf.id, metric_name="momentum").count()
        assert count == 1

        session.close()
        engine.dispose()

    def test_get_series(self):
        """get_series returns ordered time series."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf = session.query(SectorETF).filter_by(ticker="XLK").first()
        repo = FlowMetricRepository(session)

        for i in range(5):
            repo.save_metric(etf.id, datetime(2024, 1, i + 1), "momentum", float(i) * 0.1)
        session.commit()

        series = repo.get_series(etf.id, "momentum")
        assert len(series) == 5
        dates = [r.date for r in series]
        assert dates == sorted(dates)

        session.close()
        engine.dispose()

    def test_get_latest_returns_none_when_empty(self):
        """get_latest returns None when no rows exist."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf = session.query(SectorETF).filter_by(ticker="XLK").first()
        repo = FlowMetricRepository(session)

        val = repo.get_latest(etf.id, "nonexistent_metric")
        assert val is None

        session.close()
        engine.dispose()


class TestCovarianceRepository:
    def test_save_pairs_and_get_latest_matrix(self):
        """save_pairs + get_latest_matrix round-trip."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf_a = session.query(SectorETF).filter_by(ticker="XLK").first()
        etf_b = session.query(SectorETF).filter_by(ticker="XLF").first()

        repo = CovarianceRepository(session)
        computed_at = datetime(2024, 3, 1, 12, 0, 0)
        pairs = [
            {
                "etf_a_id": etf_a.id,
                "etf_b_id": etf_b.id,
                "window_days": 30,
                "computed_at": computed_at,
                "covariance": 0.015,
                "correlation": 0.72,
            }
        ]
        saved = repo.save_pairs(pairs)
        session.commit()
        assert saved == 1

        matrix = repo.get_latest_matrix(window=30)
        assert len(matrix) == 1
        assert matrix[0].correlation == pytest.approx(0.72)

        session.close()
        engine.dispose()

    def test_save_pairs_upsert(self):
        """Saving the same pair twice updates rather than inserts."""
        engine, session = _make_engine_and_session()
        etf_repo = ETFRepository(session)
        etf_repo.seed_etfs()
        session.commit()

        etf_a = session.query(SectorETF).filter_by(ticker="XLK").first()
        etf_b = session.query(SectorETF).filter_by(ticker="XLF").first()
        repo = CovarianceRepository(session)
        computed_at = datetime(2024, 3, 1, 12, 0, 0)
        pair = {
            "etf_a_id": etf_a.id,
            "etf_b_id": etf_b.id,
            "window_days": 30,
            "computed_at": computed_at,
            "covariance": 0.01,
            "correlation": 0.50,
        }
        repo.save_pairs([pair])
        session.commit()

        pair["correlation"] = 0.80
        pair["covariance"] = 0.02
        repo.save_pairs([pair])
        session.commit()

        from sector_flow.database.models import CovarianceMatrix
        count = session.query(CovarianceMatrix).count()
        assert count == 1

        matrix = repo.get_latest_matrix(window=30)
        assert matrix[0].correlation == pytest.approx(0.80)

        session.close()
        engine.dispose()

    def test_get_latest_matrix_empty(self):
        """get_latest_matrix returns [] when no data."""
        engine, session = _make_engine_and_session()
        Base.metadata.create_all(engine)
        repo = CovarianceRepository(session)
        result = repo.get_latest_matrix(window=30)
        assert result == []

        session.close()
        engine.dispose()
