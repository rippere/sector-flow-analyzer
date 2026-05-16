"""
Pure-function tests for sector_flow.visualizations.dashboard.

No running server required. Graph-builder and layout-builder functions
are called directly with synthetic data; create_app() is smoke-tested for
instantiation and layout structure only.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import plotly.graph_objects as go
import pytest
from dash import Dash, html

from sector_flow.visualizations.dashboard import (
    REGIME_COLORS,
    _regime_color,
    build_data_health_panel,
    build_flow_table,
    build_momentum_chart,
    build_network_graph,
    build_price_chart,
    create_app,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SECTORS = [
    {"ticker": "XLK", "sector_name": "Technology", "regime": "accumulation", "momentum": 0.8, "cohesion": 0.6, "latest_close": 200.0, "latest_date": "2025-01-10", "row_count": 60},
    {"ticker": "XLF", "sector_name": "Financials", "regime": "neutral", "momentum": 0.1, "cohesion": 0.4, "latest_close": 40.0, "latest_date": "2025-01-10", "row_count": 60},
    {"ticker": "XLE", "sector_name": "Energy", "regime": "distribution", "momentum": -0.5, "cohesion": 0.3, "latest_close": 80.0, "latest_date": "2025-01-09", "row_count": 60},
]

_CORRELATIONS = [
    {"ticker_a": "XLK", "ticker_b": "XLF", "correlation": 0.82, "p_value": 0.001, "significant": True},
    {"ticker_a": "XLF", "ticker_b": "XLE", "correlation": -0.3, "p_value": 0.8, "significant": False},
]

_PRICES = [
    {"date": f"2025-01-{i+1:02d}", "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i, "close": 101.0 + i, "adjusted_close": 101.0 + i}
    for i in range(30)
]


# ---------------------------------------------------------------------------
# _regime_color
# ---------------------------------------------------------------------------

class TestRegimeColor:
    def test_known_regimes_map_to_expected_color(self):
        assert _regime_color("accumulation") == REGIME_COLORS["accumulation"]
        assert _regime_color("distribution") == REGIME_COLORS["distribution"]
        assert _regime_color("neutral") == REGIME_COLORS["neutral"]

    def test_none_returns_na_color(self):
        assert _regime_color(None) == REGIME_COLORS["n/a"]

    def test_unknown_string_returns_na_color(self):
        assert _regime_color("bogus") == REGIME_COLORS["n/a"]

    def test_case_insensitive(self):
        assert _regime_color("ACCUMULATION") == REGIME_COLORS["accumulation"]
        assert _regime_color("Breakout") == REGIME_COLORS["breakout"]


# ---------------------------------------------------------------------------
# build_network_graph
# ---------------------------------------------------------------------------

class TestBuildNetworkGraph:
    def test_empty_sectors_returns_figure(self):
        fig = build_network_graph([], [])
        assert isinstance(fig, go.Figure)

    def test_empty_sectors_title_contains_no_data(self):
        fig = build_network_graph([], [])
        assert "no data" in fig.layout.title.text.lower()

    def test_populated_sectors_returns_figure(self):
        fig = build_network_graph(_SECTORS, [])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # at least the node trace

    def test_significant_correlations_add_edge_traces(self):
        fig_no_corr = build_network_graph(_SECTORS, [])
        fig_with_corr = build_network_graph(_SECTORS, _CORRELATIONS)
        assert len(fig_with_corr.data) > len(fig_no_corr.data)

    def test_insignificant_correlations_excluded(self):
        insignificant = [{"ticker_a": "XLF", "ticker_b": "XLE", "correlation": -0.3, "p_value": 0.8, "significant": False}]
        fig_none = build_network_graph(_SECTORS, [])
        fig_insig = build_network_graph(_SECTORS, insignificant)
        assert len(fig_insig.data) == len(fig_none.data)

    def test_title_shows_significant_pair_count(self):
        # _CORRELATIONS has 1 significant pair
        fig = build_network_graph(_SECTORS, _CORRELATIONS)
        assert "1" in fig.layout.title.text

    def test_node_trace_has_customdata_tickers(self):
        fig = build_network_graph(_SECTORS, [])
        node_trace = fig.data[-1]  # node trace is always last
        tickers = set(node_trace.customdata)
        assert "XLK" in tickers
        assert "XLF" in tickers


# ---------------------------------------------------------------------------
# build_momentum_chart
# ---------------------------------------------------------------------------

class TestBuildMomentumChart:
    def test_empty_returns_figure(self):
        fig = build_momentum_chart([])
        assert isinstance(fig, go.Figure)

    def test_empty_title_contains_no_data(self):
        fig = build_momentum_chart([])
        assert "no data" in fig.layout.title.text.lower()

    def test_populated_returns_bar_figure(self):
        fig = build_momentum_chart(_SECTORS)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_bars_sorted_ascending_by_momentum(self):
        fig = build_momentum_chart(_SECTORS)
        bar = fig.data[0]
        # x is momentum values; y is tickers — sorted ascending
        x_values = list(bar.x)
        assert x_values == sorted(x_values)

    def test_all_sectors_represented(self):
        fig = build_momentum_chart(_SECTORS)
        bar = fig.data[0]
        assert set(bar.y) == {"XLK", "XLF", "XLE"}

    def test_sectors_without_momentum_still_renders(self):
        no_mom = [{"ticker": "XLK", "regime": "neutral", "momentum": None, "row_count": 10}]
        fig = build_momentum_chart(no_mom)
        assert isinstance(fig, go.Figure)

    def test_orientation_is_horizontal(self):
        fig = build_momentum_chart(_SECTORS)
        assert fig.data[0].orientation == "h"


# ---------------------------------------------------------------------------
# build_price_chart
# ---------------------------------------------------------------------------

class TestBuildPriceChart:
    def test_empty_prices_returns_figure(self):
        fig = build_price_chart([], "XLK", "accumulation")
        assert isinstance(fig, go.Figure)

    def test_empty_prices_title_contains_ticker(self):
        fig = build_price_chart([], "XLK", None)
        assert "XLK" in fig.layout.title.text

    def test_regime_none_shows_na_in_title(self):
        fig = build_price_chart([], "XLK", None)
        assert "n/a" in fig.layout.title.text

    def test_populated_returns_two_traces(self):
        fig = build_price_chart(_PRICES, "XLK", "accumulation")
        assert len(fig.data) == 2

    def test_first_trace_is_candlestick(self):
        fig = build_price_chart(_PRICES, "XLK", "accumulation")
        assert isinstance(fig.data[0], go.Candlestick)

    def test_second_trace_is_sma_scatter(self):
        fig = build_price_chart(_PRICES, "XLK", "accumulation")
        sma = fig.data[1]
        assert isinstance(sma, go.Scatter)
        assert "SMA" in sma.name

    def test_sma_first_19_values_are_none(self):
        # Need at least 20 candles; _PRICES has 30
        fig = build_price_chart(_PRICES, "XLK", None)
        sma_y = list(fig.data[1].y)
        assert all(v is None for v in sma_y[:19])
        assert sma_y[19] is not None

    def test_sma_value_at_position_20_is_correct(self):
        fig = build_price_chart(_PRICES, "XLK", None)
        sma_y = list(fig.data[1].y)
        closes = [p["close"] for p in _PRICES]
        expected = sum(closes[:20]) / 20
        assert abs(sma_y[19] - expected) < 1e-9


# ---------------------------------------------------------------------------
# build_flow_table
# ---------------------------------------------------------------------------

class TestBuildFlowTable:
    def test_empty_sectors_returns_div(self):
        result = build_flow_table([])
        assert isinstance(result, html.Div)

    def test_populated_no_api_url_returns_div(self):
        result = build_flow_table(_SECTORS, api_url="")
        assert isinstance(result, html.Div)

    def test_no_api_url_uses_momentum_proxy(self):
        result = build_flow_table(_SECTORS, api_url="")
        # Without real flow data, a subtitle paragraph should be rendered
        children = result.children
        has_subtitle = any(isinstance(c, html.P) for c in children)
        assert has_subtitle

    def test_with_real_flow_data_no_subtitle(self):
        fake_flow = [{"net_inflow_usd": 1_000_000.0, "aum_usd": 5e10}]
        with patch("sector_flow.visualizations.dashboard._get", return_value=fake_flow):
            result = build_flow_table(_SECTORS, api_url="http://fake")
        children = result.children
        has_subtitle = any(isinstance(c, html.P) for c in children)
        assert not has_subtitle

    def test_row_count_matches_sector_count(self):
        result = build_flow_table(_SECTORS, api_url="")
        # Locate the table body rows
        table = _find_component(result, html.Table)
        assert table is not None
        tbody = next(c for c in table.children if isinstance(c, html.Tbody))
        assert len(tbody.children) == len(_SECTORS)


# ---------------------------------------------------------------------------
# build_data_health_panel
# ---------------------------------------------------------------------------

class TestBuildDataHealthPanel:
    def test_empty_returns_div(self):
        result = build_data_health_panel([])
        assert isinstance(result, html.Div)

    def test_empty_contains_no_data_text(self):
        result = build_data_health_panel([])
        assert "No data" in str(result.children)

    def test_populated_returns_div_with_rows(self):
        result = build_data_health_panel(_SECTORS)
        assert isinstance(result, html.Div)
        # Should have 4 info rows
        assert len(result.children) == 4

    def test_latest_date_is_max_across_sectors(self):
        result = build_data_health_panel(_SECTORS)
        # _SECTORS has latest_date "2025-01-10" and "2025-01-09" → max is "2025-01-10"
        rendered = str(result)
        assert "2025-01-10" in rendered

    def test_sectors_without_latest_date_show_unknown(self):
        no_date = [{"ticker": "XLK", "regime": "neutral", "momentum": 0.5, "row_count": 10, "latest_date": None}]
        result = build_data_health_panel(no_date)
        assert "unknown" in str(result)

    def test_flow_coverage_counts_nonzero_row_count(self):
        mixed = [
            {"ticker": "XLK", "row_count": 60, "latest_date": None, "regime": None, "momentum": None},
            {"ticker": "XLF", "row_count": 0, "latest_date": None, "regime": None, "momentum": None},
        ]
        result = build_data_health_panel(mixed)
        rendered = str(result)
        assert "1/2" in rendered

    def test_recent_date_yields_green_indicator(self):
        today = date.today()
        sectors = [{"ticker": "XLK", "row_count": 10, "latest_date": today.isoformat(), "regime": None, "momentum": None}]
        result = build_data_health_panel(sectors)
        assert "#00C853" in str(result)

    def test_stale_date_yields_red_indicator(self):
        old = date.today() - timedelta(days=30)
        sectors = [{"ticker": "XLK", "row_count": 10, "latest_date": old.isoformat(), "regime": None, "momentum": None}]
        result = build_data_health_panel(sectors)
        assert "#D50000" in str(result)


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------

class TestCreateApp:
    def test_returns_dash_instance(self):
        app = create_app()
        assert isinstance(app, Dash)

    def test_custom_api_url_stored(self):
        app = create_app(api_url="http://myserver:9000")
        # The api-url-store component should have our URL as its data
        store = _find_by_id(app.layout, "api-url-store")
        assert store is not None
        assert store.data == "http://myserver:9000"

    def test_layout_has_network_graph(self):
        app = create_app()
        assert _find_by_id(app.layout, "network-graph") is not None

    def test_layout_has_momentum_chart(self):
        app = create_app()
        assert _find_by_id(app.layout, "momentum-chart") is not None

    def test_layout_has_flow_table(self):
        app = create_app()
        assert _find_by_id(app.layout, "flow-table") is not None

    def test_layout_has_price_chart(self):
        app = create_app()
        assert _find_by_id(app.layout, "price-chart") is not None

    def test_layout_has_data_health_panel(self):
        app = create_app()
        assert _find_by_id(app.layout, "data-health-panel") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_component(root, component_type):
    """DFS search for first component of a given type."""
    if isinstance(root, component_type):
        return root
    children = getattr(root, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        result = _find_component(child, component_type)
        if result is not None:
            return result
    return None


def _find_by_id(root, component_id: str):
    """DFS search for component with matching id."""
    if hasattr(root, "id") and root.id == component_id:
        return root
    children = getattr(root, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        result = _find_by_id(child, component_id)
        if result is not None:
            return result
    return None
