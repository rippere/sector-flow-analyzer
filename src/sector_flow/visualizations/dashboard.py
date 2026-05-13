"""Sector Flow Analyzer — Dash dashboard (Phase 4).

All data is fetched from the FastAPI layer via HTTP. No direct DB access.
Run with: sector-flow dashboard  (default port 8050, API at localhost:8000)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import requests
import requests.exceptions
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, dash_table
from dash.exceptions import PreventUpdate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTOR_NAMES: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

REGIME_COLORS: dict[str, str] = {
    "accumulation": "#00C853",
    "breakout": "#64DD17",
    "distribution": "#D50000",
    "breakdown": "#FF6D00",
    "neutral": "#90A4AE",
    "n/a": "#455A64",
}

# Dark theme palette
BG_PAGE = "#0d1117"
BG_PANEL = "#161b22"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"
BORDER_COLOR = "#30363d"
ACCENT = "#58a6ff"

PLOTLY_TEMPLATE = "plotly_dark"

# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 10) -> Any | None:
    """GET JSON from url; return None on any error."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def fetch_sectors(api_url: str) -> list[dict]:
    return _get(f"{api_url}/sectors") or []


def fetch_regime(api_url: str) -> dict | None:
    return _get(f"{api_url}/analysis/regime")


def fetch_correlations(api_url: str) -> list[dict]:
    return _get(f"{api_url}/analysis/correlations") or []


def fetch_prices(api_url: str, ticker: str) -> list[dict]:
    return _get(f"{api_url}/sectors/{ticker}/prices") or []


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _regime_color(regime: str | None) -> str:
    if regime is None:
        return REGIME_COLORS["n/a"]
    return REGIME_COLORS.get(regime.lower(), REGIME_COLORS["n/a"])


def build_network_graph(sectors: list[dict], correlations: list[dict]) -> go.Figure:
    """Build sector network graph using circular layout via networkx."""
    if not sectors:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_PANEL,
            plot_bgcolor=BG_PANEL,
            title={"text": "Sector Network (no data)", "font": {"color": TEXT_PRIMARY}},
        )
        return fig

    # Build graph and layout
    G = nx.Graph()
    tickers = [s["ticker"] for s in sectors]
    G.add_nodes_from(tickers)
    pos = nx.circular_layout(G)

    # --- Edge traces (significant correlations only) ---
    edge_traces = []
    sig_pairs = [c for c in correlations if c.get("significant")]

    for pair in sig_pairs:
        ta, tb = pair["ticker_a"], pair["ticker_b"]
        if ta not in pos or tb not in pos:
            continue
        x0, y0 = pos[ta]
        x1, y1 = pos[tb]
        corr = pair["correlation"]
        width = max(1.0, abs(corr) * 5)
        if corr > 0:
            edge_color = "rgba(0, 200, 83, 0.4)"
        else:
            edge_color = "rgba(213, 0, 0, 0.4)"
        p_val = pair.get("p_value")
        p_str = f"p={p_val:.3f}" if p_val is not None else "p=n/a"
        hover_text = f"{ta} ↔ {tb}: {corr:.2f} ({p_str})"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={"width": width, "color": edge_color},
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
                name=f"{ta}-{tb}",
            )
        )

    # --- Node trace ---
    sector_map = {s["ticker"]: s for s in sectors}
    row_counts = [sector_map[t]["row_count"] for t in tickers if t in sector_map]
    min_rc = min(row_counts) if row_counts else 1
    max_rc = max(row_counts) if row_counts else 1
    rc_range = max(max_rc - min_rc, 1)

    node_x, node_y, node_text, node_hover, node_colors, node_sizes = [], [], [], [], [], []
    custom_data = []

    for ticker in tickers:
        if ticker not in pos:
            continue
        x, y = pos[ticker]
        node_x.append(x)
        node_y.append(y)
        s = sector_map.get(ticker, {})
        regime = s.get("regime") or "n/a"
        momentum = s.get("momentum")
        mom_str = f"{momentum:+.3f}" if momentum is not None else "n/a"
        sector_name = SECTOR_NAMES.get(ticker, ticker)

        node_text.append(ticker)
        node_hover.append(
            f"<b>{ticker}</b> — {sector_name}<br>"
            f"Regime: {regime}<br>"
            f"Momentum: {mom_str}"
        )
        node_colors.append(_regime_color(regime))

        rc = s.get("row_count", min_rc)
        size = 20 + 30 * (rc - min_rc) / rc_range
        node_sizes.append(size)
        custom_data.append(ticker)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker={
            "size": node_sizes,
            "color": node_colors,
            "line": {"width": 1.5, "color": BORDER_COLOR},
        },
        text=node_text,
        textposition="top center",
        textfont={"color": TEXT_PRIMARY, "size": 11},
        hoverinfo="text",
        hovertext=node_hover,
        customdata=custom_data,
        showlegend=False,
        name="sectors",
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        title={
            "text": f"Sector Network — {len(sig_pairs)} significant correlations",
            "font": {"color": TEXT_PRIMARY, "size": 14},
        },
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "range": [-1.4, 1.4]},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "range": [-1.4, 1.4]},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        hovermode="closest",
        height=480,
        clickmode="event+select",
    )
    return fig


def build_momentum_chart(sectors: list[dict]) -> go.Figure:
    """Horizontal bar chart of sector momentum, sorted descending."""
    if not sectors:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_PANEL,
            title={"text": "Sector Momentum (no data)", "font": {"color": TEXT_PRIMARY}},
        )
        return fig

    sorted_sectors = sorted(
        [s for s in sectors if s.get("momentum") is not None],
        key=lambda s: s["momentum"],
    )
    if not sorted_sectors:
        sorted_sectors = sorted(sectors, key=lambda s: s["ticker"])

    tickers = [s["ticker"] for s in sorted_sectors]
    momentum = [s.get("momentum", 0.0) for s in sorted_sectors]
    colors = [_regime_color(s.get("regime")) for s in sorted_sectors]

    fig = go.Figure(
        go.Bar(
            x=momentum,
            y=tickers,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        title={"text": "Sector Momentum (20-day)", "font": {"color": TEXT_PRIMARY, "size": 13}},
        xaxis={
            "title": "Momentum",
            "range": [-1.05, 1.05],
            "zeroline": True,
            "zerolinecolor": BORDER_COLOR,
            "gridcolor": BORDER_COLOR,
        },
        yaxis={"gridcolor": BORDER_COLOR},
        margin={"l": 60, "r": 10, "t": 40, "b": 30},
        height=280,
    )
    return fig


def build_price_chart(prices: list[dict], ticker: str, regime: str | None) -> go.Figure:
    """Candlestick chart with 20-day SMA overlay for a selected sector."""
    sector_name = SECTOR_NAMES.get(ticker, ticker)
    regime_str = regime or "n/a"
    title_text = f"{ticker} — {sector_name} | {regime_str}"

    if not prices:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_PANEL,
            title={"text": title_text, "font": {"color": TEXT_PRIMARY}},
        )
        return fig

    dates = [p["date"] for p in prices]
    opens = [p["open"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    closes = [p["close"] for p in prices]

    # 20-day SMA
    sma_period = 20
    sma = []
    for i in range(len(closes)):
        if i < sma_period - 1:
            sma.append(None)
        else:
            sma.append(sum(closes[i - sma_period + 1 : i + 1]) / sma_period)

    candle = go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name=ticker,
        increasing_line_color="#00C853",
        decreasing_line_color="#D50000",
    )
    sma_line = go.Scatter(
        x=dates,
        y=sma,
        mode="lines",
        name="20-day SMA",
        line={"color": ACCENT, "width": 1.5, "dash": "dash"},
    )

    fig = go.Figure(data=[candle, sma_line])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        title={"text": title_text, "font": {"color": TEXT_PRIMARY, "size": 13}},
        xaxis={"title": "Date", "rangeslider": {"visible": False}, "gridcolor": BORDER_COLOR},
        yaxis={"title": "Price (USD)", "gridcolor": BORDER_COLOR},
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        height=320,
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
    )
    return fig


def build_flow_table(sectors: list[dict], api_url: str = "") -> html.Div:
    """
    Flow table. Shows real net_inflow_usd if SSGA data exists, otherwise
    shows momentum-ranked table as proxy with a 'pending' label.
    """
    # Try to get the latest net_inflow_usd for each sector from the flows endpoint
    flow_rows: list[dict] = []
    has_real_flow = False

    for s in sectors:
        ticker = s["ticker"]
        flow_val = None
        if api_url:
            flows = _get(f"{api_url}/sectors/{ticker}/flows", timeout=5) or []
            if flows:
                latest = flows[-1]
                flow_val = latest.get("net_inflow_usd")
                if flow_val is not None:
                    has_real_flow = True
        flow_rows.append({
            "ticker": ticker,
            "sector_name": SECTOR_NAMES.get(ticker, ticker),
            "regime": s.get("regime") or "n/a",
            "momentum": s.get("momentum") or 0.0,
            "flow": flow_val,
        })

    # Sort: by real flow if available, else by momentum
    if has_real_flow:
        flow_rows.sort(key=lambda r: r["flow"] or 0.0, reverse=True)
        header_label = "Net Flow (USD M)"
        subtitle = None
    else:
        flow_rows.sort(key=lambda r: r["momentum"], reverse=True)
        header_label = "Momentum (proxy)"
        subtitle = html.P(
            "SSGA flow data accumulates daily. Run `sector-flow ingest` to populate real figures.",
            style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "8px", "fontStyle": "italic"},
        )

    rows = []
    for i, r in enumerate(flow_rows, 1):
        regime_color = _regime_color(r["regime"])
        if has_real_flow and r["flow"] is not None:
            val_m = r["flow"] / 1_000_000
            val_str = f"${val_m:+,.1f}M"
            direction = "↑" if r["flow"] >= 0 else "↓"
            dir_color = "#00C853" if r["flow"] >= 0 else "#D50000"
        else:
            val_str = f"{r['momentum']:+.3f}"
            direction = "↑" if r["momentum"] >= 0 else "↓"
            dir_color = "#00C853" if r["momentum"] >= 0 else "#D50000"

        rows.append(
            html.Tr([
                html.Td(str(i), style={"color": TEXT_MUTED, "fontSize": "12px", "padding": "4px 6px"}),
                html.Td(
                    r["ticker"],
                    style={"color": regime_color, "fontWeight": "600", "fontSize": "12px", "padding": "4px 6px"},
                ),
                html.Td(val_str, style={"color": TEXT_PRIMARY, "fontSize": "12px", "padding": "4px 6px", "textAlign": "right"}),
                html.Td(direction, style={"color": dir_color, "fontSize": "14px", "padding": "4px 6px", "textAlign": "center"}),
            ])
        )

    table = html.Table(
        [
            html.Thead(html.Tr([
                html.Th("#", style={"color": TEXT_MUTED, "fontSize": "11px", "padding": "4px 6px", "textAlign": "left"}),
                html.Th("Ticker", style={"color": TEXT_MUTED, "fontSize": "11px", "padding": "4px 6px", "textAlign": "left"}),
                html.Th(header_label, style={"color": TEXT_MUTED, "fontSize": "11px", "padding": "4px 6px", "textAlign": "right"}),
                html.Th("Dir", style={"color": TEXT_MUTED, "fontSize": "11px", "padding": "4px 6px", "textAlign": "center"}),
            ])),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    children = []
    if subtitle:
        children.append(subtitle)
    children.append(table)
    return html.Div(children)


# ---------------------------------------------------------------------------
# Data health panel
# ---------------------------------------------------------------------------


def build_data_health_panel(sectors: list[dict]) -> html.Div:
    """
    Build the Data Health panel content from the /sectors response.

    Shows:
    - Last ingestion date (most recent latest_date across all ETFs)
    - Flow data coverage (count of sectors with non-null row_count > 0)
    - Analysis window (hardcoded — not surfaced in /sectors)
    - Colored staleness indicator (green < 2 trading days, yellow 2-5, red > 5)

    All data is derived from the existing /sectors endpoint — no new API call.
    """
    from datetime import date as date_type, timedelta

    if not sectors:
        return html.Div(
            "No data available.",
            style={"color": TEXT_MUTED, "fontSize": "12px"},
        )

    # --- Last ingestion date ---
    latest_dates = [s.get("latest_date") for s in sectors if s.get("latest_date")]
    if latest_dates:
        most_recent_str = max(latest_dates)  # ISO date string from API, max() works lexicographically
        last_ingestion = most_recent_str
    else:
        last_ingestion = "unknown"

    # --- Staleness indicator ---
    staleness_color = "#90A4AE"  # grey default
    staleness_label = "unknown"
    trading_days_stale = None

    if latest_dates:
        try:
            from datetime import datetime as dt_cls
            # latest_date comes from API as "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD"
            raw = most_recent_str[:10]  # take date portion
            latest_dt = date_type.fromisoformat(raw)
            today = date_type.today()
            cal_days = (today - latest_dt).days
            # Rough trading-day conversion: ~5/7 of calendar days
            trading_days_stale = max(0, round(cal_days * 5 / 7))
        except (ValueError, TypeError):
            trading_days_stale = None

    if trading_days_stale is not None:
        if trading_days_stale < 2:
            staleness_color = "#00C853"   # green
            staleness_label = f"{trading_days_stale} trading day(s) ago — current"
        elif trading_days_stale <= 5:
            staleness_color = "#FFD600"   # yellow
            staleness_label = f"~{trading_days_stale} trading days ago — slightly stale"
        else:
            staleness_color = "#D50000"   # red
            staleness_label = f"~{trading_days_stale} trading days ago — stale, run ingest"

    # --- Flow coverage (sectors with row_count > 0 as proxy for flow data) ---
    # The /sectors response returns row_count; SSGA data gets merged into the same rows
    sectors_with_data = sum(1 for s in sectors if (s.get("row_count") or 0) > 0)
    total_sectors = len(sectors)
    flow_coverage = f"{sectors_with_data}/{total_sectors} sectors have price data"

    # Analysis window is fixed in config — not surfaced by /sectors
    analysis_window = "30-day correlation / 20-day momentum"

    # --- Build indicator dot ---
    dot = html.Span(
        "●",
        style={
            "color": staleness_color,
            "fontSize": "14px",
            "marginRight": "6px",
        },
    )

    def _row(label: str, value: str | html.Span) -> html.Div:
        return html.Div(
            style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
            children=[
                html.Span(
                    f"{label}:",
                    style={"color": TEXT_MUTED, "fontSize": "12px", "width": "140px", "flexShrink": "0"},
                ),
                html.Span(
                    value,
                    style={"color": TEXT_PRIMARY, "fontSize": "12px"},
                ),
            ],
        )

    staleness_cell = html.Span(
        [dot, html.Span(staleness_label, style={"color": TEXT_PRIMARY, "fontSize": "12px"})],
    )

    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "2px"},
        children=[
            _row("Last ingestion", last_ingestion),
            _row("Staleness", staleness_cell),
            _row("Flow data", flow_coverage),
            _row("Analysis window", analysis_window),
        ],
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _badge(text: str, color: str) -> html.Span:
    return html.Span(
        text,
        style={
            "backgroundColor": color,
            "color": "#ffffff",
            "padding": "3px 10px",
            "borderRadius": "4px",
            "fontWeight": "bold",
            "fontSize": "13px",
            "marginRight": "12px",
        },
    )


def _stat(label: str, value: str) -> html.Span:
    return html.Span(
        [
            html.Span(f"{label}: ", style={"color": TEXT_MUTED, "fontSize": "13px"}),
            html.Span(value, style={"color": TEXT_PRIMARY, "fontWeight": "600", "fontSize": "13px"}),
            html.Span("  ", style={"whiteSpace": "pre"}),
        ]
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(api_url: str = "http://localhost:8000") -> Dash:
    app = Dash(
        __name__,
        title="Sector Flow Analyzer",
        suppress_callback_exceptions=True,
    )

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    app.layout = html.Div(
        style={"backgroundColor": BG_PAGE, "minHeight": "100vh", "fontFamily": "'Segoe UI', sans-serif"},
        children=[
            # Store for API url and selected ticker
            dcc.Store(id="api-url-store", data=api_url),
            dcc.Store(id="selected-ticker-store", data=None),
            dcc.Store(id="sectors-store", data=[]),
            dcc.Store(id="regime-store", data=None),
            dcc.Store(id="correlations-store", data=[]),

            # Auto-refresh interval (60s)
            dcc.Interval(id="auto-refresh", interval=60_000, n_intervals=0),

            # ----------------------------------------------------------------
            # Top bar
            # ----------------------------------------------------------------
            html.Div(
                style={
                    "backgroundColor": BG_PANEL,
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                    "padding": "12px 24px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "flexWrap": "wrap",
                    "gap": "8px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"},
                        children=[
                            html.Span(
                                "Sector Flow Analyzer",
                                style={
                                    "color": TEXT_PRIMARY,
                                    "fontWeight": "700",
                                    "fontSize": "16px",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(id="regime-badge"),
                            html.Div(id="stats-bar"),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "12px"},
                        children=[
                            html.Span(id="last-updated", style={"color": TEXT_MUTED, "fontSize": "12px"}),
                            html.Button(
                                "Refresh Now",
                                id="refresh-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#21262d",
                                    "color": TEXT_PRIMARY,
                                    "border": f"1px solid {BORDER_COLOR}",
                                    "borderRadius": "6px",
                                    "padding": "6px 14px",
                                    "cursor": "pointer",
                                    "fontSize": "13px",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # ----------------------------------------------------------------
            # Error banner (hidden when healthy)
            # ----------------------------------------------------------------
            html.Div(id="error-banner"),

            # ----------------------------------------------------------------
            # Data Health panel (collapsible, above main body)
            # ----------------------------------------------------------------
            html.Details(
                style={
                    "margin": "12px 24px 0",
                    "backgroundColor": BG_PANEL,
                    "border": f"1px solid {BORDER_COLOR}",
                    "borderRadius": "8px",
                    "padding": "0",
                    "overflow": "hidden",
                },
                children=[
                    html.Summary(
                        "Data Health",
                        style={
                            "color": TEXT_MUTED,
                            "fontSize": "12px",
                            "fontWeight": "600",
                            "letterSpacing": "0.05em",
                            "cursor": "pointer",
                            "padding": "8px 14px",
                            "userSelect": "none",
                            "listStyle": "none",
                        },
                    ),
                    html.Div(
                        id="data-health-panel",
                        style={"padding": "8px 16px 12px"},
                    ),
                ],
            ),

            # ----------------------------------------------------------------
            # Main two-column body
            # ----------------------------------------------------------------
            html.Div(
                style={"display": "flex", "gap": "16px", "padding": "16px 24px", "flexWrap": "wrap"},
                children=[
                    # Left column — network graph (60%)
                    html.Div(
                        style={
                            "flex": "6",
                            "minWidth": "360px",
                            "backgroundColor": BG_PANEL,
                            "borderRadius": "8px",
                            "border": f"1px solid {BORDER_COLOR}",
                            "padding": "8px",
                        },
                        children=[dcc.Graph(id="network-graph", config={"displayModeBar": False})],
                    ),
                    # Right column — momentum + flow (40%)
                    html.Div(
                        style={
                            "flex": "4",
                            "minWidth": "280px",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "16px",
                        },
                        children=[
                            html.Div(
                                style={
                                    "backgroundColor": BG_PANEL,
                                    "borderRadius": "8px",
                                    "border": f"1px solid {BORDER_COLOR}",
                                    "padding": "8px",
                                },
                                children=[dcc.Graph(id="momentum-chart", config={"displayModeBar": False})],
                            ),
                            html.Div(
                                style={
                                    "backgroundColor": BG_PANEL,
                                    "borderRadius": "8px",
                                    "border": f"1px solid {BORDER_COLOR}",
                                    "padding": "12px 16px",
                                },
                                children=[
                                    html.H4(
                                        "Net Sector Flows",
                                        style={"color": TEXT_PRIMARY, "margin": "0 0 8px 0", "fontSize": "13px"},
                                    ),
                                    html.Div(id="flow-table"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ----------------------------------------------------------------
            # Price chart — full width, hidden until node clicked
            # ----------------------------------------------------------------
            html.Div(
                id="price-chart-container",
                style={"display": "none", "padding": "0 24px 24px"},
                children=[
                    html.Div(
                        style={
                            "backgroundColor": BG_PANEL,
                            "borderRadius": "8px",
                            "border": f"1px solid {BORDER_COLOR}",
                            "padding": "8px",
                        },
                        children=[dcc.Graph(id="price-chart", config={"displayModeBar": True})],
                    )
                ],
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    @app.callback(
        Output("sectors-store", "data"),
        Output("regime-store", "data"),
        Output("correlations-store", "data"),
        Output("last-updated", "children"),
        Output("error-banner", "children"),
        Input("auto-refresh", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
        State("api-url-store", "data"),
    )
    def refresh_data(n_intervals, n_clicks, api_url):
        sectors = fetch_sectors(api_url)
        regime = fetch_regime(api_url)
        correlations = fetch_correlations(api_url)

        ts = datetime.now().strftime("%H:%M:%S")

        if not sectors and regime is None:
            error = html.Div(
                f"API unreachable at {api_url} — check that `sector-flow serve` is running.",
                style={
                    "backgroundColor": "#3d1c1c",
                    "color": "#ff8080",
                    "border": "1px solid #7d2020",
                    "borderRadius": "6px",
                    "padding": "10px 20px",
                    "margin": "12px 24px 0",
                    "fontSize": "13px",
                },
            )
            return [], None, [], f"Last updated: {ts}", error

        return sectors, regime, correlations, f"Last updated: {ts}", None

    @app.callback(
        Output("regime-badge", "children"),
        Output("stats-bar", "children"),
        Input("regime-store", "data"),
    )
    def update_topbar(regime_data):
        if not regime_data:
            return _badge("NO DATA", REGIME_COLORS["n/a"]), ""

        market_regime = regime_data.get("market_regime", "n/a")
        cohesion = regime_data.get("cohesion", 0.0)
        sig_pairs = regime_data.get("significant_pairs", 0)
        total_pairs = regime_data.get("total_pairs", 0)

        badge = _badge(market_regime.upper(), _regime_color(market_regime))
        stats = html.Span(
            [
                _stat("Cohesion", f"{cohesion:.3f}"),
                _stat("Significant pairs", f"{sig_pairs}/{total_pairs}"),
            ]
        )
        return badge, stats

    @app.callback(
        Output("data-health-panel", "children"),
        Input("sectors-store", "data"),
    )
    def update_data_health(sectors):
        return build_data_health_panel(sectors or [])

    @app.callback(
        Output("network-graph", "figure"),
        Input("sectors-store", "data"),
        Input("correlations-store", "data"),
    )
    def update_network(sectors, correlations):
        return build_network_graph(sectors or [], correlations or [])

    @app.callback(
        Output("momentum-chart", "figure"),
        Input("sectors-store", "data"),
    )
    def update_momentum(sectors):
        return build_momentum_chart(sectors or [])

    @app.callback(
        Output("flow-table", "children"),
        Input("sectors-store", "data"),
        State("api-url-store", "data"),
    )
    def update_flow_table(sectors, stored_api_url):
        return build_flow_table(sectors or [], api_url=stored_api_url or api_url)

    @app.callback(
        Output("selected-ticker-store", "data"),
        Input("network-graph", "clickData"),
        State("selected-ticker-store", "data"),
    )
    def capture_click(click_data, current_ticker):
        if not click_data:
            raise PreventUpdate
        points = click_data.get("points", [])
        if not points:
            raise PreventUpdate
        point = points[0]
        # customdata is the ticker string
        ticker = point.get("customdata")
        if not ticker:
            raise PreventUpdate
        return ticker

    @app.callback(
        Output("price-chart", "figure"),
        Output("price-chart-container", "style"),
        Input("selected-ticker-store", "data"),
        State("api-url-store", "data"),
        State("sectors-store", "data"),
    )
    def update_price_chart(ticker, api_url, sectors):
        visible_style = {"display": "block", "padding": "0 24px 24px"}
        hidden_style = {"display": "none", "padding": "0 24px 24px"}

        if not ticker:
            return go.Figure(), hidden_style

        prices = fetch_prices(api_url, ticker)
        regime = None
        if sectors:
            sector_map = {s["ticker"]: s for s in sectors}
            regime = sector_map.get(ticker, {}).get("regime")

        fig = build_price_chart(prices, ticker, regime)
        return fig, visible_style

    return app


# ---------------------------------------------------------------------------
# Entry point (used by CLI)
# ---------------------------------------------------------------------------

def run_dashboard(api_url: str = "http://localhost:8000", port: int = 8050, debug: bool = False):
    """Start the Dash dashboard server."""
    app = create_app(api_url=api_url)
    app.run(host="0.0.0.0", port=port, debug=debug)
