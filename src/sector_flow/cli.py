import click
import requests


@click.group()
def main():
    """Sector Flow Analyzer CLI."""
    pass


@main.command()
@click.option("--days", default=90, show_default=True, help="Days of history to fetch")
@click.option("--db", default=None, help="Database URL (default: from .env)")
def backfill(days: int, db: str | None):
    """Pull historical OHLCV data for all 11 sectors via yfinance."""
    from sector_flow.pipeline import backfill as run_backfill
    result = run_backfill(days=days, database_url=db)
    click.echo(f"Backfill complete: {result['rows_saved']} rows saved")
    if result["errors"]:
        click.echo(f"Errors: {result['errors']}", err=True)


@main.command()
@click.option("--db", default=None, help="Database URL (default: from .env)")
def ingest(db: str | None):
    """Run the daily ingestion pipeline (OHLCV + SSGA flow snapshot)."""
    from sector_flow.pipeline import run_daily
    result = run_daily(database_url=db)
    click.echo(
        f"Ingestion complete — yfinance: {result['yfinance_rows']} rows, "
        f"SSGA: {result['ssga_rows']} snapshots, "
        f"flows computed: {result['flow_rows_updated']}"
    )
    if result["errors"]:
        click.echo(f"Errors: {result['errors']}", err=True)


@main.command()
@click.option("--db", default=None, help="Database URL (default: from .env)")
def status(db: str | None):
    """Show data freshness and row counts per sector."""
    from sector_flow.database.session import get_session, init_db
    from sector_flow.database.repository import ETFRepository, PriceRepository
    init_db(db)
    with get_session(db) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        etfs = etf_repo.get_all()
        click.echo(f"{'Ticker':<8} {'Rows':>6} {'Latest date':<22} {'Has flow data'}")
        click.echo("-" * 55)
        for etf in sorted(etfs, key=lambda e: e.ticker):
            rows = price_repo.get_price_data(etf.id)
            latest = price_repo.get_latest_date(etf.id)
            flow_rows = sum(1 for r in rows if r.net_inflow_usd is not None)
            click.echo(
                f"{etf.ticker:<8} {len(rows):>6} "
                f"{str(latest)[:19] if latest else 'none':<22} "
                f"{flow_rows}/{len(rows)} rows"
            )


@main.command()
@click.option("--db", default=None, help="Database URL (default: from .env)")
@click.option("--window", default=30, show_default=True, help="Rolling window in days")
def analyze(db: str | None, window: int):
    """Run the analysis engine: compute correlations, classify regimes, persist results."""
    from sector_flow.analysis.engine import run_analysis
    result = run_analysis(database_url=db, window=window)

    click.echo("\n=== Market Regime Analysis ===")
    click.echo(f"  Market Regime : {result['market_regime'].upper()}")
    click.echo(f"  Cohesion      : {result['cohesion']:.3f}")
    click.echo(f"  Sig. Pairs    : {result['significant_pairs']} / {result['total_pairs']}")
    click.echo("\n  Sector Regimes & Momentum:")
    click.echo(f"  {'Ticker':<8} {'Regime':<16} {'Momentum':>9}")
    click.echo("  " + "-" * 36)
    for ticker in sorted(result["sector_regimes"]):
        regime = result["sector_regimes"][ticker]
        mom = result["sector_momentum"].get(ticker, 0.0)
        click.echo(f"  {ticker:<8} {regime:<16} {mom:>+9.3f}")
    click.echo()


@main.command()
@click.option("--db", default=None, help="Database URL (default: from .env)")
@click.option("--force", is_flag=True, default=False,
              help="Run even when the market is closed (default: no-op when closed)")
def intraday(db: str | None, force: bool):
    """Intraday refresh: update today's price bar and recompute price-derived signals.

    Skips the SSGA flow snapshot (flows are EOD-only). No-ops outside regular
    trading hours unless --force is given.
    """
    from sector_flow.pipeline import run_intraday
    result = run_intraday(database_url=db, force=force)
    if result.get("skipped"):
        click.echo(f"Intraday skipped: {result['skipped']}")
        return
    click.echo(
        f"Intraday refresh — yfinance: {result['yfinance_rows']} bars, "
        f"regime: {result.get('market_regime', 'n/a').upper()}, "
        f"cohesion: {result.get('cohesion', 0.0):.3f}"
    )
    if result["errors"]:
        click.echo(f"Errors: {result['errors']}", err=True)


@main.command()
@click.option("--api-url", default="http://localhost:8000", show_default=True, help="FastAPI base URL")
@click.option("--device", default="console", show_default=True,
              type=click.Choice(["console", "inky", "oled"]),
              help="Output device (console saves a PNG; inky/oled need Pi hardware)")
@click.option("--out", default="/tmp/sector_flow_panel.png", show_default=True,
              help="PNG path for --device console")
def display(api_url: str, device: str, out: str):
    """Render the ambient signal panel from live API data to a device (WS4)."""
    from sector_flow.hardware import build_snapshot, get_display, ConsoleDisplay

    snapshot = build_snapshot(api_url)
    dev = ConsoleDisplay(out_path=out) if device == "console" else get_display(device)
    dev.show(snapshot)
    if device == "console":
        click.echo(f"Panel rendered → {out}  (regime: {snapshot.regime.upper()}, "
                   f"lead: {snapshot.dominant}, status: {snapshot.market_status})")
    else:
        click.echo(f"Pushed panel to {device}  (regime: {snapshot.regime.upper()})")


@main.command(name="show-regime")
@click.option("--db", default=None, help="Database URL (default: from .env)")
def show_regime(db: str | None):
    """Print current market regime and per-sector regimes from stored FlowMetric rows."""
    from sector_flow.database.session import get_session, init_db
    from sector_flow.database.repository import ETFRepository, FlowMetricRepository

    init_db(db)
    with get_session(db) as session:
        etf_repo = ETFRepository(session)
        flow_repo = FlowMetricRepository(session)
        etfs = etf_repo.get_all()

        if not etfs:
            click.echo("No ETFs found — run `sector-flow backfill` first.")
            return

        click.echo(f"\n{'Ticker':<8} {'Regime':>12} {'Momentum':>10} {'Cohesion':>10}")
        click.echo("-" * 44)
        for etf in sorted(etfs, key=lambda e: e.ticker):
            regime_val = flow_repo.get_latest(etf.id, "regime_label")
            momentum = flow_repo.get_latest(etf.id, "momentum")
            cohesion = flow_repo.get_latest(etf.id, "cohesion")

            _regime_inv = {1.0: "accumulation", 2.0: "breakout", 0.0: "neutral",
                           -1.0: "distribution", -2.0: "breakdown"}
            regime_str = _regime_inv.get(regime_val, "n/a") if regime_val is not None else "n/a"
            mom_str = f"{momentum:+.3f}" if momentum is not None else "n/a"
            coh_str = f"{cohesion:.3f}" if cohesion is not None else "n/a"
            click.echo(f"{etf.ticker:<8} {regime_str:>12} {mom_str:>10} {coh_str:>10}")
        click.echo()


@main.command(name="show-flows")
@click.option("--db", default=None, help="Database URL (default: from .env)")
def show_flows(db: str | None):
    """Print net_inflow_usd ranking across sectors (requires SSGA flow data)."""
    from sector_flow.database.session import get_session, init_db
    from sector_flow.database.repository import ETFRepository, PriceRepository

    init_db(db)
    with get_session(db) as session:
        etf_repo = ETFRepository(session)
        price_repo = PriceRepository(session)
        etfs = etf_repo.get_all()

        if not etfs:
            click.echo("No ETFs found — run `sector-flow backfill` first.")
            return

        rows = []
        for etf in etfs:
            price_rows = price_repo.get_price_data(etf.id)
            flow_vals = [(r.date, r.net_inflow_usd) for r in price_rows if r.net_inflow_usd is not None]
            if flow_vals:
                latest_date, latest_flow = max(flow_vals, key=lambda x: x[0])
                rows.append((etf.ticker, latest_flow, latest_date))
            else:
                rows.append((etf.ticker, None, None))

        rows_with_flow = [(t, f, d) for t, f, d in rows if f is not None]
        rows_no_flow = [(t, f, d) for t, f, d in rows if f is None]

        if not rows_with_flow:
            click.echo("No net_inflow_usd data found. Run SSGA ingest first (`sector-flow ingest`).")
            return

        rows_with_flow.sort(key=lambda x: x[1], reverse=True)
        click.echo(f"\n{'Rank':<6} {'Ticker':<8} {'Net Inflow (USD)':>18} {'As Of':<22}")
        click.echo("-" * 58)
        for rank, (ticker, flow, date) in enumerate(rows_with_flow, 1):
            flow_str = f"${flow:>+,.0f}"
            date_str = str(date)[:10] if date else "n/a"
            click.echo(f"{rank:<6} {ticker:<8} {flow_str:>18} {date_str:<22}")

        if rows_no_flow:
            click.echo(f"\n  No flow data: {', '.join(t for t, _, _ in rows_no_flow)}")
        click.echo()


@main.command()
@click.option("--api-url", default="http://localhost:8000", show_default=True, help="FastAPI base URL")
@click.option("--port", default=8050, show_default=True, type=int, help="Dash server port")
@click.option("--debug/--no-debug", default=False, help="Enable Dash debug mode")
def dashboard(api_url: str, port: int, debug: bool):
    """Start the interactive Dash dashboard (expects API running on --api-url)."""
    from sector_flow.visualizations.dashboard import run_dashboard
    click.echo(f"Starting dashboard on http://0.0.0.0:{port}  (API: {api_url})")
    run_dashboard(api_url=api_url, port=port, debug=debug)


@main.command()
@click.option("--host", default=None, help="Bind host (default: from settings)")
@click.option("--port", default=None, type=int, help="Bind port (default: from settings)")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode)")
def serve(host: str | None, port: int | None, reload: bool):
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    from sector_flow.config import settings

    _host = host or settings.api_host
    _port = port or settings.api_port
    uvicorn.run(
        "sector_flow.api.app:app",
        host=_host,
        port=_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@main.command(name="osc-bridge")
@click.option("--api-url", default="http://localhost:8000", show_default=True, help="FastAPI base URL")
@click.option("--osc-host", default="127.0.0.1", show_default=True, help="OSC target host")
@click.option("--osc-port", default=9000, show_default=True, type=int, help="OSC target port")
@click.option("--interval", default=1, show_default=True, type=int, help="Broadcast interval in seconds")
def osc_bridge(api_url: str, osc_host: str, osc_port: int, interval: int):
    """Start the OSC bridge — broadcasts sector flow data as OSC messages (e.g. to nw_wrld)."""
    from sector_flow.integrations.osc_bridge import OSCBridge
    bridge = OSCBridge(api_url=api_url, osc_host=osc_host, osc_port=osc_port)
    bridge.run_forever(interval_seconds=interval)


_SECTOR_FULL_NAMES: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretion",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Svcs",
}


def _build_query_output(api_url: str) -> str:
    """Fetch live data from the API and return a formatted analysis string."""
    from datetime import date

    base = api_url.rstrip("/")
    today = date.today().isoformat()

    try:
        regime_resp = requests.get(f"{base}/analysis/regime", timeout=8)
        regime_resp.raise_for_status()
        regime_data = regime_resp.json()
    except Exception as exc:
        return f"ERROR: Could not reach API at {api_url} — {exc}"

    try:
        corr_resp = requests.get(f"{base}/analysis/correlations", timeout=8)
        corr_resp.raise_for_status()
        correlations = corr_resp.json()
    except Exception:
        correlations = []

    market_regime: str = regime_data.get("market_regime", "unknown").upper()
    cohesion: float = regime_data.get("cohesion", 0.0)
    sig_pairs: int = regime_data.get("significant_pairs", 0)
    total_pairs: int = regime_data.get("total_pairs", 0)
    sector_momentum: dict[str, float] = regime_data.get("sector_momentum", {})
    sector_regimes: dict[str, str] = regime_data.get("sector_regimes", {})

    # Sort sectors by momentum descending
    sorted_tickers = sorted(sector_momentum.keys(), key=lambda t: sector_momentum[t], reverse=True)

    leaders = [t for t in sorted_tickers if sector_momentum[t] > 0][:5]
    laggards = [t for t in reversed(sorted_tickers) if sector_momentum[t] < 0][:5]

    lines: list[str] = []
    lines.append(f"=== Sector Flow Analysis — {today} ===")
    lines.append("")
    lines.append(
        f"Market: {market_regime} | Cohesion: {cohesion:.2f} | "
        f"{sig_pairs}/{total_pairs} significant pairs"
    )
    lines.append("")

    if leaders:
        lines.append("Momentum Leaders (accumulating):")
        for t in leaders:
            name = _SECTOR_FULL_NAMES.get(t, t)
            mom = sector_momentum[t]
            reg = sector_regimes.get(t, "neutral")
            lines.append(f"  {t:<6}  {name:<22}  {mom:+.3f}  [{reg}]")
    else:
        lines.append("Momentum Leaders: none")

    lines.append("")

    if laggards:
        lines.append("Momentum Laggards (distributing):")
        for t in laggards:
            name = _SECTOR_FULL_NAMES.get(t, t)
            mom = sector_momentum[t]
            reg = sector_regimes.get(t, "neutral")
            lines.append(f"  {t:<6}  {name:<22}  {mom:+.3f}  [{reg}]")
    else:
        lines.append("Momentum Laggards: none")

    lines.append("")
    lines.append("Rotation Thesis:")

    # Build narrative
    leader_names = ", ".join(_SECTOR_FULL_NAMES.get(t, t) for t in leaders[:3])
    laggard_names = " and ".join(_SECTOR_FULL_NAMES.get(t, t) for t in laggards[:2])
    cohesion_desc = "high" if cohesion > 0.7 else "moderate" if cohesion > 0.4 else "low"

    lines.append(
        f"  Current market regime is {market_regime} with {cohesion_desc} cohesion ({cohesion:.2f})."
    )
    if leader_names:
        lines.append(
            f"  Capital appears to be rotating INTO {leader_names}"
        )
    if laggard_names:
        lines.append(
            f"  and OUT OF {laggard_names}."
        )

    # Call out distribution sectors
    dist_sectors = [
        t for t in sorted_tickers
        if sector_regimes.get(t) in ("distribution", "breakdown")
    ]
    if dist_sectors:
        for dt in dist_sectors[:2]:
            dn = _SECTOR_FULL_NAMES.get(dt, dt)
            dr = sector_regimes[dt]
            lines.append(
                f"  {dt} ({dn}) shows active {dr} — watch for further weakness."
            )

    # Top correlations
    if correlations:
        lines.append("")
        lines.append("Strongest Correlations:")
        top_corr = sorted(correlations, key=lambda c: abs(c.get("correlation", 0)), reverse=True)[:6]
        pairs_str = "  " + "  ".join(
            f"{p['ticker_a']} ↔ {p['ticker_b']}: {p['correlation']:+.2f}"
            for p in top_corr
        )
        lines.append(pairs_str)

    return "\n".join(lines)


@main.command()
@click.argument("question", required=False)
@click.option("--api-url", default="http://localhost:8000", show_default=True, help="FastAPI base URL")
def query(question: str | None, api_url: str):
    """Print a structured natural-language sector rotation analysis from live API data.

    QUESTION is optional — the same analysis is always shown regardless of the query text.
    """
    output = _build_query_output(api_url)
    click.echo(output)


@main.command(name="flow-import")
@click.argument("csv_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database URL (default: from .env)")
@click.option("--dry-run", is_flag=True, default=False, help="Validate and preview without writing")
def flow_import(csv_file: str, db: str | None, dry_run: bool):
    """Import flow metrics from a CSV file into the FlowMetric table.

    Expected columns: ticker, date, metric_name, value

      ticker      — sector ETF ticker (e.g. XLK, XLF)
      date        — ISO date string (e.g. 2024-01-15)
      metric_name — free-form label (e.g. put_call_ratio, unusual_premium_usd)
      value       — numeric value

    Rows with unknown tickers are skipped with a warning.
    Existing rows for the same (etf, date, metric_name) are updated in place.
    """
    import pandas as pd
    from sector_flow.database.session import get_session, init_db
    from sector_flow.database.repository import ETFRepository, FlowMetricRepository

    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:
        raise click.ClickException(f"Cannot read CSV: {exc}") from exc

    required = {"ticker", "date", "metric_name", "value"}
    missing = required - set(df.columns.str.strip().str.lower())
    if missing:
        raise click.ClickException(f"Missing required columns: {missing}")

    df.columns = df.columns.str.strip().str.lower()
    df["ticker"] = df["ticker"].str.upper().str.strip()

    try:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception as exc:
        raise click.ClickException(f"Cannot parse date column: {exc}") from exc

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    invalid = df["value"].isna().sum()
    if invalid:
        click.echo(f"Warning: {invalid} row(s) have non-numeric value — skipped", err=True)
    df = df.dropna(subset=["value"])

    if dry_run:
        click.echo(f"[dry-run] {len(df)} valid rows from {csv_file}")
        click.echo(df.groupby(["ticker", "metric_name"]).size().to_string())
        return

    init_db(db)
    saved = skipped = 0
    with get_session(db) as session:
        etf_repo = ETFRepository(session)
        flow_repo = FlowMetricRepository(session)

        for _, row in df.iterrows():
            etf = etf_repo.get_by_ticker(row["ticker"])
            if etf is None:
                click.echo(f"  skip: unknown ticker {row['ticker']!r}", err=True)
                skipped += 1
                continue
            flow_repo.save_metric(
                etf_id=etf.id,
                date=row["date"].to_pydatetime(),
                metric_name=str(row["metric_name"]).strip(),
                value=float(row["value"]),
            )
            saved += 1

        session.commit()

    click.echo(f"flow-import complete: {saved} rows saved, {skipped} skipped")


if __name__ == "__main__":
    main()
