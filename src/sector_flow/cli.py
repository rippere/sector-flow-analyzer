import click


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


if __name__ == "__main__":
    main()
