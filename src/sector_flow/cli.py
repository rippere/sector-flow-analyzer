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


if __name__ == "__main__":
    main()
