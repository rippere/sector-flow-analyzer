import click
from loguru import logger


@click.group()
def main():
    """Sector Flow Analyzer CLI."""
    pass


@main.command()
def status():
    """Show system status."""
    logger.info("Sector Flow Analyzer v0.1.0")
    click.echo("Status: operational")


if __name__ == "__main__":
    main()
