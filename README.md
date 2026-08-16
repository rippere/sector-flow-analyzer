# Sector Flow Analyzer

Market sector money-flow visualization and analysis tool. Tracks the 11 SPDR sector
ETFs (`XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLI`, `XLB`, `XLRE`, `XLU`, `XLC`),
computes inter-sector correlation structure and fund-flow tilt, and classifies the
current market regime (`crisis | risk_on | risk_off | rotation | neutral`).

See [`docs/MODEL.md`](docs/MODEL.md) for how the regime signal is computed and what
its validated performance actually covers.

## Features

- **Ingestion pipeline** — daily OHLCV via `yfinance` + SSGA fund-flow snapshots.
- **Analysis engine** — rolling inter-sector correlation, cohesion, adaptive regime
  thresholds, and flow-tilt z-scores (`src/sector_flow/analysis`).
- **FastAPI backend** — REST endpoints for sectors, analysis, pipeline status, plus a
  live WebSocket feed (`src/sector_flow/api`).
- **Dash dashboard** — interactive sector-rotation visualization.
- **OSC bridge** — broadcasts live flow data as OSC messages (e.g. to `nw_wrld`).

## Requirements

- Python >= 3.10
- [`uv`](https://github.com/astral-sh/uv) (or `pip`)

## Setup

```bash
git clone <repo-url>
cd sector-flow-analyzer
make install          # uv pip install -e ".[dev]"
cp .env.example .env  # edit DATABASE_URL etc. as needed
```

## Usage

The `sector-flow` CLI (`src/sector_flow/cli.py`) drives everything:

```bash
sector-flow backfill              # pull historical OHLCV for all 11 sectors
sector-flow ingest                # run the daily ingestion pipeline (OHLCV + SSGA flows)
sector-flow analyze                # compute correlations, classify regimes, persist results
sector-flow status                 # data freshness / row counts per sector
sector-flow show-regime            # current market + per-sector regime, momentum, cohesion
sector-flow show-flows             # net inflow ranking across sectors
sector-flow serve                  # start the FastAPI server (uvicorn)
sector-flow dashboard              # start the interactive Dash dashboard
sector-flow osc-bridge             # broadcast live flow data as OSC messages
```

Run `sector-flow <command> --help` for the full option list of any command.

With the API running (`sector-flow serve`), health and docs are at:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Development

```bash
make test        # pytest
make coverage     # pytest with HTML coverage report
make lint         # ruff check
make format       # black
make typecheck    # mypy
```

## Deployment

`deploy/` contains a Docker Compose stack (API, dashboard, OSC bridge, and cron-style
ingest/analyze jobs) and systemd unit files for a bare-metal install (`deploy/install.sh`).

```bash
docker compose -f deploy/docker-compose.yml up
```

## License

MIT
