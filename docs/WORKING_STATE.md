# Progress / Working State

Snapshot of the last known-working state and the data sources this project depends on.
Update this file whenever the pipeline, data sources, or deploy topology change materially.

## Last known working state (2026-09-01)

- **Tests**: `make test` → 303 passed, 95% statement coverage (`pytest --cov`).
- **CLI**: `sector-flow backfill|ingest|analyze|status|show-regime|show-flows|serve|dashboard|osc-bridge`
  all present and exercised by tests (`src/sector_flow/cli.py`).
- **API**: FastAPI app (`src/sector_flow/api/app.py`) with routers for sectors, analysis,
  pipeline status, and a live WebSocket feed. `/health` and `/docs` serve locally via
  `sector-flow serve`.
- **Dashboard**: Dash app (`src/sector_flow/visualizations/dashboard.py`) for interactive
  sector-rotation visualization; 84% covered (uncovered lines are mostly callback wiring
  exercised only via a running browser session, not pytest).
- **OSC bridge**: broadcasts live flow data as OSC messages (`src/sector_flow/integrations/osc_bridge.py`),
  100% covered.
- **Regime model**: see [`docs/MODEL.md`](docs/MODEL.md) for the full model card. Key caveat
  carried over from that doc — the adaptive-threshold / flow-tilt classifier
  (`classify_market_regime()`, `compute_adaptive_thresholds()`, `risk_tilt()`) is unit-tested
  but **not** the thing the walk-forward/cross-validation backtest numbers were measured
  against; `scripts/backtest_regime.py` validates a simpler baseline momentum classifier.
  Wiring the backtest to the adaptive classifier is still open.
- **Deploy**: `deploy/` has a Docker Compose stack (API, dashboard, OSC bridge, cron-style
  ingest/analyze jobs) and systemd units for a bare-metal install
  (`sector-flow-{api,dashboard,osc,ingest,analyze,alert}.service` +
  `sector-flow-{ingest,analyze,alert}.timer`), installed via `deploy/install.sh`.
  `deploy/sector-flow-alert.py` sends a daily Alfred-inbox notification on regime/breakout
  signals (scheduled via `sector-flow-alert.timer`, 07:00 local).

## Data sources

| Source | Collector | What it provides | Auth / cost | Notes |
|---|---|---|---|---|
| `yfinance` | `src/sector_flow/collectors/yfinance_collector.py` | Daily OHLCV per sector ETF | Free, no API key | Backs `sector-flow backfill`/`ingest`; also the source for the validated backtest numbers in `docs/MODEL.md`. |
| SSGA daily Excel snapshot | `src/sector_flow/collectors/ssga_collector.py` | `shares_outstanding`, `aum_usd`, `nav`, `close` per SPDR ETF, used to derive net fund inflows (`net_inflow_usd = (shares_out_today − shares_out_yesterday) × nav_yesterday`) | Free, no API key, ToS permits personal/research use | Snapshot-only (today's data — historical range params are ignored); updated ~8:15am ET; pipeline layer computes the day-over-day delta, not the collector. Column formats are hand-parsed (`_parse_ssga_value`) and documented as of 2026-05 in the module docstring — a format change on State Street's end is the most likely silent-breakage point. |
| VIX / SPY (Yahoo, via `yfinance`) | backtest cross-validation path | Independent benchmark series used only to cross-validate the regime signal, not as model input | Free, no API key | Not part of the live ingest/analyze pipeline — used by `scripts/backtest_regime.py` only. |

**Universe**: the 11 SPDR sector ETFs — `XLK XLF XLE XLV XLY XLP XLI XLB XLRE XLU XLC`.
`XLRE` (listed 2015) and `XLC` (listed 2018) post-date the other nine (1998), so any
extended-history analysis only starts once all 11 have data.

**Storage**: SQLite via `DATABASE_URL` (`.env.example`) — `./dev.db` for local dev,
`/home/rippere/.sector_flow/sector_flow.db` for the systemd-managed production install, or
a Docker-mounted volume path when run via `docker-compose.yml`.
