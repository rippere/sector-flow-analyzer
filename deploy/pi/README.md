# Raspberry Pi 5 deployment

Two physical surfaces share one Pi:

- **Kiosk** — Chromium full-screen on an HDMI display, showing the live dashboard
  (the full-colour "Variant B").
- **Embedded panel** — a Pimoroni Inky e-ink display (optional SSD1306 OLED
  companion) showing the glanceable "Variant A" signal panel.

## Bring-up order

```bash
cd ~/sector-flow-analyzer

# 1. Base: deps, venv, systemd lingering, API + ingest/analyze/intraday timers,
#    Chromium kiosk autostart, screen-blank disable, 1080p HDMI lock.
bash deploy/pi/setup.sh
bash deploy/pi/display.sh          # HDMI (ZenScreen) resolution lock

# 2. Reliability: API/kiosk watchdog (restarts on failure, every minute).
systemctl --user enable --now sector-flow-watchdog.timer

# 3. Embedded display (optional — needs the e-ink/OLED hardware):
bash deploy/pi/hardware.sh         # install inky/luma, enable SPI/I2C, then reboot
sector-flow display --device console   # smoke test (writes a PNG, no hardware)
systemctl --user enable --now sector-flow-display.timer
```

## What keeps it alive

| Unit | Cadence | Role |
|---|---|---|
| `sector-flow-api.service` | always | REST + WebSocket (kiosk + panel read from it) |
| `sector-flow-ingest/analyze.timer` | 08:30 / 08:35 ET | daily data + EOD analysis |
| `sector-flow-intraday.timer` | every 20 min, RTH | live price/signal refresh |
| `sector-flow-display.timer` | every 20 min, RTH | repaint the e-ink panel |
| `sector-flow-watchdog.timer` | every minute | restart API / relaunch kiosk on failure |

All schedules use `America/New_York`, so they are DST-correct without edits.

## Staleness behaviour

The dashboard and e-ink panel both surface a **LIVE / DELAYED / CLOSED / STALE**
state derived from the trading calendar (`sector_flow.market_calendar`). On a
network loss the watchdog restarts the API; until fresh data returns, the panel
renders dimmed (desaturated) with the `CLOSED`/`STALE` label rather than going
blank — so a stale display is never mistaken for a live one.

## Troubleshooting

- `journalctl --user -u sector-flow-api -f` — API logs.
- `~/.sector_flow/logs/watchdog.log` — what the watchdog restarted and when.
- Panel blank? `sector-flow display --device console` renders to a PNG you can
  inspect over SSH to isolate render vs hardware issues.
