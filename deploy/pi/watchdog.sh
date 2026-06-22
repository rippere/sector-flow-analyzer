#!/usr/bin/env bash
# deploy/pi/watchdog.sh — keep the kiosk healthy (WS3)
#
# Run every minute by sector-flow-watchdog.timer. If the API health endpoint is
# unreachable, restart the API service; if Chromium (the kiosk) has died, relaunch
# the autostart entry. Idempotent and quiet on the happy path.

set -uo pipefail

HEALTH_URL="http://localhost:8000/health"
LOG="${HOME}/.sector_flow/logs/watchdog.log"
mkdir -p "$(dirname "${LOG}")"

log() { echo "$(date -Is) $*" >> "${LOG}"; }

# 1. API health — restart the service if the endpoint does not answer 200.
if ! curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1; then
    log "API health check failed → restarting sector-flow-api.service"
    systemctl --user restart sector-flow-api.service || log "  restart failed"
    # Give it a moment before the kiosk check below.
    sleep 5
fi

# 2. Kiosk process — relaunch Chromium if it is not running.
if ! pgrep -f "chromium.*--kiosk" >/dev/null 2>&1; then
    log "kiosk browser not running → relaunching"
    AUTOSTART="${HOME}/.config/autostart/sector-flow-kiosk.desktop"
    if [[ -f "${AUTOSTART}" ]]; then
        # Re-run the Exec line from the autostart entry.
        EXEC_LINE="$(grep -m1 '^Exec=' "${AUTOSTART}" | cut -d= -f2-)"
        nohup bash -c "${EXEC_LINE}" >/dev/null 2>&1 &
    else
        log "  no autostart entry at ${AUTOSTART} — run deploy/pi/setup.sh"
    fi
fi
