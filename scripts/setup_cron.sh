#!/usr/bin/env bash
# setup_cron.sh — Install sector-flow cron jobs
#
# Adds crontab entries (times are America/New_York via CRON_TZ, so they stay
# correct across DST without manual adjustment):
#   08:30 ET Mon-Fri        →  sector-flow ingest    (SSGA + yfinance pull)
#   08:35 ET Mon-Fri        →  sector-flow analyze   (EOD analysis engine)
#   every 20m 09:40–15:40 ET →  sector-flow intraday (live price/signal refresh)
#
# The intraday job no-ops on its own when the market is closed.
# Logs go to ~/.sector_flow/logs/cron.log
# Run this script once; it confirms before making any change.
#
# NOTE: CRON_TZ is honored by Vixie/cronie (Linux). If your cron does not
# support CRON_TZ, prefer the systemd timers in deploy/systemd/ instead.

set -euo pipefail

LOG_DIR="${HOME}/.sector_flow/logs"
LOG_FILE="${LOG_DIR}/cron.log"

# Resolve the sector-flow binary — prefer the venv in the project dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
VENV_BIN="${PROJECT_DIR}/.venv/bin"

if [[ -x "${VENV_BIN}/sector-flow" ]]; then
    SF_CMD="${VENV_BIN}/sector-flow"
else
    SF_CMD="$(command -v sector-flow 2>/dev/null || echo sector-flow)"
fi

# Pin the timezone so the schedule is DST-correct year-round.
TZ_LINE="CRON_TZ=America/New_York"
INGEST_CRON="30 8 * * 1-5 ${SF_CMD} ingest >> ${LOG_FILE} 2>&1"
ANALYZE_CRON="35 8 * * 1-5 ${SF_CMD} analyze >> ${LOG_FILE} 2>&1"
# Every 20 min from 09:40 to 15:40 ET (within RTH); the command gates itself too.
INTRADAY_CRON="0,20,40 9-15 * * 1-5 ${SF_CMD} intraday >> ${LOG_FILE} 2>&1"

echo ""
echo "=== Sector Flow Analyzer — Cron Setup ==="
echo ""
echo "Log directory : ${LOG_DIR}"
echo "Log file      : ${LOG_FILE}"
echo "sector-flow   : ${SF_CMD}"
echo "Timezone      : America/New_York (via CRON_TZ)"
echo ""
echo "Entries to install:"
echo "  ${TZ_LINE}"
echo "  ${INGEST_CRON}"
echo "  ${ANALYZE_CRON}"
echo "  ${INTRADAY_CRON}"
echo ""

# Create log directory now (non-destructive)
mkdir -p "${LOG_DIR}"
echo "Log directory created (or already exists): ${LOG_DIR}"
echo ""

# Prompt for confirmation
read -r -p "Install? [y/N] " REPLY
echo ""

if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
    # Safely append without duplicating entries
    (crontab -l 2>/dev/null \
        | grep -vF "CRON_TZ=America/New_York" \
        | grep -vF "${SF_CMD} ingest" \
        | grep -vF "${SF_CMD} analyze" \
        | grep -vF "${SF_CMD} intraday"; \
     echo "${TZ_LINE}"; \
     echo "${INGEST_CRON}"; \
     echo "${ANALYZE_CRON}"; \
     echo "${INTRADAY_CRON}") | crontab -

    echo "Cron entries installed successfully."
    echo ""
    echo "Current crontab:"
    crontab -l
else
    echo "Aborted — no changes made."
fi
