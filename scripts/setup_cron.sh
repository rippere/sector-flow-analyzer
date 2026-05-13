#!/usr/bin/env bash
# setup_cron.sh — Install daily sector-flow cron jobs
#
# Adds two crontab entries:
#   08:30 ET Mon-Fri  →  sector-flow ingest   (data pull)
#   08:35 ET Mon-Fri  →  sector-flow analyze  (analysis engine)
#
# Logs go to ~/.sector_flow/logs/cron.log
# Run this script once; it confirms before making any change.

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

# ET = UTC-5 (EST) / UTC-4 (EDT).
# 08:30 ET = 13:30 UTC (EST) or 12:30 UTC (EDT).
# We use 13:30 UTC as a safe default (covers EST; runs 30 min "late" during EDT).
INGEST_CRON="30 13 * * 1-5 ${SF_CMD} ingest >> ${LOG_FILE} 2>&1"
ANALYZE_CRON="35 13 * * 1-5 ${SF_CMD} analyze >> ${LOG_FILE} 2>&1"

echo ""
echo "=== Sector Flow Analyzer — Daily Cron Setup ==="
echo ""
echo "Log directory : ${LOG_DIR}"
echo "Log file      : ${LOG_FILE}"
echo "sector-flow   : ${SF_CMD}"
echo ""
echo "Entries to install:"
echo "  ${INGEST_CRON}"
echo "  ${ANALYZE_CRON}"
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
    (crontab -l 2>/dev/null | grep -vF "${SF_CMD} ingest" | grep -vF "${SF_CMD} analyze"; \
     echo "${INGEST_CRON}"; \
     echo "${ANALYZE_CRON}") | crontab -

    echo "Cron entries installed successfully."
    echo ""
    echo "Current crontab:"
    crontab -l
else
    echo "Aborted — no changes made."
fi
