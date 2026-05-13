#!/usr/bin/env bash
# install.sh — wire sector-flow services into systemd user daemon
# Run once after cloning/setting up the project.
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR="$HOME/.config/systemd/user"
LOG_DIR="$HOME/.sector_flow/logs"
DB_DIR="$HOME/.sector_flow"
SERVICES=(
    sector-flow-api.service
    sector-flow-dashboard.service
    sector-flow-osc.service
    sector-flow-ingest.service
    sector-flow-ingest.timer
    sector-flow-analyze.service
    sector-flow-analyze.timer
)

echo "=== Sector Flow Analyzer — systemd install ==="
echo "Project: $PROJECT_DIR"
echo ""

# Create dirs
mkdir -p "$LOG_DIR" "$DB_DIR"
echo "[1/4] Created $LOG_DIR and $DB_DIR"

# Symlink service files
for svc in "${SERVICES[@]}"; do
    src="$PROJECT_DIR/deploy/systemd/$svc"
    dst="$SYSTEMD_DIR/$svc"
    if [ -L "$dst" ]; then
        rm "$dst"
    fi
    ln -s "$src" "$dst"
    echo "      → linked $svc"
done
echo "[2/4] Service files linked into $SYSTEMD_DIR"

# Reload daemon
systemctl --user daemon-reload
echo "[3/4] systemd daemon reloaded"

# Backfill if DB doesn't exist yet
DB_PATH="$DB_DIR/sector_flow.db"
if [ ! -f "$DB_PATH" ]; then
    echo "[4/4] No database found — running initial 90-day backfill..."
    DATABASE_URL="sqlite:///$DB_PATH" \
        "$PROJECT_DIR/.venv/bin/sector-flow" backfill --days 90
    DATABASE_URL="sqlite:///$DB_PATH" \
        "$PROJECT_DIR/.venv/bin/sector-flow" analyze
    echo "      Backfill complete."
else
    echo "[4/4] Database already exists at $DB_PATH — skipping backfill."
fi

echo ""
echo "=== Done. Enable services with: ==="
echo ""
echo "  # Persistent API + dashboard (start on login):"
echo "  systemctl --user enable --now sector-flow-api.service"
echo "  systemctl --user enable --now sector-flow-dashboard.service"
echo ""
echo "  # Optional: OSC bridge for nw_wrld (start when nw_wrld is running):"
echo "  systemctl --user enable --now sector-flow-osc.service"
echo ""
echo "  # Daily data ingestion timers:"
echo "  systemctl --user enable --now sector-flow-ingest.timer"
echo "  systemctl --user enable --now sector-flow-analyze.timer"
echo ""
echo "  Dashboard: http://localhost:8050"
echo "  API:       http://localhost:8000"
echo "  Metrics:   http://localhost:8000/metrics"
