#!/usr/bin/env bash
# deploy/pi/setup.sh — Raspberry Pi 5 kiosk setup for Sector Flow Analyzer
#
# Run once after cloning the project on the Pi:
#   cd ~/sector-flow-analyzer && bash deploy/pi/setup.sh
#
# What this does:
#   1. Installs system deps (chromium, unclutter, python3, curl)
#   2. Installs uv + creates Python venv
#   3. Enables systemd lingering so services start at boot (no login required)
#   4. Installs and enables systemd user services (API, ingest timer, analyze timer)
#   5. Runs 90-day backfill on first install
#   6. Writes XDG autostart entries for Chromium kiosk + cursor hide + sleep disable
#   7. Configures Wayfire to never blank the screen

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$PROJECT_DIR/.venv"
DATA_DIR="$HOME/.sector_flow"
LOG_DIR="$DATA_DIR/logs"
DB_PATH="$DATA_DIR/sector_flow.db"
SYSTEMD_USER="$HOME/.config/systemd/user"
AUTOSTART_DIR="$HOME/.config/autostart"
WAYFIRE_INI="$HOME/.config/wayfire.ini"

echo "=== Sector Flow Analyzer — Pi 5 Kiosk Setup ==="
echo "Project : $PROJECT_DIR"
echo "User    : $USER ($HOME)"
echo "DB      : $DB_PATH"
echo ""

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    chromium-browser \
    unclutter \
    python3 \
    python3-pip \
    curl \
    git \
    jq
echo "      done."

# ---------------------------------------------------------------------------
# 2. uv + Python venv
# ---------------------------------------------------------------------------
echo "[2/7] Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available immediately in this script
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "      uv $(uv --version)"

echo "      Creating Python venv and installing project..."
cd "$PROJECT_DIR"
uv venv --python 3.11
uv pip install -e "."
echo "      done."

# ---------------------------------------------------------------------------
# 3. Data dirs
# ---------------------------------------------------------------------------
echo "[3/7] Creating data directories..."
mkdir -p "$LOG_DIR"
echo "      $DATA_DIR + $LOG_DIR"

# ---------------------------------------------------------------------------
# 4. Systemd lingering (services run at boot without a user session)
# ---------------------------------------------------------------------------
echo "[4/7] Enabling systemd lingering for $USER..."
sudo loginctl enable-linger "$USER"
echo "      done."

# ---------------------------------------------------------------------------
# 5. Systemd services — substitute desktop paths with Pi paths
# ---------------------------------------------------------------------------
echo "[5/7] Installing systemd user services..."
mkdir -p "$SYSTEMD_USER"

SERVICES=(
    sector-flow-api.service
    sector-flow-ingest.service
    sector-flow-ingest.timer
    sector-flow-analyze.service
    sector-flow-analyze.timer
)

for svc in "${SERVICES[@]}"; do
    src="$PROJECT_DIR/deploy/systemd/$svc"
    dst="$SYSTEMD_USER/$svc"
    # Substitute hardcoded desktop paths with actual Pi paths
    sed \
        -e "s|/mnt/external/Projects/sector-flow-analyzer|$PROJECT_DIR|g" \
        -e "s|/home/rippere|$HOME|g" \
        "$src" > "$dst"
    echo "      → $svc"
done

systemctl --user daemon-reload

# Initial 90-day backfill on first install
if [ ! -f "$DB_PATH" ]; then
    echo "      No database found — running 90-day backfill (takes ~2 min)..."
    DATABASE_URL="sqlite:///$DB_PATH" "$VENV/bin/sector-flow" backfill --days 90
    DATABASE_URL="sqlite:///$DB_PATH" "$VENV/bin/sector-flow" analyze
    echo "      Backfill complete."
else
    echo "      Database already exists — skipping backfill."
fi

# Enable and start services
systemctl --user enable --now sector-flow-api.service
systemctl --user enable --now sector-flow-ingest.timer
systemctl --user enable --now sector-flow-analyze.timer
echo "      Services enabled."

# ---------------------------------------------------------------------------
# 6. XDG autostart — kiosk browser + cursor hide + sleep disable
# ---------------------------------------------------------------------------
echo "[6/7] Writing XDG autostart entries..."
mkdir -p "$AUTOSTART_DIR"

# Chromium kiosk — polls /health until API is ready, then opens full-screen
cat > "$AUTOSTART_DIR/sector-flow-kiosk.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=Sector Flow Kiosk
Comment=Wait for API then open dashboard in kiosk mode
Exec=bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 1; done; chromium-browser --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble --disable-restore-session-state --disable-translate --no-first-run --fast --fast-start --disable-features=TranslateUI --autoplay-policy=no-user-gesture-required http://localhost:8000/dashboard'
X-GNOME-Autostart-enabled=true
EOF

# Hide mouse cursor after 1 second of inactivity
cat > "$AUTOSTART_DIR/sector-flow-cursor.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=Hide Cursor
Exec=unclutter -idle 1 -root
X-GNOME-Autostart-enabled=true
EOF

echo "      Autostart entries written to $AUTOSTART_DIR"

# ---------------------------------------------------------------------------
# 7. Wayfire — disable screen blanking and idle lock
# ---------------------------------------------------------------------------
echo "[7/7] Configuring Wayfire to disable screen sleep..."

# Create wayfire.ini if it doesn't exist
touch "$WAYFIRE_INI"

# Remove any existing [idle] section and rewrite it
python3 - "$WAYFIRE_INI" << 'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as f:
    content = f.read()

# Remove existing [idle] block if present
content = re.sub(r'\[idle\][^\[]*', '', content, flags=re.DOTALL).strip()

# Append new [idle] block that disables blanking
content += "\n\n[idle]\ndpms_timeout = -1\nscreensaver_timeout = -1\n"

with open(path, 'w') as f:
    f.write(content)

print(f"      Updated {path}")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Services running:"
systemctl --user is-active sector-flow-api.service && echo "  ✓ API" || echo "  ✗ API (check: journalctl --user -u sector-flow-api)"
systemctl --user is-active sector-flow-ingest.timer && echo "  ✓ ingest timer" || echo "  ✗ ingest timer"
echo ""
echo "On next login / reboot: Chromium opens automatically in kiosk mode."
echo ""
echo "To launch kiosk manually right now:"
echo "  chromium-browser --kiosk http://localhost:8000/dashboard &"
echo ""
echo "Logs: $LOG_DIR"
echo "DB  : $DB_PATH"
