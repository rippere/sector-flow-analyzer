#!/usr/bin/env bash
# deploy/pi/display.sh — ZenScreen display configuration for Pi 5
#
# Run this after connecting the ZenScreen and logging into the desktop.
# It detects the display, confirms resolution, and writes it into
# /boot/firmware/config.txt so it persists across reboots.
#
# Usage:
#   bash deploy/pi/display.sh              # auto-detect
#   bash deploy/pi/display.sh HDMI-A-1     # specify output name

set -euo pipefail

TARGET="${1:-}"

echo "=== ZenScreen Display Setup ==="
echo ""

# ---------------------------------------------------------------------------
# Detect connected outputs via wlr-randr (Wayland) or xrandr (X11)
# ---------------------------------------------------------------------------
if command -v wlr-randr &>/dev/null; then
    echo "Detected Wayland session."
    echo ""
    echo "Connected outputs:"
    wlr-randr
    echo ""

    if [ -z "$TARGET" ]; then
        echo "Enter the output name for the ZenScreen (e.g. HDMI-A-1 or DP-1):"
        read -r TARGET
    fi

    echo "Current mode for $TARGET:"
    wlr-randr --output "$TARGET" 2>/dev/null | head -10

    echo ""
    echo "Setting $TARGET to preferred mode..."
    wlr-randr --output "$TARGET" --on --preferred
    echo "Done."

elif command -v xrandr &>/dev/null; then
    echo "Detected X11 session."
    echo ""
    echo "Connected outputs:"
    xrandr --query | grep " connected"
    echo ""

    if [ -z "$TARGET" ]; then
        echo "Enter the output name for the ZenScreen (e.g. HDMI-1 or HDMI-2):"
        read -r TARGET
    fi

    echo "Setting $TARGET to 1920x1080..."
    xrandr --output "$TARGET" --mode 1920x1080 --rate 60
    echo "Done."
else
    echo "Neither wlr-randr nor xrandr found."
    echo "Install with: sudo apt-get install wlr-randr"
    exit 1
fi

# ---------------------------------------------------------------------------
# Lock resolution in /boot/firmware/config.txt so it survives reboot
# ---------------------------------------------------------------------------
CONFIG="/boot/firmware/config.txt"

if [ ! -f "$CONFIG" ]; then
    echo ""
    echo "Note: $CONFIG not found — skipping boot config update."
    echo "Resolution was set for this session only."
    exit 0
fi

echo ""
echo "Writing display settings to $CONFIG..."

# Remove any existing hdmi_group/hdmi_mode/hdmi_force lines we may have added
sudo sed -i '/# sector-flow display/,/# end sector-flow display/d' "$CONFIG"

sudo tee -a "$CONFIG" > /dev/null << 'EOF'
# sector-flow display — ZenScreen 1080p
hdmi_group=1
hdmi_mode=16
hdmi_force_hotplug=1
# end sector-flow display
EOF

echo "Boot config updated. Resolution will lock to 1920x1080@60Hz on next reboot."
echo ""
echo "Reboot to confirm: sudo reboot"
