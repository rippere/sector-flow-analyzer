#!/usr/bin/env bash
# deploy/pi/hardware.sh — provision the embedded signal display on a Pi 5 (WS4)
#
# Installs the e-ink / OLED Python libraries (into the project venv) and enables
# the SPI/I2C buses they need. Run once on the Pi after deploy/pi/setup.sh:
#   cd ~/sector-flow-analyzer && bash deploy/pi/hardware.sh
#
# The renderer (sector_flow.hardware.render_signal_panel) needs only Pillow,
# which the `hardware` extra already pulls in; inky/luma are device-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== Sector Flow — embedded display provisioning ==="

# 1. Display driver libraries (Pi-only wheels: inky pulls RPi.GPIO/spidev).
echo "Installing display libraries into the venv..."
(cd "${PROJECT_DIR}" && uv pip install "inky[rpi]" luma.oled gpiozero)

# 2. Enable SPI + I2C (Inky uses SPI; SSD1306 OLED uses I2C).
CONFIG=/boot/firmware/config.txt
if [[ -f "${CONFIG}" ]]; then
    for line in "dtparam=spi=on" "dtparam=i2c_arm=on"; do
        if ! grep -qxF "${line}" "${CONFIG}"; then
            echo "${line}" | sudo tee -a "${CONFIG}" >/dev/null
            echo "  added: ${line}"
        else
            echo "  present: ${line}"
        fi
    done
else
    echo "WARN: ${CONFIG} not found — enable SPI/I2C manually via raspi-config."
fi

echo ""
echo "Done. Test (no hardware needed) with:"
echo "  ${PROJECT_DIR}/.venv/bin/sector-flow display --device console"
echo "On hardware:"
echo "  ${PROJECT_DIR}/.venv/bin/sector-flow display --device inky"
echo "Then enable the periodic refresh:"
echo "  systemctl --user enable --now sector-flow-display.timer"
echo "(A reboot may be required for SPI/I2C to take effect.)"
