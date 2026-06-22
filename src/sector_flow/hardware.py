"""Embedded ambient signaling display for the Pi build (WS4).

Two layers:

* ``render_signal_panel`` — a **pure** function (snapshot -> PIL.Image) that draws
  the glanceable "Variant A" layout: a regime banner, a 3x4 grid of the 11 sectors
  coloured by momentum on a colour-blind-safe diverging palette, the dominant
  sector highlighted, a cohesion bar, and a LIVE/DELAYED/CLOSED/STALE line. It is
  fully testable without any hardware.
* ``SignalDisplay`` drivers — ``InkyDisplay`` (e-ink) and ``OledDisplay`` import
  their device libraries lazily so this module imports cleanly off-device;
  ``ConsoleDisplay`` is a no-hardware fallback that just saves the PNG.

``build_snapshot`` pulls the current numbers from the running API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Sector display order for the 3x4 grid (11 sectors + 1 spare cell).
SECTOR_ORDER = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                "XLI", "XLB", "XLRE", "XLU", "XLC"]

# Colour-blind-safe regime colours (Okabe-Ito / Tol family).
_REGIME_COLORS = {
    "risk_on": (0, 158, 115),    # bluish green
    "risk_off": (213, 94, 0),    # vermillion
    "rotation": (230, 159, 0),   # orange
    "crisis": (136, 34, 85),     # muted purple
    "neutral": (120, 120, 120),  # grey
}

# Diverging momentum endpoints: blue (−1) → light grey (0) → orange (+1).
_MOM_NEG = (0, 114, 178)     # blue
_MOM_MID = (235, 235, 235)   # near-white
_MOM_POS = (230, 159, 0)     # orange

_STATUS_COLORS = {
    "LIVE": (0, 158, 115),
    "DELAYED": (230, 159, 0),
    "CLOSED": (120, 120, 120),
    "STALE": (213, 94, 0),
}


@dataclass
class Snapshot:
    """Everything the panel needs for one render."""
    regime: str = "neutral"
    momentum: dict = field(default_factory=dict)   # ticker -> [-1, 1]
    dominant: Optional[str] = None                  # leading sector ticker
    cohesion: float = 0.0                           # [0, 1]
    market_status: str = "CLOSED"                   # LIVE/DELAYED/CLOSED/STALE
    as_of: str = ""                                 # short label, e.g. "15:47"


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def momentum_color(value: float) -> tuple:
    """Map momentum in [-1, 1] to a colour-blind-safe diverging colour."""
    v = max(-1.0, min(1.0, float(value)))
    if v < 0:
        return _lerp(_MOM_MID, _MOM_NEG, -v)
    return _lerp(_MOM_MID, _MOM_POS, v)


def _desaturate(color, amount):
    """Blend `color` toward grey by `amount` in [0, 1] (stale/closed dimming)."""
    grey = sum(color) / 3.0
    return tuple(int(round(c + (grey - c) * amount)) for c in color)


def render_signal_panel(snapshot: Snapshot, size: tuple = (800, 480)):
    """Render the Variant-A ambient panel. Returns a PIL.Image (RGB)."""
    from PIL import Image, ImageDraw

    w, h = size
    # Dim everything when the data isn't live (calm "this is not current" cue).
    dim = 0.45 if snapshot.market_status in ("STALE", "CLOSED") else 0.0

    img = Image.new("RGB", size, (16, 16, 20))
    draw = ImageDraw.Draw(img)

    # --- Regime banner (top ~18%) ---
    banner_h = int(h * 0.18)
    regime = snapshot.regime if snapshot.regime in _REGIME_COLORS else "neutral"
    banner_color = _desaturate(_REGIME_COLORS[regime], dim)
    draw.rectangle([0, 0, w, banner_h], fill=banner_color)
    draw.text((14, banner_h // 2 - 6), f"REGIME: {regime.upper()}", fill=(255, 255, 255))

    # --- Sector grid (3 cols x 4 rows) ---
    cols, rows = 3, 4
    pad = 8
    grid_top = banner_h + pad
    grid_bottom = int(h * 0.82)
    cell_w = (w - pad * (cols + 1)) // cols
    cell_h = (grid_bottom - grid_top - pad * (rows - 1)) // rows

    for idx, ticker in enumerate(SECTOR_ORDER):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = grid_top + r * (cell_h + pad)
        x1, y1 = x0 + cell_w, y0 + cell_h
        mom = float(snapshot.momentum.get(ticker, 0.0))
        fill = _desaturate(momentum_color(mom), dim)
        draw.rectangle([x0, y0, x1, y1], fill=fill)
        # Dominant sector gets a bright border.
        if ticker == snapshot.dominant:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=4)
        label_color = (20, 20, 20) if sum(fill) > 360 else (240, 240, 240)
        draw.text((x0 + 6, y0 + 6), ticker, fill=label_color)
        draw.text((x0 + 6, y1 - 16), f"{mom:+.2f}", fill=label_color)

    # --- Footer: cohesion bar + status line (bottom ~18%) ---
    foot_top = grid_bottom + pad
    bar_w = int((w - 2 * pad) * max(0.0, min(1.0, snapshot.cohesion)))
    draw.rectangle([pad, foot_top, w - pad, foot_top + 10], outline=(90, 90, 90))
    draw.rectangle([pad, foot_top, pad + bar_w, foot_top + 10], fill=(180, 180, 180))
    status = snapshot.market_status if snapshot.market_status in _STATUS_COLORS else "STALE"
    draw.text((pad, foot_top + 16),
              f"cohesion {snapshot.cohesion:.2f}", fill=(200, 200, 200))
    draw.text((w - 220, foot_top + 16),
              f"{status} - as of {snapshot.as_of or '--:--'}",
              fill=_STATUS_COLORS[status])
    return img


def build_snapshot(api_url: str = "http://localhost:8000") -> Snapshot:
    """Pull the current regime/momentum/cohesion/status from the running API."""
    import requests

    base = api_url.rstrip("/")
    regime = requests.get(f"{base}/analysis/regime", timeout=8).json()
    momentum = regime.get("sector_momentum", {})
    dominant = max(momentum, key=momentum.get) if momentum else None
    # market_status is broadcast on the websocket meta; the REST regime payload
    # carries cohesion. Default to CLOSED until a live status is available.
    return Snapshot(
        regime=regime.get("market_regime", "neutral"),
        momentum={k: float(v) for k, v in momentum.items()},
        dominant=dominant,
        cohesion=float(regime.get("cohesion", 0.0)),
        market_status=regime.get("market_status", "CLOSED"),
        as_of=regime.get("as_of", ""),
    )


class SignalDisplay:
    """Interface: render a Snapshot to a device."""

    def show(self, snapshot: Snapshot) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleDisplay(SignalDisplay):
    """No-hardware fallback — saves the panel PNG to a path (dev / CI)."""

    def __init__(self, out_path: str = "/tmp/sector_flow_panel.png"):
        self.out_path = out_path

    def show(self, snapshot: Snapshot) -> None:
        render_signal_panel(snapshot).save(self.out_path)


class InkyDisplay(SignalDisplay):  # pragma: no cover - requires e-ink hardware
    """Pimoroni Inky Impression e-ink driver (lazy import; Pi-only)."""

    def __init__(self):
        from inky.auto import auto

        self._inky = auto()

    def show(self, snapshot: Snapshot) -> None:
        img = render_signal_panel(snapshot, size=self._inky.resolution)
        self._inky.set_image(img)
        self._inky.show()


class OledDisplay(SignalDisplay):  # pragma: no cover - requires OLED hardware
    """SSD1306 OLED companion readout (lazy import; Pi-only)."""

    def __init__(self):
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306

        self._device = ssd1306(i2c(port=1, address=0x3C))

    def show(self, snapshot: Snapshot) -> None:
        from luma.core.render import canvas

        with canvas(self._device) as draw:
            draw.text((0, 0), f"{snapshot.regime.upper()}", fill="white")
            draw.text((0, 16), f"lead {snapshot.dominant or '-'}", fill="white")
            draw.text((0, 32), f"{snapshot.market_status} {snapshot.as_of}", fill="white")


def get_display(kind: str = "console") -> SignalDisplay:
    """Factory: 'inky' | 'oled' | 'console' (default)."""
    kind = (kind or "console").lower()
    if kind == "inky":
        return InkyDisplay()
    if kind == "oled":
        return OledDisplay()
    return ConsoleDisplay()
