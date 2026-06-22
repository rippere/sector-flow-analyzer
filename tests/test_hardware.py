"""Tests for the embedded display renderer + drivers (no hardware required)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

from sector_flow.hardware import (
    ConsoleDisplay,
    SECTOR_ORDER,
    Snapshot,
    build_snapshot,
    momentum_color,
    render_signal_panel,
)


def _snap(**kw):
    base = dict(
        regime="risk_on",
        momentum={t: 0.0 for t in SECTOR_ORDER},
        dominant="XLK",
        cohesion=0.5,
        market_status="LIVE",
        as_of="15:47",
    )
    base.update(kw)
    return Snapshot(**base)


# --- momentum_color -------------------------------------------------------

def test_momentum_color_diverges():
    neg = momentum_color(-1.0)
    pos = momentum_color(1.0)
    mid = momentum_color(0.0)
    # negative leans blue (B>R), positive leans orange (R>B)
    assert neg[2] > neg[0]
    assert pos[0] > pos[2]
    # midpoint is the light neutral
    assert mid == (235, 235, 235)


def test_momentum_color_clamps_out_of_range():
    assert momentum_color(5.0) == momentum_color(1.0)
    assert momentum_color(-5.0) == momentum_color(-1.0)


# --- render_signal_panel --------------------------------------------------

def test_render_returns_image_of_requested_size():
    img = render_signal_panel(_snap(), size=(800, 480))
    assert img.size == (800, 480)
    assert img.mode == "RGB"


def test_render_handles_missing_momentum():
    img = render_signal_panel(_snap(momentum={}))  # no sector data
    assert img.size == (800, 480)


def test_stale_render_is_dimmer_than_live():
    live = render_signal_panel(_snap(market_status="LIVE"))
    stale = render_signal_panel(_snap(market_status="STALE"))
    # Desaturation pulls colours toward grey → different pixels overall.
    assert list(live.getdata()) != list(stale.getdata())


def test_render_supports_small_eink_resolution():
    img = render_signal_panel(_snap(), size=(600, 448))
    assert img.size == (600, 448)


# --- ConsoleDisplay -------------------------------------------------------

def test_console_display_writes_png():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "panel.png")
        ConsoleDisplay(out_path=path).show(_snap())
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# --- build_snapshot -------------------------------------------------------

def test_build_snapshot_from_api():
    payload = {
        "market_regime": "rotation",
        "cohesion": 0.42,
        "sector_momentum": {"XLK": 0.6, "XLE": -0.3, "XLF": 0.1},
        "market_status": "DELAYED",
    }
    resp = MagicMock()
    resp.json.return_value = payload
    with patch("requests.get", return_value=resp):
        snap = build_snapshot("http://localhost:8000")
    assert snap.regime == "rotation"
    assert snap.dominant == "XLK"   # highest momentum
    assert snap.cohesion == 0.42
    assert snap.market_status == "DELAYED"
