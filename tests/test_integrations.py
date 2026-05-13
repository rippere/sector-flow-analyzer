"""
Phase 5 integration tests — OSC bridge, Alfred query CLI.

All external calls are mocked. The API layer tests use the same in-memory
SQLite setup as the Phase 3 test suite.

Note: Prometheus /metrics endpoint was removed (Step 6) — unscraped in a
single-user project, not worth the maintenance surface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sector_flow.database.models import SECTOR_ETFS


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_api.py helpers)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Deliverable 1: OSC Bridge
# ---------------------------------------------------------------------------

_FAKE_REGIME = {
    "market_regime": "rotation",
    "cohesion": 0.56,
    "significant_pairs": 27,
    "total_pairs": 55,
    "sector_regimes": {
        "XLK": "accumulation",
        "XLF": "neutral",
        "XLE": "distribution",
        "XLV": "accumulation",
        "XLY": "distribution",
        "XLP": "breakout",
        "XLI": "neutral",
        "XLB": "neutral",
        "XLRE": "neutral",
        "XLU": "neutral",
        "XLC": "neutral",
    },
    "sector_momentum": {
        "XLK": 1.0,
        "XLF": 0.1,
        "XLE": -0.5,
        "XLV": 1.0,
        "XLY": -0.38,
        "XLP": 1.0,
        "XLI": 0.05,
        "XLB": 0.0,
        "XLRE": 0.0,
        "XLU": -0.98,
        "XLC": -0.63,
    },
    "computed_at": "2026-05-13T10:00:00",
}

_FAKE_CORRELATIONS = [
    {"ticker_a": "XLK", "ticker_b": "XLF", "correlation": 0.82, "covariance": 0.001, "p_value": None, "window_days": 30, "significant": True},
    {"ticker_a": "XLI", "ticker_b": "XLB", "correlation": 0.79, "covariance": 0.001, "p_value": None, "window_days": 30, "significant": True},
]

_FAKE_SECTORS = [
    {"ticker": t, "sector_name": n, "regime": None, "momentum": None, "cohesion": None,
     "latest_close": None, "latest_date": None, "row_count": 0}
    for t, n, _ in SECTOR_ETFS
]


def _make_mock_response(data):
    """Build a minimal mock requests.Response-like object."""
    mock = MagicMock()
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


def test_osc_bridge_broadcast_builds_messages():
    """OSCBridge.broadcast_snapshot() should call UDP client and return > 0 messages."""
    from sector_flow.integrations.osc_bridge import OSCBridge

    def fake_get(url, timeout=5):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        if "/sectors" in url and "/sectors/" not in url:
            return _make_mock_response(_FAKE_SECTORS)
        return _make_mock_response({})

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=fake_get):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient") as mock_client_cls:
            mock_sock = MagicMock()
            mock_client_cls.return_value = mock_sock

            bridge = OSCBridge(api_url="http://localhost:8000", osc_host="127.0.0.1", osc_port=9000)
            count = bridge.broadcast_snapshot()

    # Should send at least 1 message per sector (5 each) + meta messages
    assert count > 0, f"Expected > 0 messages, got {count}"
    assert count >= 11 * 5, f"Expected at least {11 * 5} sector messages, got {count}"


def test_osc_bridge_returns_zero_on_api_failure():
    """When the API is unreachable, broadcast_snapshot() returns 0."""
    from sector_flow.integrations.osc_bridge import OSCBridge
    import requests as req_mod

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=req_mod.ConnectionError("down")):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient"):
            bridge = OSCBridge()
            count = bridge.broadcast_snapshot()

    assert count == 0


def test_osc_bridge_sends_expected_addresses():
    """Verify that sector weight and meta regime addresses are sent."""
    from sector_flow.integrations.osc_bridge import OSCBridge

    sent_addresses: list[str] = []

    def fake_get(url, timeout=5):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response(_FAKE_SECTORS)

    with patch("sector_flow.integrations.osc_bridge.requests.get", side_effect=fake_get):
        with patch("sector_flow.integrations.osc_bridge.udp_client.SimpleUDPClient"):
            bridge = OSCBridge()
            # Intercept _send to capture addresses
            original_send = bridge._send

            def capturing_send(address, *args):
                sent_addresses.append(address)
                return original_send(address, *args)

            bridge._send = capturing_send
            bridge.broadcast_snapshot()

    assert "/sector/XLK/weight" in sent_addresses
    assert "/meta/regime" in sent_addresses
    assert "/meta/cohesion" in sent_addresses
    assert "/meta/dominant_sector" in sent_addresses
    # Pair messages
    assert "/sector/pair/XLK/XLF/correlation" in sent_addresses


# ---------------------------------------------------------------------------
# Deliverable 2 (removed): Prometheus /metrics endpoint
# The /metrics endpoint was removed in Step 6 — unscraped, single-user
# project, not worth the maintenance surface.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deliverable 3: Alfred query CLI command
# ---------------------------------------------------------------------------


def test_query_command_output_rotation_thesis():
    """sector-flow query should output 'Rotation Thesis' section."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query", "what is the current sector rotation thesis"])

    assert result.exit_code == 0, f"CLI exited with {result.exit_code}: {result.output}"
    assert "Rotation Thesis" in result.output


def test_query_command_shows_market_regime():
    """Output should mention the market regime from the API."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query"])

    assert "ROTATION" in result.output


def test_query_command_api_unreachable():
    """When API is down, output should contain an error message, not crash."""
    from sector_flow.cli import main
    import requests as req_mod

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=req_mod.ConnectionError("refused")):
        result = runner.invoke(main, ["query"])

    assert result.exit_code == 0
    assert "ERROR" in result.output or "error" in result.output.lower()


def test_query_command_shows_momentum_leaders():
    """Output should list at least one momentum leader."""
    from sector_flow.cli import main

    def fake_get(url, timeout=8):
        if "/analysis/regime" in url:
            return _make_mock_response(_FAKE_REGIME)
        if "/analysis/correlations" in url:
            return _make_mock_response(_FAKE_CORRELATIONS)
        return _make_mock_response({})

    runner = CliRunner()
    with patch("sector_flow.cli.requests.get", side_effect=fake_get):
        result = runner.invoke(main, ["query"])

    assert "Momentum Leaders" in result.output
    assert "XLK" in result.output
