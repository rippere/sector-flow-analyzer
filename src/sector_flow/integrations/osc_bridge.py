"""OSC Bridge — broadcasts sector flow analysis as OSC messages to VJ platforms.

Reads analysis snapshots from the FastAPI layer and sends UDP OSC packets
on a configurable interval. Compatible with nw_wrld and any OSC-aware software.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder


_SECTOR_NAMES: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

_REGIME_ENCODING: dict[str, int] = {
    "accumulation": 2,
    "breakout": 1,
    "neutral": 0,
    "distribution": -1,
    "breakdown": -2,
}


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class OSCBridge:
    """Bridge that polls the Sector Flow FastAPI and broadcasts OSC messages."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.osc_host = osc_host
        self.osc_port = osc_port
        self._client = udp_client.SimpleUDPClient(osc_host, osc_port)
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, address: str, *args: Any) -> None:
        """Send a single OSC message."""
        builder = OscMessageBuilder(address=address)
        for arg in args:
            builder.add_arg(arg)
        self._client.send(builder.build())

    def _get(self, path: str, timeout: int = 5) -> Any:
        """GET from the API; returns parsed JSON or None on error."""
        try:
            resp = requests.get(f"{self.api_url}{path}", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("API call failed for {}: {}", path, exc)
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def broadcast_snapshot(self) -> int:
        """
        Build and send one complete snapshot of OSC messages.

        Returns the number of messages sent (0 on API failure).
        """
        regime_data = self._get("/analysis/regime")
        if regime_data is None:
            logger.warning("API unreachable — skipping snapshot broadcast")
            return 0

        correlations = self._get("/analysis/correlations") or []
        sectors = self._get("/sectors") or []

        # Build quick-access maps from the sector list
        sector_map: dict[str, dict] = {}
        for s in sectors:
            sector_map[s["ticker"]] = s

        sector_momentum: dict[str, float] = regime_data.get("sector_momentum", {})
        sector_regimes: dict[str, str] = regime_data.get("sector_regimes", {})

        # Normalise |momentum| to [0,1] for weight
        abs_moms = [abs(v) for v in sector_momentum.values()] if sector_momentum else [0.0]
        max_abs_mom = max(abs_moms) if abs_moms else 1.0
        if max_abs_mom == 0.0:
            max_abs_mom = 1.0

        count = 0

        # --- per-sector messages ---
        for ticker, momentum in sector_momentum.items():
            weight = abs(momentum) / max_abs_mom  # normalised [0,1]
            regime = sector_regimes.get(ticker, "neutral")

            # Flow data from sector summary (latest net_inflow_usd if available)
            net_inflow = 0.0
            s_data = sector_map.get(ticker)
            if s_data is None:
                # Fetch directly for this ticker
                s_data = self._get(f"/sectors/{ticker}")
            # net_inflow_usd is not in the summary endpoint; we use 0.0 for now
            # (the /sectors/{ticker}/flows endpoint would need an extra call per sector)

            flow_in = max(0.0, net_inflow / 1_000_000)  # convert to USD millions
            flow_out = abs(min(0.0, net_inflow / 1_000_000))

            self._send(f"/sector/{ticker}/weight", float(weight))
            self._send(f"/sector/{ticker}/flow_in", float(flow_in))
            self._send(f"/sector/{ticker}/flow_out", float(flow_out))
            self._send(f"/sector/{ticker}/momentum", float(_clamp(momentum)))
            self._send(f"/sector/{ticker}/regime", str(regime))
            count += 5

        # --- pair messages ---
        for pair in correlations:
            ta = pair.get("ticker_a", "")
            tb = pair.get("ticker_b", "")
            corr = float(pair.get("correlation", 0.0))
            mom_a = sector_momentum.get(ta, 0.0)
            mom_b = sector_momentum.get(tb, 0.0)
            flow_dir = _clamp((mom_a - mom_b) * corr)

            self._send(f"/sector/pair/{ta}/{tb}/correlation", corr)
            self._send(f"/sector/pair/{ta}/{tb}/flow_direction", float(flow_dir))
            count += 2

        # --- meta messages ---
        market_regime: str = regime_data.get("market_regime", "neutral")
        cohesion: float = float(regime_data.get("cohesion", 0.0))

        # dominant sector: highest |momentum|
        dominant = max(sector_momentum, key=lambda t: abs(sector_momentum[t])) if sector_momentum else ""

        self._send("/meta/regime", market_regime)
        self._send("/meta/cohesion", cohesion)
        self._send("/meta/dominant_sector", dominant)
        self._send("/meta/timestamp", datetime.now(tz=timezone.utc).isoformat())
        count += 4

        logger.debug("Snapshot broadcast: {} messages sent", count)
        return count

    def run_forever(self, interval_seconds: int = 1) -> None:
        """Loop: broadcast_snapshot() every interval_seconds, plus a heartbeat."""
        logger.info(
            "OSC Bridge starting — API: {}  →  OSC: {}:{}  interval: {}s",
            self.api_url,
            self.osc_host,
            self.osc_port,
            interval_seconds,
        )
        while True:
            try:
                self.broadcast_snapshot()
            except Exception as exc:
                logger.warning("Unexpected error in broadcast_snapshot: {}", exc)

            # Heartbeat — always sent even when API is down
            try:
                self._send("/control/heartbeat", int(time.time()))
                staleness = int(time.time() - self._started_at)
                self._send("/control/data_staleness_seconds", staleness)
                self._send("/control/mode", "live")
            except Exception as exc:
                logger.warning("Heartbeat send error: {}", exc)

            time.sleep(interval_seconds)
