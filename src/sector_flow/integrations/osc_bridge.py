"""OSC Bridge — broadcasts sector flow analysis as OSC messages to VJ platforms.

Reads objective flow data from the FastAPI /analysis/flows endpoint and sends
UDP OSC packets on a configurable interval. Compatible with nw_wrld and any
OSC-aware software. See osc_spec.yaml (v2.0) for the full address contract.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
from loguru import logger
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder


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

    def _send(self, address: str, *args: Any) -> None:
        builder = OscMessageBuilder(address=address)
        for arg in args:
            builder.add_arg(arg)
        self._client.send(builder.build())

    def _get(self, path: str, timeout: int = 5) -> Any:
        try:
            resp = requests.get(f"{self.api_url}{path}", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("API call failed for {}: {}", path, exc)
            return None

    def broadcast_snapshot(self) -> int:
        """
        Build and send one complete snapshot of OSC messages.

        Reads from /analysis/flows — objective data only (flows, momentum ranks,
        AUM, correlations). No regime signals. Returns message count sent (0 on
        API failure).

        Null handling:
          net_inflow_usd=None → flow_volume sent as 0.0, flow_direction as 0.0
          aum_usd=None        → aum sent as 0.0
          momentum_rank       → always present [0.0, 1.0]
        """
        flow_data = self._get("/analysis/flows")
        if flow_data is None:
            logger.warning("API unreachable — skipping snapshot broadcast")
            return 0

        flows_list: list[dict] = flow_data.get("flows", [])
        correlations: list[dict] = flow_data.get("correlations", [])

        flow_map: dict[str, dict] = {f["ticker"]: f for f in flows_list}

        # Cohesion = mean |correlation| across all pairs
        corr_vals = [abs(c.get("correlation", 0.0)) for c in correlations]
        avg_cohesion = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0

        # Dominant sector: largest absolute net_inflow_usd (0.0 if all null)
        dominant = max(
            flow_map,
            key=lambda t: abs(flow_map[t].get("net_inflow_usd") or 0.0),
            default="",
        )

        # Max absolute inflow for flow_direction normalization
        abs_flows = [abs(f.get("net_inflow_usd") or 0.0) for f in flows_list]
        max_abs_flow = max(abs_flows) if abs_flows else 1.0
        if max_abs_flow == 0.0:
            max_abs_flow = 1.0

        count = 0

        # --- Per-sector messages ---
        for ticker, f in flow_map.items():
            net_inflow = f.get("net_inflow_usd") or 0.0   # None → 0.0
            aum = f.get("aum_usd") or 0.0                 # None → 0.0
            mom_rank = float(f.get("momentum_rank") or 0.5)

            self._send(f"/sector/{ticker}/flow_volume", float(net_inflow))
            self._send(f"/sector/{ticker}/momentum_rank", mom_rank)
            self._send(f"/sector/{ticker}/aum", float(aum))
            count += 3

        # --- Pairwise messages ---
        for pair in correlations:
            ta = pair.get("ticker_a", "")
            tb = pair.get("ticker_b", "")
            corr = float(pair.get("correlation", 0.0))

            inflow_a = (flow_map.get(ta) or {}).get("net_inflow_usd") or 0.0
            inflow_b = (flow_map.get(tb) or {}).get("net_inflow_usd") or 0.0
            flow_dir = _clamp((inflow_a - inflow_b) / max_abs_flow * abs(corr))

            self._send(f"/sector/pair/{ta}/{tb}/correlation", corr)
            self._send(f"/sector/pair/{ta}/{tb}/flow_direction", float(flow_dir))
            count += 2

        # --- Meta messages ---
        self._send("/meta/cohesion", float(avg_cohesion))
        self._send("/meta/dominant_sector", dominant)
        self._send("/meta/timestamp", datetime.now(tz=timezone.utc).isoformat())
        count += 3

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

            try:
                self._send("/control/heartbeat", int(time.time()))
                staleness = int(time.time() - self._started_at)
                self._send("/control/data_staleness_seconds", staleness)
                self._send("/control/mode", "live")
            except Exception as exc:
                logger.warning("Heartbeat send error: {}", exc)

            time.sleep(interval_seconds)
