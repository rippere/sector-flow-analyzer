#!/usr/bin/env python3
"""
sector-flow-alert: Check today's signals and write to Alfred inbox if actionable.
Runs daily after sector-flow-analyze. Silent when no signals; writes only when something
meaningful is detected (breakout/breakdown, market regime change, high cohesion).
"""
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

DB_PATH = Path("/home/rippere/.sector_flow/sector_flow.db")
ALFRED_INBOX = Path("/mnt/external/obsidian-vault/inbox")

REGIME_MAP = {
    2.0: "breakout",
    1.0: "accumulation",
    0.0: "neutral",
    -1.0: "distribution",
    -2.0: "breakdown",
}
SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities", "XLC": "Communication Services",
    "SPY": "S&P 500", "GLD": "Gold", "TLT": "20yr Treasuries", "SPEM": "Emerging Markets",
}
MACRO_TICKERS = {"SPY", "GLD", "TLT", "SPEM"}
RISK_ON_TICKERS = {"XLK", "XLF", "XLY"}
RISK_OFF_TICKERS = {"XLU", "XLP", "XLV"}


def query_today(conn: sqlite3.Connection, today: str) -> dict:
    cur = conn.execute("""
        SELECT e.ticker, fm.metric_name, fm.value
        FROM flow_metrics fm
        JOIN sector_etfs e ON fm.etf_id = e.id
        WHERE DATE(fm.date) = ?
    """, (today,))
    metrics: dict[str, dict] = {}
    for ticker, metric_name, value in cur.fetchall():
        if ticker not in metrics:
            metrics[ticker] = {}
        metrics[ticker][metric_name] = value

    cur = conn.execute("""
        SELECT e.ticker, pd.net_inflow_usd
        FROM price_data pd
        JOIN sector_etfs e ON pd.etf_id = e.id
        WHERE DATE(pd.date) = ? AND pd.net_inflow_usd IS NOT NULL
    """, (today,))
    for ticker, net_inflow in cur.fetchall():
        if ticker not in metrics:
            metrics[ticker] = {}
        metrics[ticker]["net_inflow_usd"] = net_inflow

    return metrics


def detect_market_regime(metrics: dict) -> str:
    cohesion = next(
        (v.get("cohesion", 0.0) for v in metrics.values() if "cohesion" in v), 0.0
    )
    if cohesion > 0.80:
        return "crisis"

    inflows = {
        t: d.get("net_inflow_usd", 0.0)
        for t, d in metrics.items()
        if t not in MACRO_TICKERS and d.get("net_inflow_usd") is not None
    }
    if not inflows:
        return "neutral"

    top3 = set(sorted(inflows, key=lambda t: inflows[t], reverse=True)[:3])
    if RISK_ON_TICKERS <= top3:
        return "risk_on"
    if RISK_OFF_TICKERS <= top3:
        return "risk_off"

    non_neutral = [
        t for t, d in metrics.items()
        if t not in MACRO_TICKERS and REGIME_MAP.get(d.get("regime_label", 0.0), "neutral") != "neutral"
    ]
    if len(non_neutral) >= 4:
        return "rotation"

    return "neutral"


def build_signals(metrics: dict, market_regime: str) -> list[str]:
    signals = []
    cohesion = next(
        (v.get("cohesion", 0.0) for v in metrics.values() if "cohesion" in v), 0.0
    )

    if market_regime in ("crisis", "risk_on", "risk_off", "rotation"):
        signals.append(f"Market regime: **{market_regime.upper()}** (cohesion: {cohesion:.2f})")
    elif cohesion > 0.75:
        signals.append(f"High cohesion: {cohesion:.2f} — macro risk building")

    for ticker in sorted(metrics):
        if ticker in MACRO_TICKERS:
            continue
        data = metrics[ticker]
        regime = REGIME_MAP.get(data.get("regime_label", 0.0), "neutral")
        momentum = data.get("momentum", 0.0)
        inflow = data.get("net_inflow_usd")
        name = SECTOR_NAMES.get(ticker, ticker)

        if regime in ("breakout", "breakdown"):
            inflow_str = f" · inflow ${inflow / 1e6:+.0f}M" if inflow is not None else ""
            signals.append(f"{ticker} ({name}): **{regime}** · momentum {momentum:+.2f}{inflow_str}")
        elif abs(momentum) > 0.80 and regime == "accumulation":
            signals.append(f"{ticker} ({name}): strong accumulation · momentum {momentum:+.2f}")

    return signals


def write_alert(signals: list[str], metrics: dict, market_regime: str) -> Path:
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M PDT")
    cohesion = next(
        (v.get("cohesion", 0.0) for v in metrics.values() if "cohesion" in v), 0.0
    )

    inflows = [
        (t, d.get("net_inflow_usd", 0.0), REGIME_MAP.get(d.get("regime_label", 0.0), "neutral"), d.get("momentum", 0.0))
        for t, d in metrics.items()
        if t not in MACRO_TICKERS and d.get("net_inflow_usd") is not None
    ]
    inflows.sort(key=lambda x: x[1], reverse=True)

    lines = [
        f"# SSGA Signal Alert — {today}",
        f"",
        f"<!-- alfred:source ssga-agent -->",
        f"",
        f"**Generated:** {now}  |  Market regime: **{market_regime.upper()}**  |  Cohesion: {cohesion:.2f}",
        f"",
        f"## Signals Detected",
        f"",
    ]
    for s in signals:
        lines.append(f"- {s}")

    if inflows:
        lines += [
            f"",
            f"## Sector Flow Ranking",
            f"",
            f"| # | Ticker | Net Inflow | Regime | Momentum |",
            f"|---|--------|-----------|--------|----------|",
        ]
        for i, (ticker, inflow, regime, mom) in enumerate(inflows, 1):
            lines.append(f"| {i} | {ticker} | ${inflow / 1e6:+.0f}M | {regime} | {mom:+.2f} |")

    out = ALFRED_INBOX / f"ssga-signal-{today}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        metrics = query_today(conn, today)
    finally:
        conn.close()

    if not metrics:
        print(f"[ssga-alert] No metrics for {today} — analyze may not have run yet")
        sys.exit(0)

    market_regime = detect_market_regime(metrics)
    signals = build_signals(metrics, market_regime)

    if signals:
        out = write_alert(signals, metrics, market_regime)
        print(f"[ssga-alert] {len(signals)} signal(s) detected → {out}")
    else:
        print(f"[ssga-alert] No actionable signals for {today} (regime: {market_regime}, cohesion: {next((v.get('cohesion', 0.0) for v in metrics.values() if 'cohesion' in v), 0.0):.2f})")


if __name__ == "__main__":
    main()
