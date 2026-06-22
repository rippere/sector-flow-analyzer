"""Tests for sector_flow.market_calendar — XNYS trading-day / session logic.

Deterministic: every assertion pins explicit dates / `now` values so results do
not depend on when the suite runs. 2026 holidays used as anchors:
  - 2026-06-19  Juneteenth (closed)
  - 2026-07-03  observed Independence Day (Jul 4 is a Saturday)
  - 2026-12-25  Christmas (closed)
  - 2026-11-27  day after Thanksgiving (early close 13:00 ET)
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sector_flow import market_calendar as mc
from sector_flow.market_calendar import MarketStatus


def _utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# --- is_trading_day -------------------------------------------------------

def test_regular_weekday_is_trading_day():
    assert mc.is_trading_day(date(2026, 6, 18)) is True  # Thursday


def test_weekend_is_not_trading_day():
    assert mc.is_trading_day(date(2026, 6, 20)) is False  # Saturday
    assert mc.is_trading_day(date(2026, 6, 21)) is False  # Sunday


def test_juneteenth_is_not_trading_day():
    assert mc.is_trading_day(date(2026, 6, 19)) is False


def test_observed_independence_day_is_not_trading_day():
    assert mc.is_trading_day(date(2026, 7, 3)) is False


def test_christmas_is_not_trading_day():
    assert mc.is_trading_day(date(2026, 12, 25)) is False


# --- is_market_open -------------------------------------------------------

def test_market_open_during_rth():
    # 14:30 UTC = 10:30 ET (EDT) on a June session → open
    assert mc.is_market_open(_utc(2026, 6, 18, 14, 30)) is True


def test_market_closed_after_hours():
    # 22:00 UTC = 18:00 ET → closed
    assert mc.is_market_open(_utc(2026, 6, 18, 22, 0)) is False


def test_market_closed_on_holiday():
    assert mc.is_market_open(_utc(2026, 6, 19, 15, 0)) is False


# --- DST correctness ------------------------------------------------------

def test_session_open_shifts_with_dst():
    # Winter (EST, UTC-5): 09:30 ET = 14:30 UTC
    jan_open, _ = mc.session_bounds(date(2026, 1, 5))
    assert (jan_open.hour, jan_open.minute) == (14, 30)
    # Summer (EDT, UTC-4): 09:30 ET = 13:30 UTC
    jun_open, _ = mc.session_bounds(date(2026, 6, 18))
    assert (jun_open.hour, jun_open.minute) == (13, 30)


def test_half_day_early_close():
    # Day after Thanksgiving 2026 closes 13:00 ET = 18:00 UTC (EST).
    bounds = mc.session_bounds(date(2026, 11, 27))
    assert bounds is not None
    _, close = bounds
    assert (close.hour, close.minute) == (18, 0)


def test_session_bounds_none_on_holiday():
    assert mc.session_bounds(date(2026, 6, 19)) is None


# --- trading_days_between -------------------------------------------------

def test_trading_days_between_excludes_holiday_and_weekend():
    # 6/15 Mon → 6/22 Mon: sessions 16,17,18,22 (19 Juneteenth + 20/21 weekend off)
    assert mc.trading_days_between(date(2026, 6, 15), date(2026, 6, 22)) == 4


def test_trading_days_between_zero_when_not_advanced():
    assert mc.trading_days_between(date(2026, 6, 18), date(2026, 6, 18)) == 0
    assert mc.trading_days_between(date(2026, 6, 18), date(2026, 6, 17)) == 0


# --- previous_close -------------------------------------------------------

def test_previous_close_skips_holiday():
    # On Saturday 6/20, the last close is Thursday 6/18 (Fri 6/19 was Juneteenth).
    pc = mc.previous_close(_utc(2026, 6, 20, 12, 0))
    assert pc.date() == date(2026, 6, 18)


# --- market_status --------------------------------------------------------

def test_status_live_when_open_and_intraday_capable():
    now = _utc(2026, 6, 18, 15, 0)            # market open
    data = _utc(2026, 6, 18, 14, 40)          # same-session data
    assert mc.market_status(data, now=now, intraday_capable=True) == MarketStatus.LIVE


def test_status_delayed_when_open_but_eod_only():
    now = _utc(2026, 6, 18, 15, 0)
    data = _utc(2026, 6, 18, 0, 0)
    assert mc.market_status(data, now=now, intraday_capable=False) == MarketStatus.DELAYED


def test_status_closed_when_market_shut_with_recent_data():
    now = _utc(2026, 6, 20, 12, 0)            # Saturday, closed
    data = _utc(2026, 6, 18, 19, 0)           # last session's data
    assert mc.market_status(data, now=now) == MarketStatus.CLOSED


def test_status_stale_when_data_old():
    now = _utc(2026, 6, 18, 12, 0)
    data = _utc(2026, 6, 8, 19, 0)            # ~7 sessions back
    assert mc.market_status(data, now=now) == MarketStatus.STALE


def test_status_stale_when_no_data():
    assert mc.market_status(None, now=_utc(2026, 6, 18, 12, 0)) == MarketStatus.STALE
