"""US equity market calendar (XNYS) — trading-day and session awareness.

Wraps ``exchange_calendars`` (XNYS) so the rest of the codebase can ask the
questions it never could before: is *today* a trading day, is the market open
*right now*, when did the last session close, and how many **trading** days
separate two dates. This replaces the previous naive ``date.today()`` /
``calendar_days * 5/7`` approximations and makes scheduling DST-correct.

All wall-clock reasoning is done in UTC (what ``exchange_calendars`` returns);
helpers that surface a human time also expose the America/New_York view.
Half-days (e.g. the day after Thanksgiving, Christmas Eve) are handled
automatically because session close times come from the calendar itself.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Optional

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback, project targets 3.10+
    from backports.zoneinfo import ZoneInfo  # type: ignore

_CAL_NAME = "XNYS"
MARKET_TZ = ZoneInfo("America/New_York")

# Once data is older than this many trading days it is no longer "today's" view.
_STALE_TRADING_DAYS = 2


class MarketStatus(str, Enum):
    """Freshness state of the latest data relative to the market session."""

    LIVE = "LIVE"        # market open and data is from the current session
    DELAYED = "DELAYED"  # market open but data lags (e.g. EOD-only flows intraday)
    CLOSED = "CLOSED"    # market closed; data is current as of the last session
    STALE = "STALE"      # data is older than the last session — something is wrong


@lru_cache(maxsize=1)
def _calendar():
    import exchange_calendars as xcals

    return xcals.get_calendar(_CAL_NAME)


def _as_utc(dt: Optional[datetime]) -> datetime:
    """Coerce a datetime (naive assumed UTC) to a tz-aware UTC datetime."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_trading_day(d: date) -> bool:
    """True if ``d`` is a regular US equity trading session (not weekend/holiday)."""
    return bool(_calendar().is_session(pd.Timestamp(d.isoformat())))


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """True if the regular session is open at ``dt`` (defaults to now, UTC).

    Naive datetimes are assumed to be UTC.
    """
    minute = pd.Timestamp(_as_utc(dt))
    return bool(_calendar().is_open_on_minute(minute))


def previous_close(dt: Optional[datetime] = None) -> datetime:
    """The most recent session close at or before ``dt`` (tz-aware UTC)."""
    ts = pd.Timestamp(_as_utc(dt))
    return _calendar().previous_close(ts).to_pydatetime()


def session_bounds(d: date) -> Optional[tuple[datetime, datetime]]:
    """(open, close) tz-aware UTC datetimes for ``d``, or None if not a session."""
    if not is_trading_day(d):
        return None
    session = pd.Timestamp(d.isoformat())
    cal = _calendar()
    return (
        cal.session_open(session).to_pydatetime(),
        cal.session_close(session).to_pydatetime(),
    )


def trading_days_between(start: date, end: date) -> int:
    """Number of trading sessions in the half-open interval (start, end].

    0 if ``end <= start``. Used for true data-staleness in trading days rather
    than the old ``calendar_days * 5/7`` estimate.
    """
    if end <= start:
        return 0
    # sessions_in_range is inclusive of both endpoints; exclude `start` itself.
    sessions = _calendar().sessions_in_range(
        pd.Timestamp(start.isoformat()), pd.Timestamp(end.isoformat())
    )
    return int(sum(1 for s in sessions if s.date() > start))


def now_et(dt: Optional[datetime] = None) -> datetime:
    """``dt`` (or now) expressed in America/New_York."""
    return _as_utc(dt).astimezone(MARKET_TZ)


def market_status(
    last_data_dt: Optional[datetime],
    now: Optional[datetime] = None,
    *,
    intraday_capable: bool = False,
) -> MarketStatus:
    """Classify data freshness for display/staleness indicators.

    Parameters
    ----------
    last_data_dt : datetime | None
        Timestamp of the most recent data point (naive assumed UTC).
    now : datetime | None
        Reference time (defaults to now, UTC).
    intraday_capable : bool
        If True, "open + same-session data" is LIVE. If False (the default,
        reflecting EOD-only flow data), an open market with same-session data is
        reported DELAYED, since the figures are not truly intraday.
    """
    now_utc = _as_utc(now)
    open_now = is_market_open(now_utc)

    if last_data_dt is None:
        return MarketStatus.STALE

    last_utc = _as_utc(last_data_dt)
    last_session_close = previous_close(now_utc)
    # Trading-day age of the data relative to the last completed session.
    age_td = trading_days_between(last_utc.date(), now_utc.date())

    if age_td > _STALE_TRADING_DAYS:
        return MarketStatus.STALE

    if open_now:
        # Same calendar day as the open session → fresh enough to be live/delayed.
        if last_utc.date() == now_utc.date():
            return MarketStatus.LIVE if intraday_capable else MarketStatus.DELAYED
        return MarketStatus.DELAYED

    # Market closed: current iff data is from the last completed session.
    if last_utc >= last_session_close:
        return MarketStatus.CLOSED
    if age_td <= _STALE_TRADING_DAYS:
        return MarketStatus.CLOSED
    return MarketStatus.STALE
