from __future__ import annotations

from datetime import datetime, timedelta
from dateutil import tz


def to_datestr(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def us_trading_day_for(now: datetime) -> datetime:
    # Assume the run happens post-close; if weekend/holiday detection is needed, the
    # data provider grouped endpoint is authoritative. Here, just return today's date in TZ.
    # If run before 16:00 local, pick previous weekday.
    weekday = now.weekday()
    # Weekday: 0=Mon..4=Fri, 5=Sat,6=Sun
    if weekday >= 5:
        # Weekend → previous Friday
        delta = weekday - 4
        return (now - timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
    # If before 16:10 local, use previous business day
    if now.hour < 16 or (now.hour == 16 and now.minute < 10):
        prev = now - timedelta(days=1)
        if prev.weekday() == 6:  # Sun
            prev -= timedelta(days=2)
        elif prev.weekday() == 5:  # Sat
            prev -= timedelta(days=1)
        return prev.replace(hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

