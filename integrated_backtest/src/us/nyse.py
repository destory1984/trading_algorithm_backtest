"""NYSE full-day closures under the standard rules, copied from index_timing_backtest/src/live.py (special closures
such as days of mourning are not known in advance)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    """n-th weekday wd (0 = Monday) of the month; n = -1 is the last."""
    days = [date(y, m, d) for d in range(1, 32) if _valid(y, m, d) and date(y, m, d).weekday() == wd]
    return days[n]


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def nyse_holidays(y: int) -> set[date]:
    """NYSE full-day closures under the standard rules (special closures such as days of mourning are not known in
    advance; a missing bar on such a day shows up as "lag")."""
    out = {_easter(y) - timedelta(days=2), _nth_weekday(y, 1, 0, 2), _nth_weekday(y, 2, 0, 2),
           _nth_weekday(y, 5, 0, -1), _nth_weekday(y, 9, 0, 0), _nth_weekday(y, 11, 3, 3)}
    for m, d in ((1, 1), (6, 19), (7, 4), (12, 25)):
        if (m, d) == (6, 19) and y < 2022:  # Juneteenth closes the NYSE from 2022
            continue
        x = date(y, m, d)
        if x.weekday() == 5:
            if (m, d) == (1, 1):  # New Year's Day on a Saturday: no closure
                continue
            x -= timedelta(days=1)
        elif x.weekday() == 6:
            x += timedelta(days=1)
        out.add(x)
    return out


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    x = d + timedelta(days=1)
    while not is_trading_day(x):
        x += timedelta(days=1)
    return x


