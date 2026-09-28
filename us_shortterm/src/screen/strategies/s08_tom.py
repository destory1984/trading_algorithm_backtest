"""S8 turn of the month: buy on the month's second-to-last trading day, exit on the month's third trading day.
Trading days are the SPX calendar (krxbt.us.load_calendar). Close version only."""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, NAME, KIND, VERSIONS = "S8", "월말월초", "engine", ("close",)


def month_positions(cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """(0-based position from the month's first trading day, from its last trading day)."""
    ym = pd.Series(cal.year * 100 + cal.month, index=cal)
    first = ym.groupby(ym).cumcount()
    last = ym.groupby(ym).cumcount(ascending=False)
    return first, last


def signals(f, cal):
    first, last = month_positions(cal)
    cand = (last == 1).reindex(f.index, fill_value=False)
    target = (first == 2).reindex(f.index, fill_value=False)
    return cand.to_numpy(np.bool_), target.to_numpy(np.bool_)
