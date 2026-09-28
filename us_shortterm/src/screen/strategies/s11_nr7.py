"""S11 NR7 breakout: if range[t-1] < min(range[t-7..t-2]) (yesterday was the narrowest of 7 days, ties excluded),
level L = high[t-1]. If high[t] > L buy at max(open[t], L); sell at the day's close."""
from __future__ import annotations

import numpy as np

from src.indicators import bar_range

ID, NAME, KIND, VERSIONS, EXIT = "S11", "NR7 돌파", "bar", ("bar",), "close"


def entries(f):
    o, h = f["open"], f["high"]
    r = bar_range(h, f["low"])
    nr7 = r.shift(1) < r.shift(2).rolling(6, min_periods=6).min()
    lvl = h.shift(1)
    enter = (nr7 & (h > lvl)).to_numpy()
    return enter, np.maximum(o, lvl).to_numpy()
