"""S10 volatility breakout: level L = open[t] + 0.5 x (high[t-1] - low[t-1]). If high[t] >= L buy at
max(open[t], L); sell at the next open."""
from __future__ import annotations

import numpy as np

ID, NAME, KIND, VERSIONS, EXIT = "S10", "변동성 돌파", "bar", ("bar",), "next_open"


def entries(f):
    o, h, l = f["open"], f["high"], f["low"]
    lvl = o + 0.5 * (h.shift(1) - l.shift(1))
    enter = (h >= lvl).to_numpy()
    return enter, np.maximum(o, lvl).to_numpy()
