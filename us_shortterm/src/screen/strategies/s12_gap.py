"""S12 gap-down reversal: if open[t] < low[t-1] buy at the open; sell at the day's close."""
from __future__ import annotations

ID, NAME, KIND, VERSIONS, EXIT = "S12", "갭 하락 되돌림", "bar", ("bar",), "close"


def entries(f):
    o = f["open"]
    return (o < f["low"].shift(1)).to_numpy(), o.to_numpy()
