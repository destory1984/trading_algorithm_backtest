"""Weight-following account: start with 1 in cash at data.start's open. At each open, if the previous close decided
a target different from the current one, trade to target x equity at that open, paying `cost` per unit of value
traded. Cash earns 0. Equity is marked at each close.
Returns daily returns (first day: close / 1 - 1), the target in force each day, and the trade dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(px: pd.DataFrame, dec: pd.Series, start: pd.Timestamp, cost: float) -> tuple[pd.Series, pd.Series, list]:
    idx = px.index
    s0 = int(idx.searchsorted(start))
    o, c = px["open"].to_numpy(np.float64), px["close"].to_numpy(np.float64)
    d = dec.reindex(idx).to_numpy(np.float64)
    before = d[:s0][np.isfinite(d[:s0])]
    pending = before[-1] if len(before) else np.nan  # the target wanted when the curve starts
    cash, units, cur = 1.0, 0.0, 0.0
    eq = np.empty(len(idx) - s0)
    tgt = np.empty(len(idx) - s0)
    trades = []
    for k, i in enumerate(range(s0, len(idx))):
        want = pending if k == 0 else d[i - 1]
        if np.isfinite(want) and want != cur:
            value = cash + units * o[i]
            delta = want * value - units * o[i]
            cash -= delta + abs(delta) * cost
            units += delta / o[i]
            cur = want
            trades.append(idx[i])
        eq[k] = cash + units * c[i]
        tgt[k] = cur
    dates = idx[s0:]
    e = pd.Series(eq, index=dates)
    ret = e / e.shift(1).fillna(1.0) - 1
    return ret, pd.Series(tgt, index=dates), trades
