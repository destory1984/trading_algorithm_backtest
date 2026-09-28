"""Weight-following account, copied from index_timing_backtest/src/sim.py with two additions: the trade price can be
the next day's close instead of its open (SPEC 2.3), and cash can earn a yearly rate, accrued by calendar days.

Start with 1 in cash on data.start. On each day, if the previous close decided a target different from the current
one, trade to target x equity at that day's `exec` price (open or close), paying `cost` per unit of value traded.
On the first day the last decision made before it applies. Equity is marked at each close.
Returns daily returns (first day: close / 1 - 1), the target in force each day, and the trade dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(px: pd.DataFrame, dec: pd.Series, start: pd.Timestamp, cost: float, exec_at: str = "close",
             cash_rate: float = 0.0) -> tuple[pd.Series, pd.Series, list]:
    idx = px.index
    s0 = int(idx.searchsorted(start))
    c = px["close"].to_numpy(np.float64)
    p = px[exec_at].to_numpy(np.float64)
    d = dec.reindex(idx).to_numpy(np.float64)
    days = np.diff(idx.to_numpy()).astype("timedelta64[D]").astype(np.float64)
    grow = np.concatenate([[1.0], (1 + cash_rate) ** (days / 365.25)])
    before = d[:s0][np.isfinite(d[:s0])]
    pending = before[-1] if len(before) else np.nan  # the target wanted when the curve starts
    cash, units, cur = 1.0, 0.0, 0.0
    eq = np.empty(len(idx) - s0)
    tgt = np.empty(len(idx) - s0)
    trades = []
    for k, i in enumerate(range(s0, len(idx))):
        if k > 0:
            cash *= grow[i]
        want = pending if k == 0 else d[i - 1]
        if np.isfinite(want) and want != cur:
            value = cash + units * p[i]
            delta = want * value - units * p[i]
            cash -= delta + abs(delta) * cost
            units += delta / p[i]
            cur = want
            trades.append(idx[i])
        eq[k] = cash + units * c[i]
        tgt[k] = cur
    dates = idx[s0:]
    e = pd.Series(eq, index=dates)
    ret = e / e.shift(1).fillna(1.0) - 1
    return ret, pd.Series(tgt, index=dates), trades
