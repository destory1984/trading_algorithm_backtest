"""Monthly-rebalanced account.

Start with 1 in cash at data.start's open. At the open after each month-end decision (and at the first open, using
the last decision made before it) every holding is set to target x equity, paying `cost` per unit of value traded.
The trade is sized on equity net of its own cost (value x (1 - cost x turnover)), so what is left over in cash is
the untargeted part plus a rounding residue of order cost^2. Cash earns `cash` a year, accrued per trading day.
Equity is marked at each close.
Returns daily returns (first day: close / 1 - 1) and one row per rebalance: date, turnover (traded value / equity
before the trade) and cost paid (as a fraction of that equity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(opn: pd.DataFrame, cls: pd.DataFrame, dec: pd.DataFrame, start: pd.Timestamp, cost: float,
             cash_rate: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    idx = opn.index
    s0 = int(idx.searchsorted(start))
    o, c = opn.to_numpy(np.float64), cls.to_numpy(np.float64)
    dec = dec.dropna(how="any")
    pos = idx.get_indexer(dec.index)
    assert (pos >= 0).all()
    wants = {}  # execution day index -> target vector
    before = dec[dec.index < idx[s0]]
    if len(before):
        wants[s0] = before.iloc[-1].to_numpy(np.float64)
    for p, row in zip(pos, dec.to_numpy(np.float64)):
        if p + 1 >= s0 and p + 1 < len(idx) and (p + 1 != s0 or s0 not in wants):
            wants[p + 1] = row
    daily_cash = (1 + cash_rate) ** (1 / 252) - 1
    cash, units = 1.0, np.zeros(o.shape[1])
    eq = np.empty(len(idx) - s0)
    trades = []
    for k, i in enumerate(range(s0, len(idx))):
        if k > 0:
            cash *= 1 + daily_cash
        if i in wants:
            w = wants[i]
            held = units * o[i]
            value = cash + held.sum()
            turn = np.abs(w * value - held).sum() / value
            net = value * (1 - cost * turn)
            delta = w * net - held
            fee = np.abs(delta).sum() * cost
            cash -= delta.sum() + fee
            units = units + delta / o[i]
            trades.append((idx[i], np.abs(delta).sum() / value, fee / value))
        eq[k] = cash + units @ c[i]
    dates = idx[s0:]
    e = pd.Series(eq, index=dates)
    ret = e / e.shift(1).fillna(1.0) - 1
    return ret, pd.DataFrame(trades, columns=["date", "turnover", "cost"])
