"""S6 Double 7s: close > SMA(200) and close[t] = min(close[t-6..t]); exit when close[t] = max(close[t-6..t])."""
from __future__ import annotations

from src.indicators import sma

ID, NAME, KIND, VERSIONS = "S6", "Double 7s", "engine", ("close", "open")


def signals(f, cal):
    c = f["close"]
    lo = c.rolling(7, min_periods=7).min()
    hi = c.rolling(7, min_periods=7).max()
    cand = (c > sma(c, 200)) & (c == lo)
    return cand.to_numpy(), (c == hi).to_numpy()
