"""S5 %b: close > SMA(200) and %b below 0.2 on t, t-1 and t-2; exit when %b > 0.8."""
from __future__ import annotations

from src.indicators import pct_b, sma

ID, NAME, KIND, VERSIONS = "S5", "%b", "engine", ("close", "open")


def signals(f, cal):
    c = f["close"]
    b = pct_b(c, 20, 2.0)
    cand = (c > sma(c, 200)) & (b < 0.2) & (b.shift(1) < 0.2) & (b.shift(2) < 0.2)
    return cand.to_numpy(), (b > 0.8).to_numpy()
