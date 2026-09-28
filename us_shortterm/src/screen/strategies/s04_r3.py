"""S4 R3: close > SMA(200); RSI(2)[t-2] < 60; RSI(2) falls three days in a row (t-3 > t-2 > t-1 > t);
RSI(2)[t] < 10. Exit when RSI(2) > 70."""
from __future__ import annotations

from src.indicators import rsi, sma

ID, NAME, KIND, VERSIONS = "S4", "R3", "engine", ("close", "open")


def signals(f, cal):
    c = f["close"]
    r = rsi(c, 2)
    r1, r2, r3 = r.shift(1), r.shift(2), r.shift(3)
    cand = (c > sma(c, 200)) & (r2 < 60) & (r2 < r3) & (r1 < r2) & (r < r1) & (r < 10)
    return cand.to_numpy(), (r > 70).to_numpy()
