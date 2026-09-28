"""S7 3-day high/low: close > SMA(200), close < SMA(5), three lower highs and three lower lows in a row
(high[t] < high[t-1] < high[t-2] < high[t-3], same for lows). Exit when close > SMA(5)."""
from __future__ import annotations

from src.indicators import sma

ID, NAME, KIND, VERSIONS = "S7", "3일 고저", "engine", ("close", "open")


def signals(f, cal):
    c, h, l = f["close"], f["high"], f["low"]
    s5 = sma(c, 5)
    lower_h = (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3))
    lower_l = (l < l.shift(1)) & (l.shift(1) < l.shift(2)) & (l.shift(2) < l.shift(3))
    cand = (c > sma(c, 200)) & (c < s5) & lower_h & lower_l
    return cand.to_numpy(), (c > s5).to_numpy()
