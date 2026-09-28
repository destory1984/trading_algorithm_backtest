"""S3 cumulative RSI: close > SMA(200) and RSI(2)[t] + RSI(2)[t-1] < 35; exit when RSI(2) > 65."""
from __future__ import annotations

from src.indicators import rsi, sma

ID, NAME, KIND, VERSIONS = "S3", "누적 RSI", "engine", ("close", "open")


def signals(f, cal):
    c = f["close"]
    r = rsi(c, 2)
    cand = (c > sma(c, 200)) & (r + r.shift(1) < 35)
    return cand.to_numpy(), (r > 65).to_numpy()
