"""S2 RSI(2): close > SMA(200) and RSI(2) < 10; exit when close > SMA(5)."""
from __future__ import annotations

from src.indicators import rsi, sma

ID, NAME, KIND, VERSIONS = "S2", "RSI(2)", "engine", ("close", "open")


def signals(f, cal):
    c = f["close"]
    cand = (c > sma(c, 200)) & (rsi(c, 2) < 10)
    return cand.to_numpy(), (c > sma(c, 5)).to_numpy()
