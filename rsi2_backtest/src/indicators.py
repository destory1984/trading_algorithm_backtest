"""Indicators and the rule's two boolean arrays, shared by the plain loop (replicate.py) and the engine (strategy.py).

Written from the spec's rule text (the original repo has no license; nothing is copied).
rsi_wilder  RSI(n): up and down moves of close-to-close changes averaged with ewm(alpha=1/n, min_periods=n,
            adjust=False); NaN where the average down move is 0 (the spec: no value, no signal)
sma         simple moving average, NaN until n values exist
ibs         (close - low) / (high - low), `flat` where high == low (work order 1's rule)
signals     cand = entry signal at the day's close, exit_cond = exit condition at the day's close
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRENDS = ("above", "none", "below")
EXITS = ("green2", "sma5", "rsi70")


def rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    change = close.diff()
    up = change.clip(lower=0.0)
    down = (-change).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + avg_up / avg_down.where(avg_down != 0))


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ibs(high, low, close, flat: float) -> np.ndarray:
    h, l, c = (np.asarray(x, np.float64) for x in (high, low, close))
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (c - l) / rng
    return np.where(rng == 0, flat, v)


def signals(ind: dict[str, np.ndarray], thr: float, trend: str, exit: str, eligible: np.ndarray,
            rsi_exit: float) -> tuple[np.ndarray, np.ndarray]:
    """cand: eligible & RSI < thr [& close > MA | & close < MA]; a missing RSI or MA gives no signal.
    exit_cond: green2 = green today and yesterday (the first row has no yesterday); sma5 = close > SMA5;
    rsi70 = RSI > rsi_exit."""
    rsi, close, ma = ind["rsi"], ind["close"], ind["ma"]
    has_rsi, has_ma = np.isfinite(rsi), np.isfinite(ma)
    cand = eligible & has_rsi & (np.where(has_rsi, rsi, np.inf) < thr)
    if trend == "above":
        cand = cand & has_ma & (close > np.where(has_ma, ma, np.inf))
    elif trend == "below":
        cand = cand & has_ma & (close < np.where(has_ma, ma, -np.inf))
    elif trend != "none":
        raise ValueError(f"unknown trend filter {trend!r}")
    if exit == "green2":
        g = np.asarray(ind["green"], bool)
        prev = np.zeros_like(g)
        prev[1:] = g[:-1]
        cond = g & prev
    elif exit == "sma5":
        s5 = ind["sma5"]
        ok = np.isfinite(s5)
        cond = ok & (close > np.where(ok, s5, np.inf))
    elif exit == "rsi70":
        cond = has_rsi & (np.where(has_rsi, rsi, -np.inf) > rsi_exit)
    else:
        raise ValueError(f"unknown exit {exit!r}")
    return np.asarray(cand, bool), np.asarray(cond, bool)
