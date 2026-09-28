"""The three timing rules as decision series.

decision[t] = the target weight chosen at day t's close, to be traded at day t+1's open; NaN = no decision that day
(keep the current target). Month-end / week-end = a day whose next trading day falls in another month / ISO week,
so the last day of the data is never one (that would need tomorrow's date).
  T1 (n days)   1 if close > SMA(n) else 0, every day (0 while the SMA is not ready)
  T2 (n months) at month-ends: 1 if close > mean of the last n month-end closes (this one included) else 0
  T3 (target)   at week-ends: w = min(1, target / (std of the last vol_window daily returns x sqrt 252)); it becomes
                the new target only if it differs from the current target by more than `band` (current starts at 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_end(idx: pd.DatetimeIndex) -> np.ndarray:
    ym = idx.year * 100 + idx.month
    out = np.zeros(len(idx), bool)
    out[:-1] = ym[1:] != ym[:-1]
    return out


def week_end(idx: pd.DatetimeIndex) -> np.ndarray:
    iso = idx.isocalendar()
    wk = (iso["year"] * 100 + iso["week"]).to_numpy()
    out = np.zeros(len(idx), bool)
    out[:-1] = wk[1:] != wk[:-1]
    return out


def t1(close: pd.Series, n: int) -> pd.Series:
    s = close.rolling(n, min_periods=n).mean()
    return (close > s).astype(float)


def t2(close: pd.Series, n: int) -> pd.Series:
    me = month_end(close.index)
    mc = close[me]
    avg = mc.rolling(n, min_periods=n).mean()
    d = pd.Series(np.nan, index=close.index)
    d[me] = (mc > avg).astype(float).to_numpy()
    return d


def t3(close: pd.Series, target: float, window: int, band: float) -> pd.Series:
    vol = close.pct_change().rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    we = week_end(close.index)
    d = pd.Series(np.nan, index=close.index)
    cur = 0.0
    for i in np.flatnonzero(we):
        v = vol.iloc[i]
        if not np.isfinite(v) or v <= 0:
            continue
        w = min(1.0, target / v)
        if abs(w - cur) > band:
            cur = w
            d.iloc[i] = w
    return d


def decisions(close: pd.Series, rule: str, param, cfg: dict) -> pd.Series:
    if rule == "T1":
        return t1(close, int(param))
    if rule == "T2":
        return t2(close, int(param))
    if rule == "T3":
        r = cfg["rules"]["T3"]
        return t3(close, float(param), int(r["vol_window"]), float(r["band"]))
    if rule == "HOLD":
        return pd.Series(1.0, index=close.index)
    raise ValueError(rule)
