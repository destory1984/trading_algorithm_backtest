"""T3 volatility targeting, copied from index_timing_backtest/src/rules.py (week_end and t3, unchanged).

decision[t] = the target weight chosen at day t's close; NaN = no decision that day (keep the current target).
Week-end = a day whose next trading day falls in another ISO week, so the last day of the data is never one.
  T3 (target) at week-ends: w = min(1, target / (std of the last vol_window daily returns x sqrt 252)); it becomes
              the new target only if it differs from the current target by more than `band` (current starts at 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def week_end(idx: pd.DatetimeIndex) -> np.ndarray:
    iso = idx.isocalendar()
    wk = (iso["year"] * 100 + iso["week"]).to_numpy()
    out = np.zeros(len(idx), bool)
    out[:-1] = wk[1:] != wk[:-1]
    return out


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


def decisions(close: pd.Series, rule: str, target: float | None, cfg: dict) -> pd.Series:
    if rule == "HOLD":
        return pd.Series(1.0, index=close.index)
    r = cfg["rule"]
    return t3(close, float(target), int(r["vol_window"]), float(r["band"]))
