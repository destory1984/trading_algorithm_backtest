"""Strategy indicators (SPEC 6). Each value uses data up to that day's close only; NaN means no signal."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI: average up move AG and down move AL of close-to-close changes with ewm(alpha=1/n, adjust=False,
    min_periods=n). RSI = 100 - 100 / (1 + AG/AL); AL = 0 and AG > 0 gives 100, both 0 gives 50."""
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + ag / al)
    out = out.where(al > 0, np.where(ag > 0, 100.0, 50.0))
    return out.where(ag.notna() & al.notna())


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(close - low) / (high - low); NaN where high == low."""
    rng = high - low
    return ((close - low) / rng).where(rng != 0)


def pct_b(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Bollinger %b: middle SMA(n), population std (ddof=0), bands middle +- k sigma."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    lo, hi = mid - k * sd, mid + k * sd
    return ((close - lo) / (hi - lo)).where(hi > lo)


def bar_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low
