"""SPEC 3: monthly target weights, decided on each month-end close.

`me` is the month-end close frame (rows = month-end days, columns = tickers). Each rule returns a frame on the same
rows with the target weight per ticker (the rest is cash). Rows where the look-back is not full yet are NaN (no
decision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ret_n(me: pd.DataFrame, n: int) -> pd.DataFrame:
    return me / me.shift(n) - 1


def dual_momentum(me: pd.DataFrame, n: int) -> pd.DataFrame:
    """A1 (Antonacci GEM, bond = TLT, hurdle = 0)."""
    r = ret_n(me, n)
    w = pd.DataFrame(0.0, index=me.index, columns=me.columns)
    risk_on = r["SPY"] > 0
    pick = np.where(r["SPY"] >= r["EFA"], "SPY", "EFA")
    for i, day in enumerate(me.index):
        w.loc[day, pick[i] if risk_on.iloc[i] else "TLT"] = 1.0
    w[r[["SPY", "EFA"]].isna().any(axis=1)] = np.nan
    return w


def sma_filter(me: pd.DataFrame, n: int) -> pd.DataFrame:
    """A2 (Faber GTAA): 1/5 each, held only while the month-end close is above the mean of the last n month-ends."""
    ma = me.rolling(n).mean()
    w = (me > ma).astype(float) / len(me.columns)
    w[ma.isna().any(axis=1)] = np.nan
    return w


def top_k(me: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    """A3: the k best n-month returns, 1/k each, and a pick whose return is <= 0 goes to cash."""
    r = ret_n(me, n)
    rank = r.rank(axis=1, ascending=False, method="first")
    w = ((rank <= k) & (r > 0)).astype(float) / k
    w[r.isna().any(axis=1)] = np.nan
    return w


def fixed(me: pd.DataFrame, weights: dict) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=me.index, columns=me.columns)
    for t, v in weights.items():
        w[t] = float(v)
    return w


def decide(cfg: dict, rule: str, param, me: pd.DataFrame) -> pd.DataFrame:
    if rule == "A1":
        return dual_momentum(me, param)
    if rule == "A2":
        return sma_filter(me, param)
    if rule == "A3":
        return top_k(me, param, cfg["rules"]["A3"]["top"])
    return fixed(me, cfg["baselines"][rule]["weights"])
