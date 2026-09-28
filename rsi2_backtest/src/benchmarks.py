"""Stage 3 benchmarks on the same ticker and days as a strategy.

hold  buy at the first day's close (x buy cost), sell at the last day's close (x sell cost)
kdd   the ticker at a constant daily weight k, the rest in cash at 0%, k chosen so the curve's max drawdown equals the
      strategy's; k <= 1 (scaled down, never levered); when even k = 1 is shallower, k = 1 and kdd_capped = True
kexp  the same with k = the strategy's exposure (share of days whose return depends on a position)
kdd and kexp have no trading costs (as ibs_lev_backtest's k1x).

Copied from ibs_lev_backtest/src/backtest.py: hold_curve, mix_curve, max_dd, match_k (cash rate fixed at 0 here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import daily_returns, stats


def hold_curve(close: pd.Series, cal: pd.DatetimeIndex, cap: float, buy_cost: float = 1.0,
               sell_keep: float = 1.0) -> pd.Series:
    c = close.reindex(cal).ffill()
    eq = cap / buy_cost * c / c.iloc[0]
    eq.iloc[-1] *= sell_keep
    return eq


def mix_curve(r1: np.ndarray, k: float, cap: float) -> np.ndarray:
    return cap * np.cumprod(1 + k * r1)


def max_dd(eq) -> float:
    e = np.asarray(eq, np.float64)
    return float((e / np.maximum.accumulate(e) - 1).min())


def match_k(r1: np.ndarray, target_mdd: float) -> tuple[float, bool]:
    """Weight k in [0, 1] whose max drawdown equals target_mdd (bisection; deeper as k grows). If even k = 1 is
    shallower than the target, return (1.0, True): the cap binds."""
    if max_dd(mix_curve(r1, 1.0, 1.0)) > target_mdd:
        return 1.0, True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if max_dd(mix_curve(r1, mid, 1.0)) > target_mdd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def matched(r1: np.ndarray, hold: pd.Series, cap: float, strat_mdd: float, strat_exposure: float) -> dict:
    """r1: the ticker's daily close-to-close returns on hold.index, 0 on the first day."""
    k, capped = match_k(r1, strat_mdd)
    cal = hold.index
    return {"hold": hold, "kdd": pd.Series(mix_curve(r1, k, cap), cal), "kdd_k": k, "kdd_capped": capped,
            "kexp": pd.Series(mix_curve(r1, strat_exposure, cap), cal), "kexp_k": strat_exposure}


def bench_row(b: dict, cap: float) -> dict:
    row = {"kdd": b["kdd_k"], "kdd_capped": b["kdd_capped"], "kexp": b["kexp_k"]}
    for n in ("hold", "kdd", "kexp"):
        s = stats(daily_returns(b[n], cap))
        row.update({f"{n}_cagr": s["cagr"], f"{n}_sharpe": s["sharpe"], f"{n}_mdd": s["mdd"]})
    return row
