"""Buy-and-hold baselines on the same dates and ETF (SPEC 9).

hold      all in, no costs; bought at the first day's close, so the first day returns 0
risk      weight w rebalanced daily (rest in cash at 0%); w in [0, 1] found by bisection so the curve's max drawdown
          equals the strategy's. w = 1 when the strategy's drawdown is deeper than the hold's
exposure  weight w = the strategy's exposure, rebalanced daily
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity import cagr, mdd


def hold_returns(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.Series:
    r = np.zeros(len(dates))
    r[1:] = close[1:] / close[:-1] - 1
    return pd.Series(r, index=dates)


def risk_weight(hold: pd.Series, target_mdd: float, iters: int = 100) -> float:
    """w with mdd(w x hold) = target_mdd (both <= 0)."""
    if target_mdd <= mdd(hold):
        return 1.0
    if target_mdd >= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        w = (lo + hi) / 2
        lo, hi = (w, hi) if mdd(w * hold) > target_mdd else (lo, w)
    return (lo + hi) / 2


def compare(hold: pd.Series, strat_cagr: float, strat_mdd: float, exposure: float) -> dict:
    w = risk_weight(hold, strat_mdd)
    risk_cagr = cagr(w * hold)
    exp_cagr = cagr(exposure * hold)
    return {
        "hold_cagr": cagr(hold), "hold_mdd": mdd(hold),
        "risk_w": w, "risk_cagr": risk_cagr, "risk_mdd": mdd(w * hold),
        "exp_cagr": exp_cagr,
        "diff_risk": strat_cagr - risk_cagr, "diff_exp": strat_cagr - exp_cagr,
    }
