"""Trade list -> daily equity curve, and the SPEC 8 metrics.

Initial capital 1, all in, flat days return 0. A trade's daily factors:
  entry day    close / entry_px (1 for a close entry), divided by the buy cost factor
  middle days  close / previous close
  exit day     exit_px / previous close, times the sell keep factor
  same day     exit_px / entry_px with both cost factors
So a trade's factors multiply to 1 + ret exactly, and the curve's last value is prod(1 + ret).
Trades never overlap (the engine ignores signals while holding; bar_sim exits before the next entry), so a day
touched by two trades (exit at the open, new entry later that day) gets the product of both factors.

Exposure = share of days whose return depends on a position: entry day to exit day, except the entry day of a
close entry (bought at the close, nothing earned that day). A same-day trade's day counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from src.common import ZERO_COSTS, one_way

CHECK_TOL = 1e-8


def curve(tr: pd.DataFrame, dates: pd.DatetimeIndex, close: np.ndarray, entry_on_close: bool,
          costs: dict) -> tuple[pd.Series, np.ndarray]:
    """(daily returns on `dates`, exposure mask). `close` is aligned with `dates`; trades must lie inside `dates`."""
    buy_cost, sell_keep = cost_factors({"costs": costs})
    ix = pd.Index(dates)
    e_ = ix.get_indexer(pd.DatetimeIndex(tr["entry_date"]))
    x_ = ix.get_indexer(pd.DatetimeIndex(tr["exit_date"]))
    if (e_ < 0).any() or (x_ < 0).any():
        raise ValueError("trade outside the curve's dates")
    fac = np.ones(len(dates))
    pos = np.zeros(len(dates), np.bool_)
    ratio = np.ones(len(dates))
    ratio[1:] = close[1:] / close[:-1]
    for e, x, ep, xp in zip(e_, x_, tr["entry_px"].to_numpy(), tr["exit_px"].to_numpy()):
        if e == x:
            fac[e] *= xp / ep * sell_keep / buy_cost
            pos[e] = True
            continue
        fac[e] *= close[e] / ep / buy_cost
        fac[e + 1:x] *= ratio[e + 1:x]
        fac[x] *= xp / close[x - 1] * sell_keep
        pos[e + (1 if entry_on_close else 0):x + 1] = True
    return pd.Series(fac - 1, index=dates), pos


def cagr(daily: pd.Series) -> float:
    last = float((1 + daily).prod())
    days = (daily.index[-1] - daily.index[0]).days
    return last ** (365.25 / days) - 1 if last > 0 else -1.0


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def mdd(daily: pd.Series) -> float:
    eq = (1 + daily).cumprod()
    return float(min((eq / eq.cummax() - 1).min(), 0.0))


def trade_rets(tr: pd.DataFrame, costs: dict) -> np.ndarray:
    buy_cost, sell_keep = cost_factors({"costs": costs})
    return tr["exit_px"].to_numpy(np.float64) * sell_keep / (tr["entry_px"].to_numpy(np.float64) * buy_cost) - 1


def breakeven_bp(tr: pd.DataFrame, lo: float = -1000.0, hi: float = 1000.0, iters: int = 100) -> float:
    """One-way cost (bp, same on both sides) at which the mean trade return is 0; bisection. Negative when the
    trades lose on average even without costs."""
    if not len(tr):
        return np.nan
    mean = lambda bp: trade_rets(tr, one_way(bp)).mean()
    if mean(lo) <= 0:
        return lo
    if mean(hi) >= 0:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def profit_factor(ret: np.ndarray) -> float:
    if not len(ret):
        return np.nan
    loss = -ret[ret < 0].sum()
    return float(ret[ret > 0].sum() / loss) if loss > 0 else np.inf


def metrics(tr: pd.DataFrame, dates: pd.DatetimeIndex, close: np.ndarray, entry_on_close: bool,
            costs: dict) -> tuple[dict, pd.Series]:
    """SPEC 8 metrics and the daily return series. Raises if the curve's last value misses prod(1 + ret)."""
    daily, pos = curve(tr, dates, close, entry_on_close, costs)
    ret = tr["ret"].to_numpy(np.float64)
    last, prod = float((1 + daily).prod()), float(np.prod(1 + ret))
    if abs(last - prod) >= CHECK_TOL:
        raise AssertionError(f"curve {last!r} != prod(1+ret) {prod!r}")
    daily0, _ = curve(tr, dates, close, entry_on_close, ZERO_COSTS)
    return {
        "trades": len(tr),
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "expectancy": float(ret.mean()) if len(ret) else np.nan,
        "profit_factor": profit_factor(ret),
        "cagr": cagr(daily),
        "sharpe": sharpe(daily),
        "mdd": mdd(daily),
        "exposure": float(pos.mean()),
        "cagr_zero_cost": cagr(daily0),
        "breakeven_bp": breakeven_bp(tr),
        "curve_last": last,
        "prod_ret": prod,
        "curve_check": abs(last - prod) < CHECK_TOL,
    }, daily
