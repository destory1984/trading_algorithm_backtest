"""Simulator for the strategies that buy at an intraday price (S10, S11, S12).

A strategy gives, per day t, whether it buys that day and at what price. Entry is on day t itself
(signal_date = entry_date = t). Exit:
  close      the same day's close
  next_open  the next priced day's open; buying again later that day is allowed (S10)
If the data ends before the exit day, the trade is closed at the last close, reason "forced" (as the engine does).
Trades never overlap in time under these two exits, so every signal day is a trade.
Costs are krxbt.costs.cost_factors, applied to the entry and exit prices like the engine.
Output columns match krxbt.engine.run: signal_date, entry_date, exit_date, entry_px, exit_px, ret, hold_days, reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

EXITS = ("close", "next_open")


def simulate(dates, o, c, valid, enter, entry_px, exit_mode: str, buy_cost: float, sell_keep: float) -> pd.DataFrame:
    if exit_mode not in EXITS:
        raise ValueError(exit_mode)
    dates = np.asarray(dates)
    o, c, epx_all = (np.asarray(x, np.float64) for x in (o, c, entry_px))
    valid = np.asarray(valid, np.bool_)
    enter = np.asarray(enter, np.bool_) & valid & np.isfinite(epx_all)
    nxt = np.full(len(c), -1)  # next priced day after each index
    j = -1
    for i in range(len(c) - 1, -1, -1):
        nxt[i] = j
        if valid[i]:
            j = i
    t_ = np.flatnonzero(enter)
    if exit_mode == "close":
        x_ = t_.copy()
    else:
        x_ = np.where(nxt[t_] >= 0, nxt[t_], t_)
    forced = (exit_mode == "next_open") & (nxt[t_] < 0)
    ep = epx_all[t_]
    xp = np.where(exit_mode == "close", c[t_], np.where(forced, c[t_], o[x_]))
    return pd.DataFrame({
        "signal_date": dates[t_], "entry_date": dates[t_], "exit_date": dates[x_],
        "entry_px": ep, "exit_px": xp, "ret": xp * sell_keep / (ep * buy_cost) - 1.0,
        "hold_days": (x_ - t_ + 1).astype(np.int16), "reason": np.where(forced, "forced", "target"),
    })


def run(f: pd.DataFrame, a: dict, cfg: dict, enter: np.ndarray, entry_px: np.ndarray, exit_mode: str) -> pd.DataFrame:
    """Trades on a ticker frame; buys only on eligible days (same universe gate as the engine strategies)."""
    buy_cost, sell_keep = cost_factors(cfg)
    return simulate(a["dates"], a["open"], a["close"], a["valid"], np.asarray(enter, np.bool_) & a["eligible"],
                    entry_px, exit_mode, buy_cost, sell_keep)
