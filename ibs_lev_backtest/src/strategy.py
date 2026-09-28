"""The IBS rule on the krxbt engine.

cand    eligible & IBS < ibs.entry      (NaN IBS on high == low bars -> False)
target  IBS > ibs.exit                  (NaN -> False)
open    both rules false: the previous day's IBS acts at today's open (the original)
close   both true: signal-day close in, exit-signal-day close out (ibs_backtest's meaning)
No stop, no max hold (hold = NO_HOLD), one position at a time (engine default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine

from .common import ticker_costs, with_exec

NO_HOLD = 10_000


def arrays(f: pd.DataFrame) -> dict:
    a = engine.arrays(f)
    a["ibs"] = f["ibs"].to_numpy(np.float64)
    return a


def signals(a: dict, entry: float, exit: float) -> tuple[np.ndarray, np.ndarray]:
    ib = a["ibs"]
    ok = np.isfinite(ib)
    cand = a["eligible"] & ok & (np.where(ok, ib, 1.0) < entry)
    target = ok & (np.where(ok, ib, 0.0) > exit)
    return cand, target


def run(a: dict, cfg: dict, mode: str) -> pd.DataFrame:
    cand, target = signals(a, cfg["ibs"]["entry"], cfg["ibs"]["exit"])
    return engine.run(a, with_exec(cfg, mode), cand, target, stop=None, hold=NO_HOLD)


def calendar(cfg: dict, f: pd.DataFrame) -> pd.DatetimeIndex:
    """The ticker's days from data.start (its listing day when later) to its last day."""
    return f.index[f.index >= pd.Timestamp(cfg["data"]["start"])]


def trades(cfg: dict, f: pd.DataFrame, ticker: str, mode: str, with_costs: bool) -> pd.DataFrame:
    # the engine runs on the full frame (from data.start or the ticker's listing, whichever is
    # later, but always from f's first row); this cuts entries before calendar(cfg, f)[0], i.e.
    # before data.start (2008-01-01). A trade already open on that date (e.g. entered 2007) is
    # dropped whole, not clipped -- this only removes a few days at the start of the C5 sleeves,
    # which run from 2008-01-02 (the basket tickers are listed well before 2008).
    tr = run(arrays(f), ticker_costs(cfg, ticker, with_costs), mode)
    return tr[tr["entry_date"] >= calendar(cfg, f)[0]].reset_index(drop=True)
