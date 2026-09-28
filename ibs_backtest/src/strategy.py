"""The IBS rule on the krxbt engine (stage 1).

For each ticker: a krxbt.us ticker frame (dividend-adjusted OHLC) plus an
`ibs` column. The rule is two boolean arrays:
  cand    eligible & IBS < entry [& close > own 200-day MA]
  target  IBS > exit
Execution (config rules, set per run by common.with_exec):
  close  entry_on_signal_close = target_exit_on_close = true
  open   both false: next day's open in, next day's open out
Max hold: the engine exits at that day's close once holding day >= hold
(holding day 1 = entry day). "None" is hold = 10_000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine
from krxbt import portfolio as kp
from krxbt import us as kus

from .common import ibs, with_exec

NO_HOLD = 10_000


def frames(cfg: dict, tickers: list[str]) -> dict[str, pd.DataFrame]:
    cal = kus.load_calendar(cfg)
    idx = kus.index_features(cfg)
    end = cfg["data"].get("end")
    out = {}
    for t in tickers:
        f = kus.ticker_frame(cfg, t, cal, idx)
        if f is None:
            raise FileNotFoundError(f"no us_data prices for {t}; run python -m krxbt.fetch_us")
        if end:
            f = f[f.index <= pd.Timestamp(end)]
        f["ibs"] = ibs(f["high"], f["low"], f["close"], cfg["ibs"]["flat_value"])
        out[t] = f
    return out


def arrays(f: pd.DataFrame) -> dict:
    a = engine.arrays(f)
    a["ibs"] = f["ibs"].to_numpy(np.float64)
    return a


def signals(a: dict, entry: float, exit: float, trend: str) -> tuple[np.ndarray, np.ndarray]:
    cand = a["eligible"] & (a["ibs"] < entry)
    if trend == "ma200":
        cand &= a["close"] > a["ma"]
    elif trend != "none":
        raise ValueError(f"unknown trend filter {trend!r}")
    return cand, a["ibs"] > exit


def run(a: dict, cfg: dict, entry: float, exit: float, mode: str, trend: str = "none",
        hold: int | None = None) -> pd.DataFrame:
    cand, target = signals(a, entry, exit, trend)
    return engine.run(a, with_exec(cfg, mode), cand, target, stop=None, hold=hold or NO_HOLD)


def calendar(cfg: dict, f: pd.DataFrame) -> pd.DatetimeIndex:
    """Days from data.start to the ticker's last day."""
    return f.index[f.index >= pd.Timestamp(cfg["data"]["start"])]


def equity(cfg: dict, f: pd.DataFrame, tr: pd.DataFrame, ticker: str) -> tuple[pd.Series, dict]:
    """Daily equity of one ticker traded with one slot (krxbt.portfolio.run_portfolio)."""
    cal = calendar(cfg, f)
    closes = pd.DataFrame({ticker: f["close"]}).reindex(cal).ffill()
    return kp.run_portfolio(cal, tr.assign(ticker=ticker), closes, cfg["portfolio"]["initial_capital"],
                            slots=1, rank_by="signal_date")


def daily_returns(eq: pd.Series, capital: float) -> pd.Series:
    prev = eq.shift(1)
    prev.iloc[0] = capital
    return eq / prev - 1


def in_position(cal: pd.DatetimeIndex, tr: pd.DataFrame) -> pd.Series:
    """Days whose return depends on a position: entry day (after the open) or the day after the
    signal-close entry, through the exit day."""
    pos = np.zeros(len(cal), bool)
    ix = {d: i for i, d in enumerate(cal)}
    for e, x, s in zip(tr["entry_date"], tr["exit_date"], tr["signal_date"]):
        i0 = ix.get(e)
        i1 = ix.get(x)
        if i0 is None or i1 is None:
            continue
        if e == s:  # bought at the close: exposed from the next day
            i0 += 1
        pos[i0:i1 + 1] = True
    return pd.Series(pos, cal)
