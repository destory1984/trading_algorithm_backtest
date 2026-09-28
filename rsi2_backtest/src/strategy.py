"""The RSI(2) rule on the krxbt engine (stage 1).

frames(cfg, tickers)  krxbt.us ticker frames (dividend-adjusted OHLC) cut at data.end, plus raw_open / raw_close
                      (unadjusted) and rsi, sma5, green, ibs
csv_frame(cfg, ind)   an engine frame built from the original CSV (same prices as stage 0)
run_rule(...)         engine trades for one rule. The engine looks at the exit array from the entry day on:
  open   (both rules false) cand on day s -> buy at s+1's open; exit array true on day d >= entry -> sell at d+1's
         open. On the entry day e the green2 array is green[e] & green[s]: the signal day counts (the original's
         quirk). skip=1 ignores the exit array on the entry day (both green days after the signal day).
  close  (both rules true) cand on day s -> buy at s's close; exit array true on day d -> sell at d's close. The engine
         would also read the exit array on the entry day itself and sell at the same close, so the entry day is
         always ignored (skip >= 1). The signal day still counts for green2 on the day after the entry.
Ignoring the exit array on entry days = masking it there and re-running until the entry days stop changing (at most
MAX_PASSES runs). checks.grid_vs_loop compares every grid row with the plain loop replicate.simulate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine
from krxbt import portfolio as kp
from krxbt import us as kus

from .common import with_exec
from .indicators import ibs, rsi_wilder, signals, sma

NO_HOLD = 10_000
MAX_PASSES = 50


def add_indicators(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """rsi and sma5 on the adjusted close; green = raw close > raw open (a day with open == close must not turn green
    through the adjustment factor's rounding); ibs on adjusted high/low/close (a ratio, the factor cancels)."""
    f = f.copy()
    f["rsi"] = rsi_wilder(f["close"], cfg["rsi"]["period"])
    f["sma5"] = sma(f["close"], cfg["sma_exit"])
    f["green"] = (f["raw_close"] > f["raw_open"]).to_numpy(bool)
    f["ibs"] = ibs(f["high"], f["low"], f["close"], cfg["ibs"]["flat_value"])
    return f


def frames(cfg: dict, tickers: list[str]) -> dict[str, pd.DataFrame]:
    cal = kus.load_calendar(cfg)
    idx = kus.index_features(cfg)
    end = pd.Timestamp(cfg["data"]["end"])
    out = {}
    for t in tickers:
        f = kus.ticker_frame(cfg, t, cal, idx)
        if f is None:
            raise FileNotFoundError(f"no us_data prices for {t}")
        raw = kus.load_prices(cfg, t)
        raw = raw[~raw.index.duplicated(keep="last")]
        f["raw_open"] = raw["open"].reindex(f.index)
        f["raw_close"] = raw["close"].reindex(f.index)
        out[t] = add_indicators(f[f.index <= end], cfg)
    return out


def csv_frame(cfg: dict, ind: pd.DataFrame) -> pd.DataFrame:
    """Engine columns from replicate.with_indicators rows: every day valid, eligible from data.start once the SMA200
    exists; market columns unused; ibs not used on the CSV."""
    f = ind[["open", "high", "low", "close", "ma", "rsi", "sma5", "green"]].copy()
    f["disp"] = f["close"] / f["ma"] * 100
    f["idx_disp"] = np.nan
    f["valid"] = True
    f["eligible"] = f["ma"].notna() & (f.index >= pd.Timestamp(cfg["data"]["start"]))
    f["idx_regime"] = False
    f["idx_crash"] = False
    f["data_break"] = False
    f["ibs"] = np.nan
    return f


def arrays(f: pd.DataFrame) -> dict:
    a = engine.arrays(f)
    for k in ("rsi", "sma5", "ibs"):
        a[k] = f[k].to_numpy(np.float64)
    a["green"] = f["green"].to_numpy(bool)
    return a


def rule(a: dict, cfg: dict, thr: float, trend: str, exit: str) -> tuple[np.ndarray, np.ndarray]:
    ind = {"rsi": a["rsi"], "close": a["close"], "ma": a["ma"], "sma5": a["sma5"], "green": a["green"]}
    return signals(ind, thr, trend, exit, a["eligible"], cfg["rsi"]["exit_above"])


def ibs_rule(a: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Work order 1's original rule: cand = eligible & IBS < ibs.entry, target = IBS > ibs.exit."""
    ib = a["ibs"]
    ok = np.isfinite(ib)
    cand = a["eligible"] & ok & (np.where(ok, ib, 1.0) < cfg["ibs"]["entry"])
    target = ok & (np.where(ok, ib, 0.0) > cfg["ibs"]["exit"])
    return cand, target


def run_rule(a: dict, cfg: dict, cand: np.ndarray, target: np.ndarray, mode: str, skip: int = 0,
             independent: bool = False) -> pd.DataFrame:
    """Engine trades; the exit array is ignored on the first `skip` days of every trade (close: at least 1)."""
    c = with_exec(cfg, mode)
    skip = max(skip, 1) if mode == "close" else skip
    if independent and skip:
        raise ValueError("independent trades overlap: entry-day masking needs one position at a time")
    n = len(cand)
    ix = pd.DatetimeIndex(a["dates"])
    mask = np.zeros(n, bool)
    for _ in range(MAX_PASSES):
        tr = engine.run(a, c, cand, target & ~mask, stop=None, hold=NO_HOLD, independent=independent)
        new = np.zeros(n, bool)
        e = ix.get_indexer(tr["entry_date"])
        for k in range(skip):
            j = e + k
            new[j[j < n]] = True
        if np.array_equal(new, mask):
            return tr
        mask = new
    raise RuntimeError(f"entry-day mask did not settle in {MAX_PASSES} passes")


def calendar(cfg: dict, f: pd.DataFrame) -> pd.DatetimeIndex:
    return f.index[f.index >= pd.Timestamp(cfg["data"]["start"])]


def equity(cfg: dict, f: pd.DataFrame, tr: pd.DataFrame, ticker: str, cap: float | None = None,
           rank_by: str = "signal_date") -> tuple[pd.Series, dict]:
    """Daily equity of one ticker traded with one slot (krxbt.portfolio.run_portfolio)."""
    cal = calendar(cfg, f)
    closes = pd.DataFrame({ticker: f["close"]}).reindex(cal).ffill()
    return kp.run_portfolio(cal, tr.assign(ticker=ticker), closes,
                            cap if cap is not None else cfg["portfolio"]["initial_capital"], slots=1, rank_by=rank_by)
