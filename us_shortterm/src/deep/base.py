"""Shared pieces of the second round (SPEC2.md): the S1 rule with any thresholds, the two data seals, the stretches.

Stretches:
  screen   2008-01-02 -> 2022-12-30, run on frames sealed at data.screen_end (identical to the first round)
  holdout  first trading day on/after data.holdout_start -> data.deep_end, a slice of the deep run's daily curve;
           the trade open on 2022-12-30 carries on (not closed at the seal); its trades are the ones exiting in it
  full     2008-01-02 -> data.deep_end, the deep run (frames sealed at data.deep_end)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine

from src import benchmark, common, equity
from src.indicators import ibs

STRETCHES = ("screen", "holdout", "full")
NAMES = {"screen": "선별 구간", "holdout": "이미 본 구간", "full": "전체 구간"}


def deep_end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["data"]["deep_end"])


def holdout_start(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["data"]["holdout_start"])


def load(cfg: dict, deep: bool):
    """(calendar, frames) sealed at deep_end (deep=True) or screen_end."""
    return common.load_frames(cfg, end=deep_end(cfg) if deep else None)


def ibs_trades(f: pd.DataFrame, a: dict, cfg: dict, entry: float, exit_: float) -> pd.DataFrame:
    """S1 with thresholds entry / exit, next-day open fills (same as the first round's s01_ibs + run.trades_for)."""
    x = ibs(f["high"], f["low"], f["close"])
    cand = (x < entry).to_numpy() & a["eligible"]
    target = (x > exit_).to_numpy()
    c = common.with_rules(cfg, False, False)
    return engine.run(a, c, cand, target, None, int(cfg["screen"]["hold_max"]))


def rule_trades(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = cfg["deep"]["rule"]
    return ibs_trades(f, engine.arrays(f), cfg, r["entry"], r["exit"])


def curve_inputs(f: pd.DataFrame, cfg: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    g = f[f.index >= pd.Timestamp(cfg["data"]["start"])]
    assert g["valid"].all(), "gap in an ETF's prices"
    return g.index, g["close"].to_numpy(np.float64)


def stretch(tr: pd.DataFrame, f: pd.DataFrame, cfg: dict, costs: dict, lo: pd.Timestamp | None = None) -> dict:
    """Metrics (SPEC 8) and the three baselines (SPEC 9) for the curve from `lo` on (None = from data.start).
    Returns {"m": metrics, "daily": daily returns, "hold": hold returns} on the stretch."""
    dates, close = curve_inputs(f, cfg)
    daily, pos = equity.curve(tr, dates, close, False, costs)
    daily0, _ = equity.curve(tr, dates, close, False, common.ZERO_COSTS)
    hold = benchmark.hold_returns(dates, close)
    k = np.ones(len(dates), bool) if lo is None else np.asarray(dates >= lo)
    sub = tr if lo is None else tr[pd.DatetimeIndex(tr["exit_date"]) >= lo]
    ret = sub["ret"].to_numpy(np.float64)
    if lo is None:
        last = float((1 + daily).prod())
        assert abs(last - float(np.prod(1 + tr["ret"]))) < equity.CHECK_TOL
    m = equity.summarize(sub.assign(ret=ret), daily[k], pos[k], daily0[k])
    m.update(benchmark.compare(hold[k], m["cagr"], m["mdd"], m["exposure"]))
    m.update(first_day=dates[k][0].date(), last_day=dates[k][-1].date())
    return {"m": m, "daily": daily[k], "hold": hold[k]}


def all_stretches(cfg: dict, costs: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Rows (stretch x ticker) and {(stretch, ticker): stretch dict}. Trades are identical across costs; only the
    cost factors change."""
    costs = costs or cfg["costs"]
    _, fs = load(cfg, deep=False)
    _, fd = load(cfg, deep=True)
    rows, out = [], {}
    for t in cfg["screen"]["tickers"]:
        ts = with_costs_ret(rule_trades(fs[t], cfg), costs)
        td = with_costs_ret(rule_trades(fd[t], cfg), costs)
        for name, tr, f, lo in (("screen", ts, fs[t], None), ("holdout", td, fd[t], holdout_start(cfg)),
                                ("full", td, fd[t], None)):
            s = stretch(tr, f, cfg, costs, lo)
            s["trades"] = tr if lo is None else tr[pd.DatetimeIndex(tr["exit_date"]) >= lo]
            out[(name, t)] = s
            rows.append({"stretch": name, "ticker": t, **s["m"]})
    return pd.DataFrame(rows), out


def with_costs_ret(tr: pd.DataFrame, costs: dict) -> pd.DataFrame:
    """The same trades with `ret` recomputed under `costs`."""
    return tr.assign(ret=equity.trade_rets(tr, costs))
