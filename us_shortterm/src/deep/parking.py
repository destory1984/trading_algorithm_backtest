"""SPEC2 4.7: hold SPY at weight w while idle, go all in on an S1 signal. Reference only, not a criterion.

park_equity and static_mix are copied from ibs_backtest/src/park.py (same bookkeeping: units of SPY and of the
signal ETF, costs only on what is actually bought or sold, trades at their own bar's open or close). Each ETF's
S1 trades run on their own (one ETF per account). Full stretch; the screen and holdout stretches are slices of
the same equity curve. The comparison is a fixed SPY + cash mix at the run's average SPY weight.
Usage: python -m src.deep.parking   -> results/deep/parking.csv
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from src import common, equity
from src.deep import base


def bar_time(f: pd.DataFrame, d: pd.Timestamp, px: float) -> str:
    for k in ("open", "close"):
        if math.isclose(float(f.at[d, k]), px, rel_tol=1e-9):
            return k
    raise ValueError(f"{d.date()} price {px} is neither open nor close")


def park_equity(cal: pd.DatetimeIndex, trades: pd.DataFrame, fr: dict[str, pd.DataFrame], cap: float,
                w: float, park_ticker: str, costs: tuple[float, float]) -> pd.Series:
    prev_exit = None
    for r in trades.itertuples():
        if prev_exit is not None and not (r.entry_date > prev_exit):
            raise ValueError(f"entry_date {r.entry_date} is not strictly after previous exit_date {prev_exit}")
        prev_exit = r.exit_date
    bc, sk = costs
    sp = fr[park_ticker]
    ent = {d: r for d, r in zip(trades["entry_date"], trades.itertuples())}
    ext = {d: r for d, r in zip(trades["exit_date"], trades.itertuples())}
    cash, spy, units, tk = float(cap), 0.0, 0.0, None  # spy = parked SPY units; units/tk = open trade
    started = False
    eq = np.empty(len(cal))
    wts = np.zeros((len(cal), 2))
    for i, d in enumerate(cal):
        if d in ent and tk is None:
            r = ent[d]
            f = fr[r.ticker]
            px = float(r.entry_px)
            bar_time(f, d, px)
            if r.ticker == park_ticker:
                units, spy = spy + cash / (px * bc), 0.0
            else:
                cash += spy * float(sp.at[d, bar_time(f, d, px)]) * sk
                spy = 0.0
                units = cash / (px * bc)
            cash, tk = 0.0, r.ticker
            started = True
        if d in ext and tk is not None and ext[d].ticker == tk:
            r = ext[d]
            f = fr[tk]
            px = float(r.exit_px)
            t = bar_time(f, d, px)
            if tk == park_ticker:
                target = w * (units * px + cash) / px  # SPY units to keep
                cash += (units - target) * px * sk
                spy = target
            else:
                cash += units * px * sk
                spx = float(sp.at[d, t])
                spy = w * cash / (spx * bc)
                cash -= w * cash
            units, tk = 0.0, None
        if not started and tk is None:  # first idle close: park w of the capital
            spx = float(sp.at[d, "close"])
            spy = w * cash / (spx * bc)
            cash -= w * cash
            started = True
        pv = spy * float(sp.at[d, "close"])
        tv = units * float(fr[tk].at[d, "close"]) if tk is not None else 0.0
        eq[i] = cash + pv + tv
        wts[i] = (pv / eq[i], tv / eq[i])
    s = pd.Series(eq, cal)
    s.attrs["weights"] = pd.DataFrame(wts, index=cal, columns=["park_w", "trade_w"])
    return s


def static_mix(cal: pd.DatetimeIndex, fr: dict[str, pd.DataFrame], cap: float, w: float, park_ticker: str,
               costs: tuple[float, float]) -> pd.Series:
    c = fr[park_ticker]["close"].reindex(cal)
    units = w * cap / (float(c.iloc[0]) * costs[0])
    return cap * (1 - w) + units * c


def to_daily(eq: pd.Series) -> pd.Series:
    prev = eq.shift(1)
    prev.iloc[0] = 1.0
    return eq / prev - 1


def row(daily: pd.Series, k: np.ndarray) -> dict:
    d = daily[k]
    return {"cagr": equity.cagr(d), "mdd": equity.mdd(d), "sharpe": equity.sharpe(d)}


def main() -> None:
    cfg = common.load_config()
    d = cfg["deep"]
    pt = d["park_ticker"]
    costs = cost_factors(cfg)
    _, fd = base.load(cfg, deep=True)
    lo = base.holdout_start(cfg)
    end_screen = common.screen_end(cfg)
    rows = []
    for t in cfg["screen"]["tickers"]:
        cal, _ = base.curve_inputs(fd[t], cfg)
        tr = base.rule_trades(fd[t], cfg).assign(ticker=t)
        masks = {"screen": np.asarray(cal <= end_screen), "holdout": np.asarray(cal >= lo),
                 "full": np.ones(len(cal), bool)}
        for w in d["park_w"]:
            eq = park_equity(cal, tr, fd, 1.0, w, pt, costs)
            wt = eq.attrs["weights"]
            spy_w = float((wt["park_w"] + (wt["trade_w"] if t == pt else 0)).mean())
            mix = static_mix(cal, fd, 1.0, spy_w, pt, costs)
            de, dm = to_daily(eq), to_daily(mix)
            for name, k in masks.items():
                rows.append({"ticker": t, "w": w, "stretch": name, **row(de, k),
                             **{f"mix_{a}": b for a, b in row(dm, k).items()}, "spy_w": spy_w,
                             "etf_w": float(wt["trade_w"].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(common.results_dir("deep") / "parking.csv", index=False)
    print(df[df["stretch"] == "full"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
