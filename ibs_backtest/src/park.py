"""Hold SPY at weight w while idle, go all in on an IBS signal.

Trades are the walk-forward account's trades (results/walkforward_trades.parquet),
so entries and exits do not change. Bookkeeping is in units:
  idle         w of equity in SPY, the rest cash at 0%, no rebalancing
  entry        signal on SPY: the cash buys more SPY (the parked SPY is kept);
               other ETF: all SPY is sold, everything buys the ETF
  exit         back to w of equity in SPY (on SPY: only the excess is sold)
Every trade happens at the bar of the trade's own price (open or close),
found by matching entry_px / exit_px to the adjusted open or close.
Costs (costs section) apply only to amounts actually bought or sold.

Usage:  python -m src.park     (after src.walkforward)
Writes  results/park.csv, results/park_equity.csv
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .common import load_config, results_dir, stats
from .strategy import frames


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
    book = np.zeros((len(cal), 3))
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
        book[i] = (cash, pv, tv)
        wts[i] = (pv / eq[i], tv / eq[i])
    s = pd.Series(eq, cal)
    s.attrs["book"] = pd.DataFrame(book, index=cal, columns=["cash", "park_value", "trade_value"])
    s.attrs["weights"] = pd.DataFrame(wts, index=cal, columns=["park_w", "trade_w"])
    return s


def _conserve_idle(cal: pd.DatetimeIndex, t: pd.DataFrame, fr: dict[str, pd.DataFrame], cap: float, w: float,
                   park_ticker: str) -> float:
    """Real conservation check for the idle path: park in a ticker whose price never moves
    (1.0 every day), no cost. A zero-return, free park asset is bookkeeping-equivalent to cash,
    so this curve must equal the true w = 0 curve at any w -- both taken at zero cost, so the
    comparison isolates the parking bookkeeping itself from the trading cost (checked separately
    by w0_diff, which uses real costs)."""
    const = pd.DataFrame({"open": 1.0, "close": 1.0}, index=cal)
    fr2 = dict(fr)
    fr2["_CONST"] = const
    w0z = park_equity(cal, t, fr, cap, 0.0, park_ticker, (1.0, 1.0))
    eq = park_equity(cal, t, fr2, cap, w, "_CONST", (1.0, 1.0))
    return float((eq - w0z).abs().max())


def _conserve_switch(cal: pd.DatetimeIndex, t: pd.DataFrame, fr: dict[str, pd.DataFrame], cap: float, w: float,
                     park_ticker: str) -> float | None:
    """Real conservation check for the SPY-signal path: with no cost, keeping the already-held
    w share of SPY and buying only the missing (1 - w) on an SPY signal must be worth exactly
    the same as fully switching, at that instant, into a copy of SPY under another name (a
    full sell of the park sleeve, then a full buy of the signal ticker). None if the trade list
    has no trade on park_ticker, since the optimization never triggers."""
    if not (t["ticker"] == park_ticker).any():
        return None
    fr2 = dict(fr)
    fr2["_TWIN"] = fr[park_ticker]
    a = park_equity(cal, t, fr2, cap, w, park_ticker, (1.0, 1.0))
    b = park_equity(cal, t, fr2, cap, w, "_TWIN", (1.0, 1.0))
    return float((a - b).abs().max())


def static_mix(cal: pd.DatetimeIndex, fr: dict[str, pd.DataFrame], cap: float, w: float, park_ticker: str,
               costs: tuple[float, float]) -> pd.Series:
    c = fr[park_ticker]["close"].reindex(cal)
    units = w * cap / (float(c.iloc[0]) * costs[0])
    return cap * (1 - w) + units * c


def _stats(eq: pd.Series, cap: float) -> dict:
    prev = eq.shift(1)
    prev.iloc[0] = cap
    st = stats(eq / prev - 1)
    st.pop("exposure")
    return st


def main() -> None:
    cfg = load_config()
    pk = cfg["park"]
    cap = cfg["portfolio"]["initial_capital"]
    rd = results_dir()
    costs = cost_factors(cfg)
    fr = frames(cfg, cfg["grid"]["tickers"])
    wfe = pd.read_csv(rd / "walkforward_equity.csv", index_col=0, parse_dates=True)
    cal = pd.DatetimeIndex(wfe.index)
    fr = {t: f.reindex(cal).ffill() for t, f in fr.items()}
    tr = pd.read_parquet(rd / "walkforward_trades.parquet")
    rows, curves = [], {}
    for var, t in tr.groupby("variant", sort=False):
        t = t.sort_values("entry_date", ignore_index=True)
        base = park_equity(cal, t, fr, cap, 0.0, pk["ticker"], costs)
        idle = base.attrs["weights"]["trade_w"] == 0
        w0_diff = float((base - wfe[f"wf_{var}"])[idle].abs().max())
        spy_only = t[t["ticker"] == pk["ticker"]]
        full = park_equity(cal, spy_only, fr, cap, 1.0, pk["ticker"], costs)
        bh = cap / costs[0] * fr[pk["ticker"]]["close"] / float(fr[pk["ticker"]]["close"].iloc[0])
        w1_diff = float((full - bh).abs().max())
        for w in [0.0] + list(pk["weights"]):
            eq = base if w == 0 else park_equity(cal, t, fr, cap, w, pk["ticker"], costs)
            b, wt = eq.attrs["book"], eq.attrs["weights"]
            conserve_idle = _conserve_idle(cal, t, fr, cap, w, pk["ticker"]) if w else np.nan
            conserve_switch = _conserve_switch(cal, t, fr, cap, w, pk["ticker"]) if w else np.nan
            rows.append({"variant": var, "name": "IBS + SPY 대기" if w else "IBS 만 (현금 대기)", "w": w,
                         **_stats(eq, cap), "trades": len(t),
                         "avg_spy_w": float(wt["park_w"].mean()), "avg_etf_w": float(wt["trade_w"].mean()),
                         "ledger_diff": float((b.sum(axis=1) - eq).abs().max()),
                         "min_cash": float(b["cash"].min()),
                         "w0_diff": w0_diff if w == 0 else np.nan, "w1_diff": w1_diff if w == 0 else np.nan,
                         "spy_only_trades": len(spy_only) if w == 0 else np.nan,
                         "conserve_idle_diff": conserve_idle, "conserve_switch_diff": conserve_switch})
            curves[f"{var}_w{w:.1f}"] = eq
    for w in pk["weights"]:
        eq = static_mix(cal, fr, cap, w, pk["ticker"], costs)
        rows.append({"variant": "-", "name": f"{pk['ticker']} {w:.0%} + 현금 고정", "w": w, **_stats(eq, cap),
                     "avg_spy_w": float((eq - cap * (1 - w)).div(eq).mean())})
        curves[f"static_w{w:.1f}"] = eq
    bh = cap * fr[pk["ticker"]]["close"] / float(fr[pk["ticker"]]["close"].iloc[0])
    rows.append({"variant": "-", "name": f"{pk['ticker']} 보유", "w": 1.0, **_stats(bh, cap), "avg_spy_w": 1.0})
    curves[f"{pk['ticker']} 보유"] = bh
    out = pd.DataFrame(rows)
    out.to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")
    pd.set_option("display.width", 250)
    print(out[["variant", "name", "w", "cagr", "mdd", "sharpe", "avg_spy_w", "avg_etf_w"]].round(4).to_string())


if __name__ == "__main__":
    main()
