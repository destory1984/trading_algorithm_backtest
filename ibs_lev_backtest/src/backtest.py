"""Stage 2: TQQQ, SPXL, SOXL on their real listing days, fixed IBS 0.13 / 0.5.

Combos: ticker 3 x exec (open, close) 2 x sizing (full, vol target 0.2 / 0.3 / 0.4) 4
x cash (zero, ^IRX) 2 = 48, each with costs (commission + ticker slippage) and without = 96 rows.
No synthetic history: the dot-com and most of the 2008 crash are not in these windows.
Benchmarks per row, same days, same cash rule:
  hold  the leveraged ETF itself, bought at the first close, sold at the last (costs if the row has costs)
  u1x   the underlying 1x ETF (lev.underlying), same days
  k1x   the 1x ETF at a constant daily weight k, the rest in cash at the row's cash rate; k is
        chosen so its max drawdown equals the strategy's, capped at 1 (the spec's "줄인": scaled
        down, never levered). When the strategy's drawdown is deeper than the 1x ETF's own, k = 1
        and k1x_capped = True (marked in the report). No costs, no rebalancing costs.
Capital: tax.base_capital_krw / tax.fx. Pre-tax results do not depend on it.

Usage:  python -m src.backtest
Writes  results/lev.csv, results/lev_daily.parquet, results/lev_trades.parquet
"""
from __future__ import annotations

import itertools
import time
from typing import Iterator

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .common import capital_usd, daily_returns, load_config, results_dir, stats, ticker_costs, trade_stats
from .data import cash_factors, frames, irx
from .sizing import account, realized_vol, weights
from .strategy import calendar, trades


def combos(cfg: dict) -> pd.DataFrame:
    L = cfg["lev"]
    rows = itertools.product(L["tickers"], L["exec"], [str(s) for s in L["sizing"]], L["cash"], L["costs"])
    df = pd.DataFrame(rows, columns=["ticker", "exec", "sizing", "cash", "costs"])
    df.index.name = "combo"
    return df


def prepared(cfg: dict) -> Iterator[tuple[int, pd.Series, dict]]:
    L = cfg["lev"]
    fr = frames(cfg, list(dict.fromkeys(L["tickers"] + list(L["underlying"].values()))))
    rate = irx(cfg)
    vw, days, lag = cfg["indicators"]["vol_window"], cfg["cash"]["days_per_year"], cfg["cash"]["irx_lag"]
    cache: dict[tuple, pd.DataFrame] = {}
    for cid, c in combos(cfg).iterrows():
        t = c["ticker"]
        f = fr[t]
        cal = calendar(cfg, f)
        key = (t, c["exec"], c["costs"])
        if key not in cache:
            cache[key] = trades(cfg, f, t, c["exec"], c["costs"] == "with")
        tr = cache[key]
        target = None if c["sizing"] == "full" else float(c["sizing"])
        yield cid, c, {"f": f, "u": fr[L["underlying"][t]], "cal": cal, "tr": tr,
                       "w": weights(tr, realized_vol(f, vw), target),
                       "fac": cash_factors(cal, rate if c["cash"] == "irx" else None, lag, days)}


def hold_curve(close: pd.Series, cal: pd.DatetimeIndex, cap: float, buy_cost: float = 1.0,
               sell_keep: float = 1.0) -> pd.Series:
    c = close.reindex(cal).ffill()
    eq = cap / buy_cost * c / c.iloc[0]
    eq.iloc[-1] *= sell_keep
    return eq


def mix_curve(r1: np.ndarray, cash_r: np.ndarray, k: float, cap: float) -> np.ndarray:
    return cap * np.cumprod(1 + k * r1 + (1 - k) * cash_r)


def max_dd(eq) -> float:
    e = np.asarray(eq, np.float64)
    return float((e / np.maximum.accumulate(e) - 1).min())


def match_k(r1: np.ndarray, cash_r: np.ndarray, target_mdd: float) -> tuple[float, bool]:
    """Weight k in [0, 1] of the 1x ETF whose max drawdown equals target_mdd (bisection; deeper
    as k grows). If even k = 1 is shallower than the target, return (1.0, True): the cap binds."""
    if max_dd(mix_curve(r1, cash_r, 1.0, 1.0)) > target_mdd:
        return 1.0, True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if max_dd(mix_curve(r1, cash_r, mid, 1.0)) > target_mdd:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def benchmarks(cfg: dict, c: pd.Series, ctx: dict, cap: float, strat_mdd: float) -> dict:
    t, u = c["ticker"], cfg["lev"]["underlying"][c["ticker"]]
    cal, with_c = ctx["cal"], c["costs"] == "with"
    bt, st = cost_factors(ticker_costs(cfg, t, with_c))
    bu, su = cost_factors(ticker_costs(cfg, u, with_c))
    r1 = ctx["u"]["close"].reindex(cal).ffill().pct_change().fillna(0.0).to_numpy()
    cash_r = ctx["fac"] - 1.0
    k, capped = match_k(r1, cash_r, strat_mdd)
    return {"hold": hold_curve(ctx["f"]["close"], cal, cap, bt, st),
            "u1x": hold_curve(ctx["u"]["close"], cal, cap, bu, su),
            "k1x": pd.Series(mix_curve(r1, cash_r, k, cap), cal), "k": k, "k_capped": capped}


def main() -> None:
    cfg = load_config()
    cap = capital_usd(cfg)
    rows, daily, trs = [], {}, []
    t0 = time.time()
    for cid, c, ctx in prepared(cfg):
        eq, info = account(ctx["cal"], ctx["tr"], ctx["f"]["close"], cap, ctx["w"], ctx["fac"])
        d = daily_returns(eq, cap)
        s = stats(d, info["invested_share"])
        b = benchmarks(cfg, c, ctx, cap, s["mdd"])
        row = {"combo": cid, **c.to_dict(), **s, **trade_stats(ctx["tr"]["ret"].to_numpy()),
               "avg_weight": float(np.mean(ctx["w"])) if len(ctx["w"]) else np.nan, "taken": info["trades_taken"],
               "start": ctx["cal"][0].date(), "end": ctx["cal"][-1].date(), "k1x": b["k"],
               "k1x_capped": b["k_capped"]}
        for n in ("hold", "u1x", "k1x"):
            bs = stats(daily_returns(b[n], cap))
            row.update({f"{n}_cagr": bs["cagr"], f"{n}_sharpe": bs["sharpe"], f"{n}_mdd": bs["mdd"]})
        rows.append(row)
        daily[str(cid)] = d
        trs.append(ctx["tr"].assign(combo=cid, weight=ctx["w"]))
    g = pd.DataFrame(rows).set_index("combo")
    rd = results_dir()
    g.to_csv(rd / "lev.csv", encoding="utf-8-sig")
    pd.DataFrame(daily).to_parquet(rd / "lev_daily.parquet")
    pd.concat(trs, ignore_index=True).to_parquet(rd / "lev_trades.parquet")
    print(f"{len(g)} rows in {time.time() - t0:.0f}s")
    view = g[(g["costs"] == "with") & (g["cash"] == "irx")]
    print(view[["ticker", "exec", "sizing", "cagr", "sharpe", "mdd", "exposure", "trades", "u1x_sharpe", "k1x", "k1x_cagr"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()
