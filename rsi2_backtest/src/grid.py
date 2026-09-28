"""Stage 2: the fixed grid. 4 tickers x RSI(2) threshold 3 x trend filter 3 x exit 3 x execution 2 = 216 combos, each
with the us costs (commission 0.07% + slippage 0.05% one way) and the original costs (5bp one way): 432 rows.

Per row: trade stats (trades, win rate, PF, PF without the largest trade, expectancy, payoff), daily-curve stats
(CAGR, daily max drawdown, Sharpe without a risk-free rate, exposure = share of days whose return depends on a
position), break-even one-way cost (the cost c at which prod(1 + gross) x ((1 - c) / (1 + c)) ** n = 1, i.e. the net
compounded return is zero), open_at_end (the last trade is still open on data.end, valued at that close), and the
three stage-3 benchmarks of the same ticker (benchmarks.py).
Equity: krxbt.portfolio.run_portfolio(slots=1) from data.start.

Usage:  python -m src.grid
Writes  results/combos.csv, results/grid.csv, results/grid_daily.parquet (us costs), results/grid_trades.parquet
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .benchmarks import bench_row, hold_curve, matched
from .common import daily_returns, in_position, load_config, results_dir, stats, trade_stats, with_costs
from .strategy import arrays, calendar, equity, frames, rule, run_rule

TREND = {"above": "200일선 위", "none": "필터 없음", "below": "200일선 아래"}
EXIT = {"green2": "2연속 양봉", "sma5": "종가>5일선", "rsi70": "RSI>70"}


def combos(cfg: dict) -> pd.DataFrame:
    g = cfg["grid"]
    rows = itertools.product(g["tickers"], g["rsi_thr"], g["trend"], g["exit"], g["exec"])
    df = pd.DataFrame(rows, columns=["ticker", "rsi_thr", "trend", "exit", "exec"])
    df.index.name = "combo"
    return df


def label(c) -> str:
    return f"{c['ticker']} RSI<{int(c['rsi_thr'])} {TREND[c['trend']]} {EXIT[c['exit']]} {c['exec']}"


def breakeven(gross: np.ndarray, target_factor: float = 1.0) -> float:
    """One-way cost c with prod(1 + g) * ((1 - c) / (1 + c)) ** n == target_factor. Negative when even zero cost
    does not reach it."""
    n = len(gross)
    if n == 0:
        return np.nan
    k = np.exp((np.log(target_factor) - np.log1p(gross).sum()) / n)
    return (1 - k) / (1 + k)


def main() -> None:
    cfg = load_config()
    cmb = combos(cfg)
    fr = frames(cfg, cfg["grid"]["tickers"])
    ar = {t: arrays(f) for t, f in fr.items()}
    cap = cfg["portfolio"]["initial_capital"]
    rows, daily, trades = [], {}, []
    t0 = time.time()
    for cost in cfg["grid"]["costs"]:
        ccfg = with_costs(cfg, cost)
        bc, sk = cost_factors(ccfg)
        holds = {}
        for t, f in fr.items():
            cal = calendar(cfg, f)
            r1 = f["close"].reindex(cal).ffill().pct_change().fillna(0.0).to_numpy()
            holds[t] = (r1, hold_curve(f["close"], cal, cap, bc, sk))
        for cid, c in cmb.iterrows():
            t = c["ticker"]
            cand, target = rule(ar[t], cfg, c["rsi_thr"], c["trend"], c["exit"])
            tr = run_rule(ar[t], ccfg, cand, target, c["exec"])
            eq, info = equity(cfg, fr[t], tr, t)
            d = daily_returns(eq, cap)
            s = stats(d, float(in_position(eq.index, tr).mean()))
            gross = (tr["exit_px"] / tr["entry_px"] - 1).to_numpy()
            r1, hold = holds[t]
            rows.append({"combo": cid, **c.to_dict(), "costs": cost, **trade_stats(tr["ret"].to_numpy()), **s,
                         "gross_expectancy": float(gross.mean()) if len(gross) else np.nan,
                         "breakeven_bp": breakeven(gross) * 1e4,
                         "avg_hold": float(tr["hold_days"].mean()) if len(tr) else np.nan,
                         "taken": info["trades_taken"], "open_at_end": bool((tr["reason"] == "forced").any()),
                         **bench_row(matched(r1, hold, cap, s["mdd"], s["exposure"]), cap)})
            if cost == "us":
                daily[str(cid)] = d
            trades.append(tr.assign(combo=cid, costs=cost))
        print(f"  costs {cost}: {len(cmb)} combos, {time.time() - t0:.0f}s")
    g = pd.DataFrame(rows)
    rd = results_dir()
    cmb.to_csv(rd / "combos.csv", encoding="utf-8-sig")
    g.to_csv(rd / "grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily).to_parquet(rd / "grid_daily.parquet")
    pd.concat(trades, ignore_index=True).to_parquet(rd / "grid_trades.parquet")
    u = g[(g["costs"] == "us") & (g["trades"] >= cfg["walkforward"]["min_train_trades"])]
    pd.set_option("display.width", 250)
    print(u.sort_values("sharpe", ascending=False).head(10)[
        ["ticker", "rsi_thr", "trend", "exit", "exec", "trades", "win_rate", "profit_factor", "cagr", "mdd", "sharpe",
         "exposure", "hold_cagr", "kdd_cagr", "kexp_cagr"]].round(3).to_string())
    print(g[g["costs"] == "us"].groupby("trend")[["sharpe", "cagr", "trades"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
