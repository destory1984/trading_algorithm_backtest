"""Stage 2: the fixed grid (config grid), one ticker per combo, one slot.

Per combo: trades, win rate, expectancy, payoff and profit factor, CAGR, max
drawdown, Sharpe (no risk-free rate, as the original), exposure, and the
break-even one-way cost: the cost c (buy at x(1+c), sell at x(1-c)) at which
the combo's CAGR equals buying and holding the benchmark over the same days.

Usage:  python -m src.grid
Writes  results/combos.csv, results/grid.csv, results/grid_daily.parquet,
        results/grid_trades.parquet, results/benchmark.csv
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from .common import load_config, results_dir, stats, trade_stats
from .strategy import arrays, calendar, daily_returns, equity, frames, in_position, run


def combos(cfg: dict) -> pd.DataFrame:
    g = cfg["grid"]
    rows = itertools.product(g["tickers"], g["entry"], g["exit"], g["exec"], g["trend"], g["max_hold"])
    df = pd.DataFrame(rows, columns=["ticker", "entry", "exit", "exec", "trend", "max_hold"])
    df.index.name = "combo"
    return df


def label(c) -> str:
    hold = "보유제한 없음" if pd.isna(c["max_hold"]) else f"최대 {int(c['max_hold'])}일"
    trend = "200일선 위" if c["trend"] == "ma200" else "필터 없음"
    return f"{c['ticker']} <{c['entry']:.2f} >{c['exit']:.2f} {c['exec']} {trend} {hold}"


def benchmark(cfg: dict, fr: dict[str, pd.DataFrame]) -> pd.Series:
    f = fr[cfg["grid"]["benchmark"]]
    cal = calendar(cfg, f)
    return f["close"].pct_change().reindex(cal).rename("benchmark")


def breakeven(gross: np.ndarray, target_factor: float) -> float:
    """One-way cost c with prod(1 + g) * ((1 - c) / (1 + c)) ** n == target_factor.
    Negative when the combo trails the benchmark even at zero cost."""
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
    bench = benchmark(cfg, fr)
    bench_factor = float((1 + bench).prod())
    cap = cfg["portfolio"]["initial_capital"]
    rows, daily, trades = [], {}, []
    t0 = time.time()
    for cid, c in cmb.iterrows():
        t = c["ticker"]
        hold = None if pd.isna(c["max_hold"]) else int(c["max_hold"])
        tr = run(ar[t], cfg, c["entry"], c["exit"], c["exec"], c["trend"], hold)
        tr = tr[tr["entry_date"] >= pd.Timestamp(cfg["data"]["start"])]
        eq, info = equity(cfg, fr[t], tr, t)
        d = daily_returns(eq, cap)
        pos = in_position(eq.index, tr)
        gross = (tr["exit_px"] / tr["entry_px"] - 1).to_numpy()
        rows.append({"combo": cid, **stats(d, pos), **trade_stats(tr["ret"].to_numpy()),
                     "gross_expectancy": float(gross.mean()) if len(gross) else np.nan,
                     "breakeven_bp": breakeven(gross, bench_factor) * 1e4,
                     "avg_hold": float(tr["hold_days"].mean()) if len(tr) else np.nan,
                     "taken": info["trades_taken"]})
        daily[cid] = d
        trades.append(tr.assign(combo=cid))
        if (cid + 1) % 96 == 0:
            print(f"  {cid + 1}/{len(cmb)} combos, {time.time() - t0:.0f}s")
    g = cmb.join(pd.DataFrame(rows).set_index("combo"))
    rd = results_dir()
    cmb.to_csv(rd / "combos.csv", encoding="utf-8-sig")
    g.to_csv(rd / "grid.csv", encoding="utf-8-sig")
    pd.DataFrame(daily).rename(columns=str).to_parquet(rd / "grid_daily.parquet")
    pd.concat(trades, ignore_index=True).to_parquet(rd / "grid_trades.parquet")
    b = stats(bench.fillna(0))
    pd.DataFrame([{"name": f"{cfg['grid']['benchmark']} 보유", **b}]).to_csv(rd / "benchmark.csv", index=False, encoding="utf-8-sig")
    print(f"{len(cmb)} combos in {time.time() - t0:.0f}s; benchmark CAGR {b['cagr']:.2%} MDD {b['mdd']:.1%} Sharpe {b['sharpe']:.2f}")
    top = g[g["trades"] >= 50].sort_values("sharpe", ascending=False).head(10)
    print(top[["ticker", "entry", "exit", "exec", "trend", "max_hold", "trades", "cagr", "mdd", "sharpe", "breakeven_bp"]].round(4).to_string())


if __name__ == "__main__":
    main()
