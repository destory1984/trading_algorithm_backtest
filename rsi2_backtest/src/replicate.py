"""Stage 0, our side: the original rule re-written as a plain loop on the original repo's CSV.

Written from the spec's rule text (the original has no license; nothing is copied).
Sample: rows of data/QQQ_D1.csv from the first day with both a 200-day SMA and an RSI(2) value (1999-12-21).
Split: the first int(0.7 x rows) rows are in-sample (IS), the rest out-of-sample (OOS); each part is backtested on its
own (a position open at the end of IS is valued at IS's last close); ALL is the whole sample.
Rule: cand on day s's close -> buy at s+1's open x 1.0005; exit condition on day d's close -> sell at d+1's open x
0.9995. A position still open on the last day is valued at the last close x 0.9995 and flagged (forced). The first row of
each part never gives a signal (the original's first signal day is the second row of its data).
Metrics as the original reports them: trade PF, win rate, expectancy; CAGR of the daily mark-to-market curve over
calendar days; daily max drawdown; exposure = share of days whose daily return is not zero.
Variant no_signal_day (skip=1) ignores the exit condition on the entry day, so both green days fall after the signal day.

simulate() is also the reference loop for the engine checks (checks.engine_match, checks.grid_vs_loop).

Usage:  python -m src.replicate
Writes  results/replicate.csv, results/replicate_trades.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .common import ROOT, cost_set, load_config, results_dir, stats, trade_stats
from .indicators import rsi_wilder, signals, sma

PARTS = ("IS", "OOS", "ALL")
VARIANTS = {"original": 0, "no_signal_day": 1}


def load_csv(cfg: dict) -> pd.DataFrame:
    path = (ROOT / cfg["original"]["ref_dir"] / cfg["original"]["csv"]).resolve()
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close"]].astype(float)
    df.columns = ["open", "high", "low", "close"]
    df.index.name = "date"
    return df


def with_indicators(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi_wilder(df["close"], cfg["rsi"]["period"])
    out["ma"] = sma(df["close"], cfg["original"]["sma"])
    out["sma5"] = sma(df["close"], cfg["sma_exit"])
    out["green"] = df["close"] > df["open"]
    return out


def sample(ind: pd.DataFrame) -> pd.DataFrame:
    return ind[ind["ma"].notna() & ind["rsi"].notna()]


def parts(cfg: dict, smp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    k = int(len(smp) * cfg["original"]["is_frac"])
    return {"IS": smp.iloc[:k], "OOS": smp.iloc[k:], "ALL": smp}


def simulate(o: np.ndarray, c: np.ndarray, cand: np.ndarray, exit_cond: np.ndarray, mode: str,
             buy_cost: float, sell_keep: float, skip: int = 0) -> list[tuple]:
    """One position at a time, bar indices. Returns (s, e, x, entry_px, exit_px, ret, forced) per trade.

    open   buy at o[s+1]. The exit condition counts from bar e + skip; true on bar d -> sell at o[d+1].
           The next signal may be the exit bar itself (the position is gone after that open).
    close  buy at c[s] (e = s). The exit condition counts from bar e + max(skip, 1); true on bar d -> sell at c[d].
           The next signal comes from bar d + 1.
    A position without an exit before the data ends is sold at the last close (forced). Prices: x buy_cost in,
    x sell_keep out."""
    n = len(c)
    out = []
    s = 0
    while s < n:
        if not cand[s]:
            s += 1
            continue
        if mode == "open":
            e = s + 1
            if e >= n:
                break
            epx = o[e]
            d = e + skip
        elif mode == "close":
            e, epx = s, c[s]
            d = e + max(skip, 1)
        else:
            raise ValueError(mode)
        while d < n and not exit_cond[d]:
            d += 1
        if mode == "open":
            forced = d >= n - 1
            x = n - 1 if forced else d + 1
            xpx = c[x] if forced else o[x]
            nxt = x
        else:
            forced = d >= n
            x = n - 1 if forced else d
            xpx = c[x]
            nxt = x + 1
        out.append((s, e, x, float(epx), float(xpx), float(xpx * sell_keep / (epx * buy_cost) - 1.0), bool(forced)))
        if forced:
            break
        s = nxt
    return out


def loop_trades(dates, o, c, cand, exit_cond, mode: str, costs: dict, skip: int = 0) -> pd.DataFrame:
    buy_cost, sell_keep = cost_factors({"costs": costs})
    rows = simulate(np.asarray(o, np.float64), np.asarray(c, np.float64), cand, exit_cond, mode, buy_cost, sell_keep, skip)
    df = pd.DataFrame(rows, columns=["s", "e", "x", "entry_px", "exit_px", "ret", "forced"])
    dates = pd.DatetimeIndex(dates)
    return df.assign(signal_date=dates[df["s"].to_numpy(int)], entry_date=dates[df["e"].to_numpy(int)],
                     exit_date=dates[df["x"].to_numpy(int)], bars=(df["x"] - df["e"]).astype(int))


def daily_curve(c: np.ndarray, tr: pd.DataFrame, buy_cost: float, sell_keep: float) -> np.ndarray:
    """Daily mark-to-market return, 0 when flat. Entry day: close / (entry price x buy_cost) - 1; days in between:
    close / previous close - 1; exit day: exit price x sell_keep / previous close - 1; a trade that enters and ends on
    the same bar: its whole return that day."""
    r = np.zeros(len(c))
    for e, x, epx, xpx in zip(tr["e"], tr["x"], tr["entry_px"], tr["exit_px"]):
        paid, got = epx * buy_cost, xpx * sell_keep
        if x == e:
            r[e] = got / paid - 1
            continue
        r[e] = c[e] / paid - 1
        r[e + 1:x] = c[e + 1:x] / c[e:x - 1] - 1
        r[x] = got / c[x - 1] - 1
    return r


def run_part(cfg: dict, part: pd.DataFrame, skip: int, costs_name: str = "orig", thr: float | None = None,
             trend: str = "above", exit: str = "green2", mode: str = "open") -> tuple[pd.DataFrame, pd.Series]:
    thr = cfg["original"]["rsi_thr"] if thr is None else thr
    ind = {k: part[k].to_numpy(np.float64) for k in ("rsi", "close", "ma", "sma5")}
    ind["green"] = part["green"].to_numpy(bool)
    eligible = np.ones(len(part), bool)
    eligible[0] = False  # the first row of a part never signals
    cand, cond = signals(ind, thr, trend, exit, eligible, cfg["rsi"]["exit_above"])
    costs = cost_set(cfg, costs_name)
    tr = loop_trades(part.index, part["open"].to_numpy(np.float64), ind["close"], cand, cond, mode, costs, skip)
    buy_cost, sell_keep = cost_factors({"costs": costs})
    return tr, pd.Series(daily_curve(ind["close"], tr, buy_cost, sell_keep), part.index)


def report_metrics(tr: pd.DataFrame, daily: pd.Series) -> dict:
    s = stats(daily)
    return {**trade_stats(tr["ret"].to_numpy()), "cagr": s["cagr"], "mdd": s["mdd"], "sharpe": s["sharpe"],
            "exposure": float((daily != 0).mean()),
            # the exit condition held on the entry day's close: with green2 that used the signal day's candle
            "signal_day_exits": int(((tr["x"] - tr["e"] == 1) & ~tr["forced"].astype(bool)).sum()),
            "open_at_end": int(tr["forced"].astype(bool).sum())}


def compare_trades(ref: pd.DataFrame, eng: pd.DataFrame) -> tuple[int, float]:
    """(trades whose signal, entry or exit date or forced flag differ + the difference in count, largest return gap).
    ref: loop_trades output; eng: krxbt.engine.run output."""
    n = min(len(ref), len(eng))
    cols = ["signal_date", "entry_date", "exit_date"]
    same = (ref[cols].iloc[:n].to_numpy() == eng[cols].iloc[:n].to_numpy()).all(axis=1)
    same &= ref["forced"].to_numpy(bool)[:n] == (eng["reason"] == "forced").to_numpy()[:n]
    diff = float(np.abs(ref["ret"].to_numpy(np.float64)[:n] - eng["ret"].to_numpy(np.float64)[:n]).max()) if n else 0.0
    return int((~same).sum()) + abs(len(ref) - len(eng)), diff


def main() -> None:
    cfg = load_config()
    smp = sample(with_indicators(cfg, load_csv(cfg)))
    rows, trades = [], []
    for var, skip in VARIANTS.items():
        for name, part in parts(cfg, smp).items():
            tr, daily = run_part(cfg, part, skip)
            rows.append({"variant": var, "part": name, "first": part.index[0].date(), "last": part.index[-1].date(),
                         "days": len(part), **report_metrics(tr, daily)})
            trades.append(tr.assign(variant=var, part=name))
    out = pd.DataFrame(rows)
    rd = results_dir()
    out.to_csv(rd / "replicate.csv", index=False, encoding="utf-8-sig")
    pd.concat(trades, ignore_index=True).to_csv(rd / "replicate_trades.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["variant", "part", "first", "last", "days", "trades", "win_rate", "profit_factor", "expectancy", "cagr",
               "mdd", "exposure", "signal_day_exits"]].round(4).to_string())


if __name__ == "__main__":
    main()
