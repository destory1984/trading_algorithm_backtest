"""Stage 4: volatility-target sizing on the walk-forward trades.

At entry, weight = min(1, target / realized vol), realized vol = standard
deviation of the ticker's last vol_window close-to-close returns up to the
signal day, x sqrt(252). The rest stays in cash (return 0). No leverage.
The trades are the ones the stage-3 account actually took, so entries and
exits do not change, only the size.

Usage:  python -m src.volsize        (after src.walkforward)
Writes  results/volsize.csv, results/volsize_equity.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import load_config, results_dir, stats
from .grid import benchmark
from .strategy import frames


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252)


def sized_equity(cal: pd.DatetimeIndex, trades: pd.DataFrame, closes: pd.DataFrame, cap: float,
                 weight: np.ndarray) -> pd.Series:
    """One position at a time, alloc = weight x equity at the previous close.
    Same bookkeeping as krxbt.portfolio.run_portfolio(slots=1): enter at the
    open of entry_date, mark to market at each close at close/entry_px, book
    alloc x (1 + ret) on exit_date."""
    by_entry = {d: i for i, d in enumerate(trades["entry_date"])}
    cash, prev_eq, pos = float(cap), float(cap), None
    eq = np.empty(len(cal))
    for i, d in enumerate(cal):
        if pos is None and d in by_entry:
            j = by_entry[d]
            t = trades.iloc[j]
            alloc = min(prev_eq * float(weight[j]), cash)
            cash -= alloc
            pos = {"tk": t["ticker"], "alloc": alloc, "px": float(t["entry_px"]), "exit": t["exit_date"],
                   "ret": float(t["ret"])}
        if pos is not None and pos["exit"] <= d:
            cash += pos["alloc"] * (1 + pos["ret"])
            pos = None
        val = 0.0 if pos is None else pos["alloc"] * closes.at[d, pos["tk"]] / pos["px"]
        prev_eq = cash + val
        eq[i] = prev_eq
    return pd.Series(eq, cal)


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    cap = cfg["portfolio"]["initial_capital"]
    fr = frames(cfg, cfg["grid"]["tickers"])
    wfe = pd.read_csv(rd / "walkforward_equity.csv", index_col=0, parse_dates=True)
    cal = pd.DatetimeIndex(wfe.index)
    closes = pd.DataFrame({t: f["close"] for t, f in fr.items()}).reindex(cal).ffill()
    vol = {t: realized_vol(f["close"], cfg["indicators"]["vol_window"]) for t, f in fr.items()}
    tr = pd.read_parquet(rd / "walkforward_trades.parquet")
    bench = benchmark(cfg, fr).reindex(cal).fillna(0)
    rows, curves = [], {}
    for var, t in tr.groupby("variant", sort=False):
        t = t.sort_values("entry_date", ignore_index=True)
        v = np.array([vol[k].get(s, np.nan) for k, s in zip(t["ticker"], t["signal_date"])])
        base = sized_equity(cal, t, closes, cap, np.ones(len(t)))
        diff = float((base - wfe[f"wf_{var}"]).abs().max())
        for tv in [None] + list(cfg["vol_target"]["targets"]):
            w = np.ones(len(t)) if tv is None else np.minimum(1.0, tv / v)
            eq = base if tv is None else sized_equity(cal, t, closes, cap, w)
            prev = eq.shift(1)
            prev.iloc[0] = cap
            name = "늘 100%" if tv is None else f"목표 {tv:.0%}"
            st = stats(eq / prev - 1)
            st.pop("exposure")
            rows.append({"variant": var, "sizing": name, **st, "avg_weight": float(np.mean(w)),
                         "check_vs_stage3": diff if tv is None else np.nan})
            curves[f"{var}_{name}"] = eq
    b = stats(bench)
    b.pop("exposure")
    rows.append({"variant": "-", "sizing": f"{cfg['grid']['benchmark']} 보유", **b})
    out = pd.DataFrame(rows)
    out.to_csv(rd / "volsize.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "volsize_equity.csv")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
