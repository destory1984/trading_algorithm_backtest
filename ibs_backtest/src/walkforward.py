"""Stage 3: walk-forward. Pick the combo with the past only, trade it the next year.

For every test year Y (walkforward.first_test_year .. last year):
  train = each combo's daily returns from data.start to Y-1 (expanding)
  pick  = highest train Sharpe among combos with >= min_train_trades trades
          entered before Y
  test  = the pick's trades entered in year Y
The picks' trades are run as one account with one slot: in year Y only the
pick of Y may open a position; a position already open keeps running and
blocks new entries until it exits. Compared with holding the benchmark and
with the combo that has the best Sharpe over the whole period (hindsight).

Run three times: all combos, close-execution combos only, open-execution
combos only (the decision rule asks for both).

Usage:  python -m src.walkforward
Writes  results/walkforward.csv (picks), results/walkforward_summary.csv,
        results/walkforward_equity.csv, results/walkforward_trades.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import portfolio as kp

from .common import load_config, results_dir, stats, trade_stats
from .grid import benchmark, breakeven, label
from .strategy import frames

VARIANTS = {"all": None, "close": "close", "open": "open"}


def sharpe(d: pd.DataFrame) -> pd.Series:
    sd = d.std()
    return (d.mean() * 252 / (sd * np.sqrt(252))).where(sd > 0, 0.0)


def pick_years(g: pd.DataFrame, daily: pd.DataFrame, tr: pd.DataFrame, years: list[int], min_trades: int,
               allowed: pd.Index) -> pd.DataFrame:
    rows = []
    for Y in years:
        start = pd.Timestamp(f"{Y}-01-01")
        end = pd.Timestamp(f"{Y + 1}-01-01")
        d_tr = daily.loc[daily.index < start, allowed]
        d_te = daily.loc[(daily.index >= start) & (daily.index < end), allowed]
        n_tr = tr[tr["entry_date"] < start].groupby("combo").size().reindex(allowed, fill_value=0)
        s_tr = sharpe(d_tr)[n_tr >= min_trades].sort_values(ascending=False, kind="mergesort")  # ties: lower combo id
        if s_tr.empty:
            continue
        pick = s_tr.index[0]
        s_te = sharpe(d_te)
        rows.append({"year": Y, "combo": int(pick), "label": label(g.loc[pick]), "train_sharpe": s_tr.iloc[0],
                     "train_trades": int(n_tr[pick]), "test_sharpe": s_te[pick],
                     "test_return": float((1 + d_te[pick]).prod() - 1),
                     "test_pctile": float((s_te < s_te[pick]).mean()), "median_test_sharpe": float(s_te.median())})
    return pd.DataFrame(rows)


def stitched(picks: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    parts = [tr[(tr["combo"] == p["combo"]) & (tr["entry_date"].dt.year == p["year"])] for _, p in picks.iterrows()]
    return pd.concat(parts, ignore_index=True).sort_values("entry_date", ignore_index=True)


def account(cal: pd.DatetimeIndex, trades: pd.DataFrame, closes: pd.DataFrame, cap: float) -> tuple[pd.Series, dict]:
    return kp.run_portfolio(cal, trades, closes, cap, slots=1, rank_by="signal_date")


def summarize(name: str, eq: pd.Series, cap: float, bench_factor: float, taken: pd.DataFrame | None,
              info: dict | None = None) -> dict:
    prev = eq.shift(1)
    prev.iloc[0] = cap
    d = eq / prev - 1
    row = {"name": name, **stats(d)}
    if info is not None:
        row["exposure"] = info["invested_share"]  # days ending with a position open
    if taken is not None:
        row.update(trade_stats(taken["ret"].to_numpy()))
        gross = (taken["exit_px"] / taken["entry_px"] - 1).to_numpy()
        row["breakeven_bp"] = breakeven(gross, bench_factor) * 1e4
    return row


def taken_trades(eq: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    led = eq.attrs["ledger"]
    if led.empty:
        return cand.iloc[:0]
    return led[["ticker", "entry_date"]].merge(cand, on=["ticker", "entry_date"], how="left")


def main() -> None:
    cfg = load_config()
    wf = cfg["walkforward"]
    cap = cfg["portfolio"]["initial_capital"]
    rd = results_dir()
    g = pd.read_csv(rd / "grid.csv", index_col="combo")
    daily = pd.read_parquet(rd / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr["ticker"] = g["ticker"].reindex(tr["combo"]).to_numpy()
    fr = frames(cfg, cfg["grid"]["tickers"])
    start = pd.Timestamp(f"{wf['first_test_year']}-01-01")
    cal = daily.index[daily.index >= start]
    closes = pd.DataFrame({t: f["close"] for t, f in fr.items()}).reindex(cal).ffill()
    bench = benchmark(cfg, fr).reindex(cal)
    bench_factor = float((1 + bench).prod())
    years = list(range(wf["first_test_year"], int(daily.index[-1].year) + 1))

    all_picks, summary, curves, st_trades = [], [], {}, []
    for var, mode in VARIANTS.items():
        allowed = g.index if mode is None else g.index[g["exec"] == mode]
        picks = pick_years(g, daily, tr, years, wf["min_train_trades"], allowed)
        picks.insert(0, "variant", var)
        all_picks.append(picks)
        cand = stitched(picks, tr)
        eq, info = account(cal, cand, closes, cap)
        taken = taken_trades(eq, cand)
        summary.append({"variant": var, **summarize("걸어가며", eq, cap, bench_factor, taken, info),
                        "skipped": info["trades_available"] - info["trades_taken"],
                        "distinct_picks": picks["combo"].nunique(),
                        "changes": int((picks["combo"] != picks["combo"].shift()).sum() - 1)})
        curves[f"wf_{var}"] = eq
        st_trades.append(taken.assign(variant=var))
        # hindsight: best full-period Sharpe with enough trades, same rule, same window
        ok = g.loc[allowed]
        best = int(ok[ok["trades"] >= wf["min_train_trades"]]["sharpe"].idxmax())
        hb = tr[(tr["combo"] == best) & (tr["entry_date"] >= start)]
        eq_hb, info_hb = account(cal, hb, closes, cap)
        summary.append({"variant": var, **summarize(f"뒤늦게 고른 1등: {label(g.loc[best])}", eq_hb, cap, bench_factor,
                                                    taken_trades(eq_hb, hb), info_hb), "combo": best})
        curves[f"hindsight_{var}"] = eq_hb
    bh = (1 + bench.fillna(0)).cumprod() * cap
    summary.append({"variant": "-", **summarize(f"{cfg['grid']['benchmark']} 보유", bh, cap, bench_factor, None)})
    curves["benchmark"] = bh
    picks = pd.concat(all_picks, ignore_index=True)
    picks.to_csv(rd / "walkforward.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(rd / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "walkforward_equity.csv")
    pd.concat(st_trades, ignore_index=True).to_parquet(rd / "walkforward_trades.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(picks[["variant", "year", "label", "train_sharpe", "test_sharpe", "test_return"]].round(3).to_string())
    print(sm[["variant", "name", "cagr", "mdd", "sharpe", "trades", "breakeven_bp", "exposure"]].round(4).to_string())


if __name__ == "__main__":
    main()
