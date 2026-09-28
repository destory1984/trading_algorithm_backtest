"""Stage 4: walk-forward. Pick the combo with the past only, trade it the next year (as work order 1).

For every test year Y (walkforward.first_test_year .. data.end's year):
  train = each combo's daily returns (us costs) from data.start to Y-1 (expanding)
  pick  = highest train Sharpe among combos with >= min_train_trades trades entered before Y (ties: lower combo id)
  test  = the pick's trades entered in year Y
A year with no combo reaching min_train_trades is skipped (no pick, no row) and listed explicitly below and in
walkforward_summary.csv's "wf" row (skipped_years) -- never silently dropped.
The picks' trades run as one account with one slot (run_portfolio): a position that crosses into the next year keeps
running and blocks that year's pick until it exits.
Benchmarks: holding each year's picked ticker ("hold", stitched, no switching cost), and that stitched hold scaled to
the walk-forward curve's drawdown (kdd) and exposure (kexp); SPY and QQQ holds for context; the combo with the best
full-period Sharpe (hindsight).
Three variants: all combos, open-execution combos, close-execution combos. candidates.csv holds each variant's pick for
the last year: the one combo to use from now on, used by stages 5 and 6.
The original's "3-year train / 1-year test rolling window" keeps the rule fixed and only moves the window; it is not a
walk-forward selection (stated in the report).

Usage:  python -m src.walkforward      (after src.grid)
Writes  results/walkforward.csv, results/walkforward_summary.csv, results/walkforward_equity.csv,
        results/walkforward_trades.parquet, results/candidates.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import portfolio as kp

from .benchmarks import matched
from .common import daily_returns, in_position, load_config, results_dir, stats, trade_stats
from .grid import label
from .strategy import frames

WF_VARIANTS = {"all": None, "open": "open", "close": "close"}


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
        s_tr = sharpe(d_tr)[n_tr >= min_trades].sort_values(ascending=False, kind="mergesort")
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


def taken_trades(eq: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    led = eq.attrs["ledger"]
    if led.empty:
        return cand.iloc[:0]
    return led[["ticker", "entry_date"]].merge(cand, on=["ticker", "entry_date"], how="left")


def stitched_hold(picks: pd.DataFrame, g: pd.DataFrame, rets: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Daily return of holding, in each test year, the ticker picked for that year; 0 on the first day."""
    r = pd.Series(0.0, index=cal)
    for p in picks.itertuples():
        m = cal.year == p.year
        r[m] = rets.loc[cal[m], g.at[p.combo, "ticker"]].to_numpy()
    r.iloc[0] = 0.0
    return r


def summarize(name: str, lab: str, eq: pd.Series, cap: float, taken: pd.DataFrame | None = None,
              exposure: float | None = None) -> dict:
    row = {"name": name, "label": lab, **stats(daily_returns(eq, cap), exposure)}
    if taken is not None:
        row.update(trade_stats(taken["ret"].to_numpy()))
    return row


def main() -> None:
    cfg = load_config()
    wf = cfg["walkforward"]
    cap = cfg["portfolio"]["initial_capital"]
    rd = results_dir()
    g = pd.read_csv(rd / "grid.csv")
    g = g[g["costs"] == "us"].set_index("combo")
    daily = pd.read_parquet(rd / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"].reset_index(drop=True)
    tr["ticker"] = g["ticker"].reindex(tr["combo"]).to_numpy()
    fr = frames(cfg, cfg["grid"]["tickers"])
    start = pd.Timestamp(f"{wf['first_test_year']}-01-01")
    cal = daily.index[daily.index >= start]
    closes = pd.DataFrame({t: f["close"] for t, f in fr.items()}).reindex(cal).ffill()
    rets = pd.DataFrame({t: f["close"].pct_change() for t, f in fr.items()}).reindex(cal).fillna(0.0)
    years = list(range(wf["first_test_year"], int(cal[-1].year) + 1))
    all_picks, summary, curves, taken_all, cands = [], [], {}, [], []
    skip_lines = []
    for var, mode in WF_VARIANTS.items():
        allowed = g.index if mode is None else g.index[g["exec"] == mode]
        picks = pick_years(g, daily, tr, years, wf["min_train_trades"], allowed)
        if picks.empty:
            raise RuntimeError(f"no pick for variant {var}")
        skipped_years = sorted(set(years) - set(picks["year"].tolist()))
        skip_lines.append(f"{var}: {len(skipped_years)} skipped year(s) of {len(years)} "
                           f"(no combo with >= {wf['min_train_trades']} train trades)"
                           + (f" -> {skipped_years}" if skipped_years else ""))
        picks.insert(0, "variant", var)
        # pick table shows the change itself, not just an aggregate count in the summary
        picks["changed"] = picks["combo"].ne(picks["combo"].shift())
        picks.iloc[0, picks.columns.get_loc("changed")] = False
        all_picks.append(picks)
        cand = stitched(picks, tr)
        eq, info = kp.run_portfolio(cal, cand, closes, cap, slots=1, rank_by="signal_date")
        taken = taken_trades(eq, cand)
        expo = float(in_position(cal, taken).mean())
        hr = stitched_hold(picks, g, rets, cal)
        hold = pd.Series(cap * np.cumprod(1 + hr.to_numpy()), cal)
        s = summarize("wf", "걸어가며 검증", eq, cap, taken, expo)
        s.update({"distinct_picks": int(picks["combo"].nunique()),
                  "changes": int((picks["combo"] != picks["combo"].shift()).sum() - 1),
                  "skipped": info["trades_available"] - info["trades_taken"],
                  "skipped_years": ",".join(str(y) for y in skipped_years)})
        b = matched(hr.to_numpy(), hold, cap, s["mdd"], expo)
        summary += [{"variant": var, **s},
                    {"variant": var, **summarize("hold", "같은 종목 보유 (해마다 고른 종목)", hold, cap)},
                    {"variant": var, **summarize("kdd", "같은 종목 보유, 낙폭 맞춤", b["kdd"], cap, exposure=b["kdd_k"]),
                     "k": b["kdd_k"], "capped": b["kdd_capped"]},
                    {"variant": var, **summarize("kexp", "같은 종목 보유, 노출 맞춤", b["kexp"], cap, exposure=b["kexp_k"]),
                     "k": b["kexp_k"]}]
        ok = g.loc[allowed]
        best = int(ok[ok["trades"] >= wf["min_train_trades"]]["sharpe"].idxmax())
        hb = tr[(tr["combo"] == best) & (tr["entry_date"] >= start)]
        eq_hb, _ = kp.run_portfolio(cal, hb, closes, cap, slots=1, rank_by="signal_date")
        tk_hb = taken_trades(eq_hb, hb)
        summary.append({"variant": var, **summarize("hindsight", f"뒤늦게 고른 1등: {label(g.loc[best])}", eq_hb, cap, tk_hb,
                                                     float(in_position(cal, tk_hb).mean())), "combo": best})
        curves.update({f"wf_{var}": eq, f"hold_{var}": hold, f"kdd_{var}": b["kdd"], f"kexp_{var}": b["kexp"]})
        taken_all.append(taken.assign(variant=var))
        last = picks.iloc[-1]
        c = g.loc[int(last["combo"])]
        cands.append({"variant": var, "year": int(last["year"]), "combo": int(last["combo"]), "label": last["label"],
                      "ticker": c["ticker"], "rsi_thr": int(c["rsi_thr"]), "trend": c["trend"], "exit": c["exit"],
                      "exec": c["exec"], "train_sharpe": float(last["train_sharpe"]), "train_trades": int(last["train_trades"])})
    for t in ("SPY", "QQQ"):
        r = rets[t].to_numpy().copy()
        r[0] = 0.0
        h = pd.Series(cap * np.cumprod(1 + r), cal)
        summary.append({"variant": "-", **summarize(f"hold_{t}", f"{t} 보유", h, cap)})
        curves[f"hold_{t}"] = h
    picks = pd.concat(all_picks, ignore_index=True)
    picks.to_csv(rd / "walkforward.csv", index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(rd / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "walkforward_equity.csv")
    pd.concat(taken_all, ignore_index=True).to_parquet(rd / "walkforward_trades.parquet")
    pd.DataFrame(cands).to_csv(rd / "candidates.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("skipped years (explicit, not silently dropped):")
    for line in skip_lines:
        print(f"  {line}")
    print(picks[["variant", "year", "label", "train_sharpe", "test_sharpe", "test_return", "changed"]].round(3).to_string())
    print(sm[["variant", "name", "cagr", "mdd", "sharpe", "trades", "exposure"]].round(4).to_string())
    print(pd.DataFrame(cands)[["variant", "year", "label"]].to_string())


if __name__ == "__main__":
    main()
