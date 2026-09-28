"""Our engine + account vs the original package under the same conditions:
TQQQ real listing to 2026-07-30, no costs, ^IRX cash, full weight, open fills, $10,000.
Ours is run with irx_lag 0 (the original's same-day rate) and 1 (the spec).
Differences left: whole shares (original) vs fractional (ours); Yahoo auto_adjust (original)
vs adj_close / close scaling (us_data); the calendar (TQQQ's own days vs the SPX days).

Usage:  python -m src.crosscheck     (after src.stage0 --run)
Writes  results/crosscheck.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import daily_returns, load_config, results_dir, stats
from .data import cash_factors, frames, irx
from .sizing import account
from .strategy import calendar, trades


def compare(cfg: dict) -> pd.DataFrame:
    xc = cfg["crosscheck"]
    s0 = results_dir("stage0")
    orig = pd.read_csv(s0 / "orig_tqqq_trades.csv", parse_dates=["entry_date", "exit_date"])
    od = pd.read_csv(s0 / "orig_tqqq_daily.csv", index_col=0, parse_dates=True)
    closed = orig[orig["exit_date"].notna()]
    od_d = od["Capital"].pct_change().fillna(0.0)
    o = stats(od_d, float(od["Position"].mean()))
    rows = [{"source": "original", "trades": len(closed), **{k: o[k] for k in ("cagr", "sharpe", "mdd", "exposure")}}]
    f = frames(cfg, ["TQQQ"], end=xc["end"])["TQQQ"]
    cal = calendar(cfg, f)
    tr = trades(cfg, f, "TQQQ", "open", with_costs=False)
    ours = tr[tr["reason"] != "forced"]
    same = len(set(closed["entry_date"]) & set(ours["entry_date"]))
    common = od.index.intersection(f.index[f["valid"]])
    a, b = od.loc[common, "IBS"].to_numpy(), f.loc[common, "ibs"].to_numpy()
    sig_diff = int(((a < cfg["ibs"]["entry"]) != (b < cfg["ibs"]["entry"])).sum()
                   + ((a > cfg["ibs"]["exit"]) != (b > cfg["ibs"]["exit"])).sum())
    rate = irx(cfg)
    for lag in (0, 1):
        fac = cash_factors(cal, rate, lag, cfg["cash"]["days_per_year"])
        eq, info = account(cal, tr, f["close"], xc["capital"], np.ones(len(tr)), fac)
        s = stats(daily_returns(eq, xc["capital"]), info["invested_share"])
        rows.append({"source": f"ours irx_lag {lag}", "trades": len(ours),
                     **{k: s[k] for k in ("cagr", "sharpe", "mdd", "exposure")},
                     "same_entries": same, "trade_diff": abs(len(ours) - len(closed)) / len(closed),
                     "cagr_diff": s["cagr"] - o["cagr"], "ibs_days": len(common), "signal_days_differ": sig_diff})
    return pd.DataFrame(rows)


def main() -> None:
    df = compare(load_config())
    df.to_csv(results_dir() / "crosscheck.csv", index=False, encoding="utf-8-sig")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
