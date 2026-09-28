"""SPEC2 4.5: dependence on a few crashes.

Drop the trades whose signal day falls in each stage2.crash_periods window (one at a time, then all), and redo the
trade-level (sequential list) and portfolio (independent candidates) results.
Profit concentration: signal days ranked by the summed return of their trades (sequential list); the share of the
total summed return from the best 5 and 10 days, and the mean trade return with those days' trades removed.
Usage: python -m src.deep.crashes   -> results/deep/crashes.csv, top_days.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.deep import shared as S


def main() -> None:
    cfg = S.config()
    key = S.rule_key(cfg)
    seq = S.candidates("seq", key)
    ind = S.candidates("ind", key)
    periods = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in cfg["stage2"]["crash_periods"].items()}
    cases = {"없음": []} | {k: [k] for k in periods} | {"6개 모두": list(periods)}
    rows = []
    for name, drop in cases.items():
        ms = np.zeros(len(seq), bool)
        mi = np.zeros(len(ind), bool)
        for k in drop:
            a, b = periods[k]
            ms |= seq["signal_date"].between(a, b).to_numpy()
            mi |= ind["signal_date"].between(a, b).to_numpy()
        eq, info, cmp = S.run(cfg, ind[~mi])
        full = next(r for r in cmp if r["stretch"] == "full")
        rows.append({"dropped": name, "seq_dropped": int(ms.sum()), **S.trade_stats(seq.loc[~ms, "ret"]),
                     "cagr": full["cagr"], "mdd": full["mdd"], "risk_cagr": full["risk_cagr"],
                     "diff_risk": full["diff_risk"], "trades_taken": info["trades_taken"]})
    pd.DataFrame(rows).to_csv(S.results_dir() / "crashes.csv", index=False)
    by_day = seq.groupby("signal_date")["ret"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
    total = by_day["sum"].sum()
    top = []
    for n in cfg["deep"]["top_days"]:
        days = by_day.index[:n]
        rest = seq[~seq["signal_date"].isin(days)]["ret"]
        top.append({"top_days": n, "days": ", ".join(d.strftime("%Y-%m-%d") for d in days),
                    "trades_on_days": int(by_day["count"].iloc[:n].sum()), "share_of_sum": float(by_day["sum"].iloc[:n].sum() / total),
                    "mean_without": float(rest.mean()), "trades_without": len(rest)})
    pd.DataFrame(top).to_csv(S.results_dir() / "top_days.csv", index=False)
    print(pd.DataFrame(rows)[["dropped", "seq_dropped", "trades", "mean_ret", "cagr", "diff_risk"]].round(4).to_string(index=False))
    print(pd.DataFrame(top).drop(columns="days").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
