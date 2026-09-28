"""SPEC2 4.6: slippage 0.1 / 0.2 / 0.3 / 0.5% one way, capacity and opening gaps.

Slippage: trade returns recomputed with krxbt.costs.net_return (fees and the yearly sell tax unchanged), then the
sequential list's mean and the portfolio again.
Capacity: for each trade the portfolio took, its allocation / the stock's 20-day average traded value on the signal
day. Gap: entry open / signal-day close - 1.
Dividend sensitivity (not judged): criteria 3, 4 and 5 again with deep.kospi_dividend replaced by 0 / 0.7 / 1.7 /
2.7%, because the 1.7% is a rough figure.
Usage: python -m src.deep.costs   -> results/deep/costs.csv, capacity.csv, dividend.csv
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd
from krxbt.costs import net_return

from src.deep import shared as S


def with_slippage(cfg: dict, tr: pd.DataFrame, slip: float) -> pd.DataFrame:
    c = copy.deepcopy(cfg)
    c["costs"]["slippage"] = slip
    return tr.assign(ret=net_return(c, tr["entry_px"].to_numpy(np.float64), tr["exit_px"].to_numpy(np.float64),
                                    tr["exit_date"]))


def main() -> None:
    cfg = S.config()
    key = S.rule_key(cfg)
    seq = S.candidates("seq", key)
    ind = S.candidates("ind", key)
    rows = []
    for slip in cfg["deep"]["slippages"]:
        s2, i2 = with_slippage(cfg, seq, slip), with_slippage(cfg, ind, slip)
        eq, info, cmp = S.run(cfg, i2)
        for r in cmp:
            rows.append({"slippage": slip, **r, "seq_mean": float(s2["ret"].mean()), "trades_taken": info["trades_taken"]})
        print(slip, [(r["stretch"], round(r["diff_risk"], 4)) for r in cmp])
    pd.DataFrame(rows).to_csv(S.results_dir() / "costs.csv", index=False)

    eq, info, _ = S.run(cfg, ind)
    led = eq.attrs["ledger"]
    led["ticker"] = led["ticker"].astype(str)
    j = led.merge(ind[["ticker", "entry_date", "avg_value", "sig_close", "entry_px"]], on=["ticker", "entry_date"],
                  how="left", validate="one_to_one")
    j["share"] = j["alloc"] / j["avg_value"].astype(np.float64)
    j["gap"] = j["entry_px"].astype(np.float64) / j["sig_close"].astype(np.float64) - 1
    ind_gap = ind["entry_px"].astype(np.float64) / ind["sig_close"].astype(np.float64) - 1
    lim = cfg["deep"]["capacity_share"]
    cap = pd.DataFrame([{
        "taken": len(j), "share_median": j["share"].median(), "share_p90": j["share"].quantile(0.9),
        "share_max": j["share"].max(), "over_limit": int((j["share"] > lim).sum()), "limit": lim,
        "gap_taken_median": j["gap"].median(), "gap_taken_p10": j["gap"].quantile(0.1),
        "gap_taken_p90": j["gap"].quantile(0.9), "gap_all_median": ind_gap.median(),
        "gap_all_p10": ind_gap.quantile(0.1), "gap_all_p90": ind_gap.quantile(0.9)}])
    cap.to_csv(S.results_dir() / "capacity.csv", index=False)
    print(cap.T.to_string())

    nb = cfg["deep"]["neighbors"]
    cand = S.candidates("ind")
    rows = []
    for dv in (0.0, 0.007, 0.017, 0.027):
        c = copy.deepcopy(cfg)
        c["deep"]["kospi_dividend"] = dv
        _, _, cmp = S.run(c, ind)
        wins = sum(S.run(c, cand[cand["combo"] == S.combo_key(t, None, h, m)])[2][0]["diff_risk"] > 0
                   for t, m, h in itertools.product(nb["threshold"], nb["market_filter"], nb["hold"]))
        _, _, sl = S.run(c, with_slippage(cfg, ind, cfg["deep"]["slippage_pass"]))
        rows.append({"dividend": dv, **{f"diff_{r['stretch']}": r["diff_risk"] for r in cmp},
                     "neighbors_won": int(wins), "slip_full": sl[0]["diff_risk"]})
    dvd = pd.DataFrame(rows)
    dvd.to_csv(S.results_dir() / "dividend.csv", index=False)
    print(dvd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
