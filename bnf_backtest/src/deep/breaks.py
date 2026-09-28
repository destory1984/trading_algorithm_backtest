"""SPEC2 4.7: trades the engine drops because a price-data break falls inside them.

Three versions of the rule's sequential list and portfolio: breaks dropped (the default), kept at their recorded
prices, kept as -100%.
Usage: python -m src.deep.breaks   -> results/deep/breaks.csv, break_trades.csv
"""
from __future__ import annotations

import pandas as pd

from src.deep import shared as S


def main() -> None:
    cfg = S.config()
    key = S.rule_key(cfg)
    seq = S.candidates("seq", key, breaks=True)
    ind = S.candidates("ind", key, breaks=True)
    rows = []
    for name in ("뺌 (기본)", "기록된 가격", "-100%"):
        if name.startswith("뺌"):
            s, i = seq[~seq["data_break"]], ind[~ind["data_break"]]
        elif name == "-100%":
            s = seq.assign(ret=seq["ret"].where(~seq["data_break"], -1.0))
            i = ind.assign(ret=ind["ret"].where(~ind["data_break"], -1.0))
        else:
            s, i = seq, ind
        eq, info, cmp = S.run(cfg, i)
        full = next(r for r in cmp if r["stretch"] == "full")
        rows.append({"version": name, "seq_breaks": int(seq["data_break"].sum()), "ind_breaks": int(ind["data_break"].sum()),
                     **S.trade_stats(s["ret"]), "cagr": full["cagr"], "mdd": full["mdd"], "diff_risk": full["diff_risk"],
                     "trades_taken": info["trades_taken"]})
    pd.DataFrame(rows).to_csv(S.results_dir() / "breaks.csv", index=False)
    ind[ind["data_break"]][["ticker", "signal_date", "entry_date", "exit_date", "entry_px", "exit_px", "ret"]].to_csv(
        S.results_dir() / "break_trades.csv", index=False)
    print(pd.DataFrame(rows)[["version", "seq_breaks", "ind_breaks", "trades", "mean_ret", "cagr", "diff_risk"]].round(4).to_string(index=False))
    print(ind[ind["data_break"]][["ticker", "signal_date", "exit_date", "ret"]].to_string(index=False))


if __name__ == "__main__":
    main()
