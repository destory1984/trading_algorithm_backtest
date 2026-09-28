"""SPEC2 4.1: rebuild the candidate file, then the rule's trade-level and portfolio results.

Trade level = the rule's per-ticker sequential trades (the grid's list, 804 trades). Portfolio = the rule's
independent candidates in the 5-slot portfolio. Baselines are the KOSPI total-return approximation.
Usage: python -m src.deep.base   -> results/deep/cand.parquet, metrics.csv, years.csv, equity.csv
"""
from __future__ import annotations

import pandas as pd

from src.deep import shared as S


def main() -> None:
    cfg = S.config()
    S.collect(cfg)
    out = S.results_dir()
    key = S.rule_key(cfg)
    seq = S.candidates("seq", key)
    ind = S.candidates("ind", key)
    eq, info, rows = S.run(cfg, ind)
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    trade_rows = {"full": seq, "train": seq[seq["entry_date"] <= te], "test": seq[seq["entry_date"] > te]}
    met = pd.DataFrame([{**r, **{f"t_{k}": v for k, v in S.trade_stats(trade_rows[r["stretch"]]["ret"]).items()},
                         **({f"p_{k}": v for k, v in info.items()} if r["stretch"] == "full" else {})} for r in rows])
    met.to_csv(out / "metrics.csv", index=False)
    yrs = seq.groupby(seq["entry_date"].dt.year)["ret"].agg(["count", "mean"]).rename_axis("year")
    ds, dk = S.daily(eq), S.kospi_tr(cfg)
    w = met.loc[met["stretch"] == "full", "risk_w"].iloc[0]
    g = pd.DataFrame({"s": ds, "k": dk})
    py = g.groupby(g.index.year).apply(lambda x: pd.Series({
        "portfolio": (1 + x["s"]).prod() - 1, "kospi": (1 + x["k"]).prod() - 1,
        "risk_kospi": (1 + w * x["k"]).prod() - 1}))
    py["diff_risk"] = py["portfolio"] - py["risk_kospi"]
    yrs.join(py, how="outer").rename_axis("year").to_csv(out / "years.csv")
    pd.DataFrame({"equity": eq, "held": eq.attrs["held"]}).to_csv(out / "equity.csv")
    eq.attrs["ledger"].to_csv(out / "ledger.csv", index=False)
    print(met[["stretch", "cagr", "mdd", "sharpe", "kospi_cagr", "risk_w", "risk_cagr", "diff_risk", "t_trades",
               "t_mean_ret"]].round(4).to_string(index=False))
    print(info)


if __name__ == "__main__":
    main()
