"""SPEC2 4.6: one-way cost 0 / 5 / 8 / 12 / 20 bp (same on both sides), screen and holdout stretches.
Usage: python -m src.deep.costs   -> results/deep/costs.csv
"""
from __future__ import annotations

import pandas as pd

from src import common
from src.deep import base


def main() -> None:
    cfg = common.load_config()
    rows = []
    for bp in cfg["deep"]["costs_bp"]:
        df, _ = base.all_stretches(cfg, common.one_way(bp))
        df = df[df["stretch"].isin(["screen", "holdout"])]
        rows += [{"bp": bp, **r} for r in df[["stretch", "ticker", "cagr", "risk_cagr", "diff_risk"]].to_dict("records")]
    out = pd.DataFrame(rows)
    out.to_csv(common.results_dir("deep") / "costs.csv", index=False)
    print(out.pivot_table(index=["stretch", "ticker"], columns="bp", values="diff_risk").round(4).to_string())


if __name__ == "__main__":
    main()
