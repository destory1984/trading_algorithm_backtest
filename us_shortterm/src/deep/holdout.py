"""SPEC2 4.1: the S1 rule on the screen, holdout and full stretches, 4 ETFs -> results/deep/metrics.csv.
Usage: python -m src.deep.holdout
"""
from __future__ import annotations

from src import common
from src.deep import base


def main() -> None:
    cfg = common.load_config()
    rows, _ = base.all_stretches(cfg)
    rows.to_csv(common.results_dir("deep") / "metrics.csv", index=False)
    print(rows[["stretch", "ticker", "trades", "cagr", "risk_cagr", "diff_risk", "sharpe", "mdd"]].to_string(index=False))


if __name__ == "__main__":
    main()
