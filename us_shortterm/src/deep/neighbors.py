"""SPEC2 4.4: the 25 neighbouring thresholds x 4 ETFs on the screen stretch (100 runs). Reported, never chosen from.
Usage: python -m src.deep.neighbors   -> results/deep/neighbors.csv
"""
from __future__ import annotations

import pandas as pd
from krxbt import engine

from src import common
from src.deep import base


def main() -> None:
    cfg = common.load_config()
    nb = cfg["deep"]["neighbors"]
    _, fs = base.load(cfg, deep=False)
    rows = []
    for t in cfg["screen"]["tickers"]:
        f = fs[t]
        a = engine.arrays(f)
        for en in nb["entry"]:
            for ex in nb["exit"]:
                tr = base.ibs_trades(f, a, cfg, en, ex)
                rows.append({"ticker": t, "entry": en, "exit": ex, **base.stretch(tr, f, cfg, cfg["costs"])["m"]})
    df = pd.DataFrame(rows)
    assert len(df) == 100
    df.to_csv(common.results_dir("deep") / "neighbors.csv", index=False)
    print(df.pivot_table(index=["ticker", "entry"], columns="exit", values="diff_risk").round(4).to_string())


if __name__ == "__main__":
    main()
