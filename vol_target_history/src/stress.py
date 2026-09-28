"""SPEC 4.5: each big drawdown window, main assumption - max drawdown inside the window for T3 (every target) and
the hold, the return from the window's first day to after_days trading days past its end, and the average target
weight inside the window.
Usage: python -m src.stress   -> results/stress.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.run import key, runs


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    weights = pd.read_parquet(out / "weights.parquet")
    idx = daily.index
    rows = []
    for w, (a, b) in cfg["stress"]["windows"].items():
        inside = (idx >= a) & (idx <= b)
        last = idx.searchsorted(pd.Timestamp(b), side="right") - 1 + cfg["stress"]["after_days"]
        after = (idx >= a) & (idx <= idx[min(last, len(idx) - 1)])
        for r, t in runs(cfg):
            k = key(r, t, "본")
            rows.append({"window": w, "rule": r, "target": t, "mdd": C.mdd(daily[k][inside]),
                         "return_to_after": float((1 + daily[k][after]).prod() - 1),
                         "avg_weight": float(weights[k][inside].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress.csv", index=False)
    m = df[df["target"].isna() | (df["target"] == cfg["rule"]["main"])]
    print(m.pivot(index="window", columns="rule", values=["mdd", "return_to_after", "avg_weight"]).round(3).to_string())


if __name__ == "__main__":
    main()
