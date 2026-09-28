"""SPEC 5.4: Deflated Sharpe (Bailey and Lopez de Prado 2014) of each rule's monthly excess return over EW5, one
observation per month (the rules decide once a month). Reported only, not judged (SPEC 7).

Trials N = dsr.trials (27). The variance of the Sharpe ratios across trials is taken from the 9 trials of this
folder (the 18 index_timing trials are not on the same excess series). Monthly returns come from the net, cash-0
daily curves, compounded per calendar month.
Usage: python -m src.dsr   -> results/dsr.csv
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import common as C

N01 = NormalDist()
EULER = 0.5772156649015329


def monthly(d: pd.Series) -> pd.Series:
    return (1 + d).groupby([d.index.year, d.index.month]).prod() - 1


def moments(x: np.ndarray) -> tuple[float, float, float]:
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=0)
    return float(sr), float((z ** 3).mean()), float((z ** 4).mean())


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    ew = monthly(daily[C.key("EW5", None, cfg["cash"], True)])
    rows = []
    for r, s in cfg["rules"].items():
        for p in [s["main"], *s["neighbors"]]:
            x = (monthly(daily[C.key(r, p, cfg["cash"], True)]) - ew).to_numpy()
            sr, sk, ku = moments(x)
            rows.append({"rule": r, "param": p, "main": p == s["main"], "months": len(x), "sr_month": sr,
                         "sr_year": sr * math.sqrt(12), "skew": sk, "kurt": ku})
    df = pd.DataFrame(rows)
    n = cfg["dsr"]["trials"]
    v = df["sr_month"].var(ddof=1)
    sr0 = math.sqrt(v) * ((1 - EULER) * N01.inv_cdf(1 - 1 / n) + EULER * N01.inv_cdf(1 - 1 / (n * math.e)))
    df["sr0_month"] = sr0
    df["dsr"] = [N01.cdf((r.sr_month - sr0) * math.sqrt(r.months - 1)
                         / math.sqrt(1 - r.skew * r.sr_month + (r.kurt - 1) / 4 * r.sr_month ** 2))
                 for r in df.itertuples()]
    df["psr0"] = [N01.cdf(r.sr_month * math.sqrt(r.months - 1)
                          / math.sqrt(1 - r.skew * r.sr_month + (r.kurt - 1) / 4 * r.sr_month ** 2))
                  for r in df.itertuples()]
    df.to_csv(out / "dsr.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
