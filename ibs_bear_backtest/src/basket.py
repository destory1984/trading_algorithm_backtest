"""SPEC 4.6 (reference; criterion 6 uses its daily returns): the 6 judged tickers as 1/6 sleeves run separately
(basket equity = mean of the six equity curves, no rebalancing), against the same 1/6 sleeves of buy-and-hold.
Run after metrics.py.
Usage: python -m src.basket   -> results/basket.csv, basket_daily.parquet
"""
from __future__ import annotations

import pandas as pd

from src import benchmark, equity
from src import common as C


def basket_daily(daily: pd.DataFrame, cols: list[str]) -> pd.Series:
    eq = (1 + daily[cols]).cumprod().mean(axis=1)
    d = eq.pct_change()
    d.iloc[0] = eq.iloc[0] - 1
    return d


def main() -> None:
    cfg = C.load_config()
    daily = pd.read_parquet(C.results_dir() / "daily.parquet")
    js = cfg["tickers"]["judged"]
    hold = basket_daily(daily, [f"{t}|hold" for t in js])
    rows, keep = [], {"hold": hold}
    for v in C.VERSIONS:
        d = basket_daily(daily, [f"{t}|{v}" for t in js])
        keep[v] = d
        m = {"cagr": equity.cagr(d), "mdd": equity.mdd(d), "sharpe": equity.sharpe(d)}
        expo = pd.read_csv(C.results_dir() / "metrics.csv").query("version == @v and ticker in @js")["exposure"].mean()
        rows.append({"version": v, **m, "exposure": expo, **benchmark.compare(hold, m["cagr"], m["mdd"], expo)})
    df = pd.DataFrame(rows)
    df.to_csv(C.results_dir() / "basket.csv", index=False)
    pd.DataFrame(keep).to_parquet(C.results_dir() / "basket_daily.parquet")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
