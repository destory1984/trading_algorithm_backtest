"""SPEC 5.3: each big drawdown window - return and max drawdown inside the window for the main rules, EW5, SPY and
60/40 (net, cash 0), and what each rule held when the window opened and after each change inside it.
Usage: python -m src.stress   -> results/stress.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def held(row: pd.Series) -> str:
    parts = [f"{t} {v:.0%}" for t, v in row.items() if v > 1e-9]
    return ", ".join(parts) if parts else "현금"


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    tg = pd.read_csv(out / "targets.csv", parse_dates=["date"])
    tick = cfg["data"]["tickers"]
    idx = daily.index
    runs = [(r, s["main"]) for r, s in cfg["rules"].items()] + [(b, None) for b in cfg["baselines"]]
    rows = []
    for w, (a, b) in cfg["stress"]["windows"].items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        for r, p in runs:
            k = C.key(r, p, cfg["cash"], True)
            d = daily[k][(idx >= a) & (idx <= b)]
            t = tg[tg["run"] == k].set_index("date")[tick]
            # a decision on day x is in force from the next trading day
            exec_day = pd.Series(idx[idx.searchsorted(t.index) + 1], index=t.index)
            live = t[(exec_day > a) & (exec_day <= b)]
            start = t[exec_day <= a].iloc[-1]
            prev = t.shift(1)
            path = [held(start)] + [f"{exec_day[x].date()} {held(v)}" for x, v in live.iterrows()
                                    if not (v == prev.loc[x]).all()]
            rows.append({"window": w, "rule": r, "param": p, "return": float((1 + d).prod() - 1), "mdd": C.mdd(d),
                         "holdings": " → ".join(path) if r in cfg["rules"] else ""})
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress.csv", index=False)
    print(df.pivot(index="window", columns="rule", values="mdd").round(3).to_string())


if __name__ == "__main__":
    main()
