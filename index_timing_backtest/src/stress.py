"""SPEC 4.5: big declines. For each window: the worst drawdown inside it (hold and rule), the return from the window's
first day to `after_days` trading days after its end (did the rule catch the rebound?), and the average target.
Main rules, base dividend. Run after run.py.
Usage: python -m src.stress   -> results/stress.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    weights = pd.read_parquet(out / "weights.parquet")
    st = cfg["stress"]
    rows = []
    for m in C.MARKETS:
        dv = "" if m == "US" else cfg["markets"]["KR"]["dividend"]
        h = daily[f"{m}|HOLD||{dv}"].dropna()
        for name, (a, b) in st["windows"].items():
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            win = (h.index >= a) & (h.index <= b)
            if not win.any():
                continue
            last = min(int(win.nonzero()[0][-1]) + int(st["after_days"]), len(h) - 1)
            span = (h.index >= a) & (h.index <= h.index[last])
            for r, rc in cfg["rules"].items():
                k = f"{m}|{r}|{rc['main']}|{dv}"
                d, w = daily[k].dropna(), weights[k].dropna()
                rows.append({"market": m, "window": name, "rule": r, "hold_dd": C.mdd(h[win]), "rule_dd": C.mdd(d[win]),
                             "hold_ret": float((1 + h[span]).prod() - 1), "rule_ret": float((1 + d[span]).prod() - 1),
                             "avg_weight": float(w[win].mean()), "to": h.index[last].date()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
