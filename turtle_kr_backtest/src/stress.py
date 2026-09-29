"""SPEC 4.6: each window - return and max drawdown of the main rules and KOSPI total return, average invested
share, and the 5 most profitable stocks among positions sold inside the window.
Usage: python -m src.stress   -> results/stress.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.metrics import daily


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    eq = pd.read_parquet(out / "equity.parquet")
    inv = pd.read_parquet(out / "invested.parquet")
    tr = pd.read_parquet(out / "trades.parquet")
    names = C.tickers(cfg)["name"]
    cap = cfg["turtle"]["capital"]
    kospi = C.kospi_tr(cfg, C.calendar(cfg)).reindex(eq.index)
    rows = []
    for w, (a, b) in cfg["stress"]["windows"].items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        m = (eq.index >= a) & (eq.index <= b)
        rows.append({"window": w, "rule": "KOSPI", "return": float((1 + kospi[m]).prod() - 1), "mdd": C.mdd(kospi[m])})
        for r, s in cfg["rules"].items():
            k = C.vkey(r, s["entry"])
            d = daily(eq[k], cap)[m]
            t = tr[(tr["variant"] == k) & (tr["exit_day"] >= a) & (tr["exit_day"] <= b)]
            top = t.nlargest(5, "pnl")
            rows.append({"window": w, "rule": r, "return": float((1 + d).prod() - 1), "mdd": C.mdd(d),
                         "avg_invested": float(inv[k][m].mean()), "trades": len(t),
                         "win_rate": float((t["pnl"] > 0).mean()) if len(t) else None,
                         "top5": ", ".join(f"{names.get(x, x)} {y:+.0%}" for x, y in zip(top["ticker"], top["ret"]) if y > 0)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress.csv", index=False)
    print(df.drop(columns="top5").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
