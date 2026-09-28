"""SPEC 4.1: every rule (main + neighbours) and the hold on both markets; for Korea also at the sensitivity dividends.
Usage: python -m src.run   -> results/daily.parquet (returns), weights.parquet (targets), trades.csv
Columns are "<market>|<rule>|<param>|<dividend>".
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src import rules as R
from src import sim as S


def runs(cfg: dict) -> list[tuple]:
    out = []
    for m in C.MARKETS:
        mk = cfg["markets"][m]
        divs = [mk["dividend"]] + (mk.get("dividend_sens") or []) if m == "KR" else [None]
        for dv in divs:
            out.append((m, "HOLD", "", dv))
            for r, rc in cfg["rules"].items():
                for p in [rc["main"], *rc["neighbors"]]:
                    out.append((m, r, p, dv))
    return out


def key(m, r, p, dv) -> str:
    return f"{m}|{r}|{p}|{'' if dv is None else dv}"


def one(cfg: dict, m: str, r: str, p, dv, table: pd.DataFrame | None = None):
    px = C.prices(cfg, m, dv if m == "KR" else "config", table)
    dec = R.decisions(px["close"], r, p, cfg)
    return S.simulate(px, dec, pd.Timestamp(cfg["data"]["start"]), cfg["markets"][m]["cost"])


def main() -> None:
    cfg = C.load_config()
    daily, weights, trades = {}, {}, []
    for m, r, p, dv in runs(cfg):
        d, w, t = one(cfg, m, r, p, dv)
        k = key(m, r, p, dv)
        daily[k], weights[k] = d, w
        trades += [{"run": k, "date": x.date(), "weight": w[x]} for x in t]
    out = C.results_dir()
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")
    pd.DataFrame(weights).to_parquet(out / "weights.parquet")
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    print(len(daily), "runs")


if __name__ == "__main__":
    main()
