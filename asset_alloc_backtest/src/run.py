"""SPEC 5.1: monthly targets and daily curves for the 3 rules, 6 neighbours and 3 fixed mixes, at every cash rate,
net and gross of costs.
Usage: python -m src.run   -> results/daily.parquet, results/targets.csv, results/trades.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.rules import decide
from src.sim import simulate


def runs(cfg: dict) -> list[tuple]:
    out = []
    for r, spec in cfg["rules"].items():
        for p in [spec["main"], *spec["neighbors"]]:
            out.append((r, p))
    out += [(b, None) for b in cfg["baselines"]]
    return out


def targets(cfg: dict, opn: pd.DataFrame, cls: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    me = cls.loc[C.month_ends(cls.index)]
    return {(r, p): decide(cfg, r, p, me) for r, p in runs(cfg)}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    opn, cls = C.prices(cfg)
    start = pd.Timestamp(cfg["data"]["start"])
    tg = targets(cfg, opn, cls)
    daily, trades, rows = {}, [], []
    for (r, p), dec in tg.items():
        t = dec.copy()
        t.insert(0, "run", C.key(r, p, cfg["cash"], True))
        rows.append(t.dropna(how="any").reset_index(names="date"))
        for cash in [cfg["cash"], *cfg["cash_sens"]]:
            for net in (True, False):
                k = C.key(r, p, cash, net)
                d, tr = simulate(opn, cls, dec, start, cfg["cost"] if net else 0.0, cash)
                daily[k] = d
                if net and cash == cfg["cash"]:
                    tr.insert(0, "run", k)
                    trades.append(tr)
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")
    pd.concat(rows).to_csv(out / "targets.csv", index=False)
    pd.concat(trades).to_csv(out / "trades.csv", index=False)
    d = pd.DataFrame(daily)
    print(f"{len(daily)} curves, {d.index[0].date()} -> {d.index[-1].date()}")


if __name__ == "__main__":
    main()
