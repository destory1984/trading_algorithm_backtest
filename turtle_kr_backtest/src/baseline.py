"""SPEC 4.3: the baselines.
  random entry  for each main rule, `random.runs` accounts that buy, on each day the rule opened k new positions,
                k random stocks that can be bought that day (eligible, not locked, N known, not held, not sold today)
                at the open, and manage them with the same Turtle sizing, adds, stops and exit channel
  order (sensitivity, not judged, SPEC 11)  the rule itself with the day's candidates in random order, 50 runs
KOSPI total return and the drawdown-matched KOSPI are built in metrics.py.
Run after portfolio.py.
Usage: python -m src.baseline   -> results/random.csv, results/order.csv
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import common as C
from src import signals as SG
from src.portfolio import Panel, simulate

ORDER_RUNS = 50
_P = None


def _init():
    global _P, _E, _CFG
    _CFG = C.load_config()
    p = SG.load(mmap=True)
    _P = Panel(p, in_ram=False)
    _E = np.asarray(p["eligible_days"])


def _stats(eq: pd.Series, capital: float) -> tuple[float, float]:
    d = eq.pct_change().fillna(eq.iloc[0] / capital - 1)
    return C.cagr(d), C.mdd(d)


def _random(job):
    rule, exit_days, k_series, seed = job
    res = simulate(_CFG, _P, exit_days, None, random_k=k_series, elig_days=_E, seed=seed)
    cg, dd = _stats(res["equity"], _CFG["turtle"]["capital"])
    return {"rule": rule, "seed": seed, "cagr": cg, "mdd": dd, "trades": len(res["trades"]),
            "win": float((res["trades"]["ret"] > 0).mean()) if len(res["trades"]) else np.nan}


def _order(job):
    rule, key, exit_days, seed = job
    c = SG.load(mmap=True)["cands"][key]
    rng = np.random.default_rng(seed)
    c = c.assign(_r=rng.random(len(c))).sort_values(["day", "_r"], kind="stable").drop(columns="_r")
    res = simulate(_CFG, _P, exit_days, c)
    cg, dd = _stats(res["equity"], _CFG["turtle"]["capital"])
    return {"rule": rule, "seed": seed, "cagr": cg, "mdd": dd, "trades": len(res["trades"])}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    ent = pd.read_parquet(out / "entries.parquet")
    jobs, ojobs = [], []
    for r, s in cfg["rules"].items():
        k = C.vkey(r, s["entry"])
        ks = ent[k].to_numpy()
        jobs += [(r, s["exit"], ks, cfg["random"]["seed"] + i) for i in range(cfg["random"]["runs"])]
        ojobs += [(r, k, s["exit"], 1000 + i) for i in range(ORDER_RUNS)]
    with ProcessPoolExecutor(cfg["random"]["workers"], initializer=_init) as ex:
        rnd = pd.DataFrame(list(ex.map(_random, jobs, chunksize=4)))
        order = pd.DataFrame(list(ex.map(_order, ojobs, chunksize=2)))
    rnd.to_csv(out / "random.csv", index=False)
    order.to_csv(out / "order.csv", index=False)
    print(rnd.groupby("rule")[["cagr", "mdd", "trades", "win"]].describe().T.round(3).to_string())
    print(order.groupby("rule")[["cagr", "mdd"]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
