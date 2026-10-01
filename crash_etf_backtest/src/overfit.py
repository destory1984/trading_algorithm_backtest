"""SPEC 4.6 / criterion 8: Deflated Sharpe per entry day (mean return of the trades entered that day = one
observation). N = 3, V = variance of the three variants' entry-day Sharpes. Sensitivity (not judged): N = 1,443 and
V from the bnf grid's signal-day Sharpes when bnf's local results/trades.parquet exists.
DSR functions copied from ibs_bear_backtest/src/overfit.py; bnf_day_var from turtle_kr_backtest/src/dsr.py.
Usage: python -m src.overfit   -> results/dsr.csv
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import common as C

EULER = 0.5772156649015329
ND = NormalDist()


def expected_max_z(n: int) -> float:
    return (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))


def moments(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    m, s0 = x.mean(), x.std(ddof=0)
    return {"sr": float(m / x.std(ddof=1)), "T": len(x), "skew": float(np.mean((x - m) ** 3) / s0 ** 3),
            "kurt": float(np.mean((x - m) ** 4) / s0 ** 4)}


def dsr(mo: dict, n_trials: int, var_sr: float) -> tuple[float, float]:
    sr0 = math.sqrt(var_sr) * expected_max_z(n_trials)
    z = (mo["sr"] - sr0) * math.sqrt(mo["T"] - 1) / math.sqrt(1 - mo["skew"] * mo["sr"] + (mo["kurt"] - 1) / 4 * mo["sr"] ** 2)
    return sr0, ND.cdf(z)


def day_series(t: pd.DataFrame) -> np.ndarray:
    return t.groupby("entry_day")["ret"].mean().to_numpy(np.float64)


def bnf_day_var(cfg: dict) -> float | None:
    p = (C.ROOT / cfg["sens"]["bnf_grid_trades"]).resolve()
    if not Path(p).exists():
        return None
    t = pd.read_parquet(p, columns=["combo", "market", "signal_date", "ret"])
    t["ret"] = t["ret"].astype(np.float64)
    d = t.groupby(["combo", "market", "signal_date"], observed=True)["ret"].mean().groupby(["combo", "market"], observed=True).agg(["mean", "std"])
    s = (d["mean"] / d["std"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.var(ddof=1))


def names(cfg: dict) -> list[str]:
    return [f"{round(th * 100)}" for th in [cfg["rule"]["main"], *cfg["rule"]["neighbors"]]]


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    tr = pd.read_parquet(out / "trades.parquet")
    mo = {k: moments(day_series(tr[tr["variant"] == k])) for k in names(cfg)}
    v = float(np.var([m["sr"] for m in mo.values()], ddof=1))
    n = cfg["pass"]["trials"]
    vb = bnf_day_var(cfg)
    rows = []
    for k in names(cfg):
        sr0, d = dsr(mo[k], n, v)
        row = {"variant": k, **mo[k], "n_trials": n, "var_sr": v, "sr0": sr0, "dsr": d}
        if vb is not None:
            row["n_sens"], row["var_bnf"] = cfg["sens"]["dsr_trials"], vb
            row["sr0_sens"], row["dsr_sens"] = dsr(mo[k], cfg["sens"]["dsr_trials"], vb)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "dsr.csv", index=False)
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
