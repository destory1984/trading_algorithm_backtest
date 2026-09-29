"""SPEC 4.4: entry-day Deflated Sharpe. A position's return runs from its first buy to its sale (adds included,
proceeds / cost - 1); positions opened on the same day are averaged into one value per entry day.
N = dsr.trials (1,532). V = variance of the entry-day Sharpe over this folder's 6 variants. Sensitivity (not
judged): V from the bnf grid's signal-day Sharpes, when bnf's local results/trades.parquet exists.
DSR formula as in bnf_backtest/src/deep/overfit.py.
Usage: python -m src.dsr   -> results/dsr.csv
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
    m, s0 = x.mean(), x.std(ddof=0)
    return {"sr": float(m / x.std(ddof=1)), "T": len(x), "skew": float(np.mean((x - m) ** 3) / s0 ** 3),
            "kurt": float(np.mean((x - m) ** 4) / s0 ** 4)}


def dsr(mo: dict, n: int, v: float) -> tuple[float, float]:
    sr0 = math.sqrt(v) * expected_max_z(n)
    sr = mo["sr"]
    z = (sr - sr0) * math.sqrt(mo["T"] - 1) / math.sqrt(1 - mo["skew"] * sr + (mo["kurt"] - 1) / 4 * sr ** 2)
    return sr0, ND.cdf(z)


def day_series(t: pd.DataFrame) -> np.ndarray:
    return t.groupby("entry_day")["ret"].mean().to_numpy(np.float64)


def bnf_day_var(cfg: dict) -> float | None:
    p = (C.ROOT / cfg["dsr"]["bnf_grid_trades"]).resolve()
    if not Path(p).exists():
        return None
    t = pd.read_parquet(p, columns=["combo", "market", "signal_date", "ret"])
    t["ret"] = t["ret"].astype(np.float64)
    d = t.groupby(["combo", "market", "signal_date"], observed=True)["ret"].mean().groupby(["combo", "market"], observed=True).agg(["mean", "std"])
    s = (d["mean"] / d["std"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.var(ddof=1))


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    tr = pd.read_parquet(out / "trades.parquet")
    mo = {k: moments(day_series(g)) for k, g in tr.groupby("variant")}
    v = float(np.var([m["sr"] for m in mo.values()], ddof=1))
    vb = bnf_day_var(cfg)
    n = cfg["dsr"]["trials"]
    rows = []
    for r, e, main_ in C.variants(cfg):
        k = C.vkey(r, e)
        sr0, d = dsr(mo[k], n, v)
        row = {"variant": k, "main": main_, **mo[k], "n_trials": n, "var_sr": v, "sr0": sr0, "dsr": d}
        if vb is not None:
            row["sr0_bnf_v"], row["dsr_bnf_v"] = dsr(mo[k], n, vb)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "dsr.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
