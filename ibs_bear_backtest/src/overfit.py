"""SPEC 4.4 / criterion 6: daily Deflated Sharpe of the judged basket (bear version). Run after basket.py.

N = trials_before (1,204) + this folder's 12 runs = 1,216. V = variance of the daily Sharpe of us_shortterm's
first-round S1 -> S8 runs (56 columns of its daily.parquet). DSR functions copied from us_shortterm/src/deep/overfit.py.
Sensitivity rows (not judged): V over all 72 runs, and each judged ticker alone.
Usage: python -m src.overfit   -> results/dsr.csv
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import common as C

EULER = 0.5772156649015329
ND = NormalDist()
HIGH_TURNOVER = ("s09_", "s10_", "s11_", "s12_")


def expected_max_z(n: int) -> float:
    return (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))


def dsr_from_moments(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> tuple[float, float]:
    sr0 = math.sqrt(var_sr) * expected_max_z(n_trials)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return sr0, ND.cdf(z)


def deflated_sharpe(daily: np.ndarray, n_trials: int, var_sr: float) -> dict:
    x = np.asarray(daily, np.float64)
    x = x[np.isfinite(x)]
    m, s0 = x.mean(), x.std(ddof=0)
    sr = float(m / x.std(ddof=1))
    skew = float(np.mean((x - m) ** 3) / s0 ** 3)
    kurt = float(np.mean((x - m) ** 4) / s0 ** 4)
    sr0, dsr = dsr_from_moments(sr, len(x), skew, kurt, n_trials, var_sr)
    return {"sr_daily": sr, "T": len(x), "skew": skew, "kurt": kurt, "n_trials": n_trials, "var_sr": var_sr,
            "sr0": sr0, "dsr": dsr}


def trial_variance(cfg: dict, exclude: tuple[str, ...] = HIGH_TURNOVER) -> tuple[float, int]:
    d = pd.read_parquet(C.ROOT / cfg["sibling_daily"])
    assert d.shape[1] == 72, d.shape
    if exclude:
        d = d[[c for c in d.columns if not c.startswith(exclude)]]
    sr = d.mean() / d.std(ddof=1)
    sr = sr[np.isfinite(sr)]
    return float(sr.var(ddof=1)), len(sr)


def n_trials(cfg: dict) -> int:
    return int(cfg["trials_before"]) + len(cfg["tickers"]["judged"]) * len(C.VERSIONS)


def main() -> None:
    cfg = C.load_config()
    n = n_trials(cfg)
    v, nv = trial_variance(cfg)
    v72, _ = trial_variance(cfg, ())
    assert nv == 56, nv
    b = pd.read_parquet(C.results_dir() / "basket_daily.parquet")["bear"].to_numpy()
    rows = [{"what": "바구니 bear", "role": "판정", **deflated_sharpe(b, n, v)},
            {"what": "바구니 bear", "role": "민감도 V 72개", **deflated_sharpe(b, n, v72)}]
    daily = pd.read_parquet(C.results_dir() / "daily.parquet")
    for t in cfg["tickers"]["judged"]:
        rows.append({"what": t, "role": "민감도 종목 하나", **deflated_sharpe(daily[f"{t}|bear"].to_numpy(), n, v)})
    df = pd.DataFrame(rows)
    df.to_csv(C.results_dir() / "dsr.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
