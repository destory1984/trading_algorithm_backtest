"""SPEC2 4.2: Deflated Sharpe (Bailey and Lopez de Prado 2014) of the S1 rule on the screen stretch, per ETF.

The DSR functions are copied from rsi2_backtest/src/overfit.py (expected_max_z, dsr_from_moments, deflated_sharpe).
N = sum of deep.trials_before (1,104). V = variance of the daily Sharpe of the first round's 72 runs.
Other N values are reported as sensitivity only.
Usage: python -m src.deep.overfit   -> results/deep/dsr.csv
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import common
from src.deep import base

EULER = 0.5772156649015329
ND = NormalDist()


def expected_max_z(n: int) -> float:
    """Approximate expected maximum of n independent standard normals (the paper's SR0 factor)."""
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


def trial_variance(exclude: tuple[str, ...] = ()) -> float:
    """Variance of the first round's daily Sharpes; `exclude` drops strategies by label prefix (sensitivity)."""
    d = pd.read_parquet(common.results_dir() / "daily.parquet")
    assert d.shape[1] == 72, d.shape
    d = d[[c for c in d.columns if not c.startswith(exclude)]] if exclude else d
    sr = d.mean() / d.std(ddof=1)
    return float(sr[np.isfinite(sr)].var(ddof=1))


HIGH_TURNOVER = ("s09_", "s10_", "s11_", "s12_")  # sensitivity V without them (56 runs)


def n_trials(cfg: dict) -> int:
    return int(sum(cfg["deep"]["trials_before"].values()))


def main() -> None:
    cfg = common.load_config()
    n, v, v56 = n_trials(cfg), trial_variance(), trial_variance(HIGH_TURNOVER)
    _, st = base.all_stretches(cfg)
    rows = []
    for t in cfg["screen"]["tickers"]:
        daily = st[("screen", t)]["daily"].to_numpy()
        for nn, vv, role in ((n, v, "판정"), (72, v, "민감도"), (n + 100, v, "민감도"), (n, v56, "민감도 V: S1 → S8 56개")):
            rows.append({"ticker": t, "role": role, **deflated_sharpe(daily, nn, vv)})
    df = pd.DataFrame(rows)
    df.to_csv(common.results_dir("deep") / "dsr.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
