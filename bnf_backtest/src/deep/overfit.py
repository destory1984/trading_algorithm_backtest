"""SPEC2 4.2: trade-level Deflated Sharpe of the rule.

Sharpe = mean trade return / std of trade returns, T = trades (the rule's sequential list, 804).
N = sum of deep.trials (1,526). V = variance of the trade-level Sharpe over the grid's 1,440 combos (combo x market,
market ALL = KOSPI + KOSDAQ), recomputed from results/trades.parquet.
DSR functions copied from us_shortterm/src/deep/overfit.py (itself from rsi2_backtest). Sensitivity rows (other N,
V over KOSPI combos only) are reported, not judged.
Usage: python -m src.deep.overfit   -> results/deep/dsr.csv
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.common import results_dir as grid_dir
from src.deep import shared as S

EULER = 0.5772156649015329
ND = NormalDist()


def expected_max_z(n: int) -> float:
    return (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))


def dsr_from_moments(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> tuple[float, float]:
    sr0 = math.sqrt(var_sr) * expected_max_z(n_trials)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return sr0, ND.cdf(z)


def deflated_sharpe(x: np.ndarray, n_trials: int, var_sr: float) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    m, s0 = x.mean(), x.std(ddof=0)
    sr = float(m / x.std(ddof=1))
    skew = float(np.mean((x - m) ** 3) / s0 ** 3)
    kurt = float(np.mean((x - m) ** 4) / s0 ** 4)
    sr0, dsr = dsr_from_moments(sr, len(x), skew, kurt, n_trials, var_sr)
    return {"sr_trade": sr, "T": len(x), "skew": skew, "kurt": kurt, "n_trials": n_trials, "var_sr": var_sr,
            "sr0": sr0, "dsr": dsr}


def grid_sharpes() -> pd.DataFrame:
    """Trade-level Sharpe of every grid combo x market (KOSPI, KOSDAQ, ALL)."""
    t = pd.read_parquet(grid_dir() / "trades.parquet", columns=["combo", "market", "ret"])
    t["ret"] = t["ret"].astype(np.float64)
    t["market"] = t["market"].astype(str)
    g = t.groupby(["combo", "market"])["ret"].agg(["mean", "std", "count"])
    a = t.groupby("combo")["ret"].agg(["mean", "std", "count"])
    a["market"] = "ALL"
    g = pd.concat([g.reset_index(), a.reset_index()], ignore_index=True)
    g["sr"] = g["mean"] / g["std"]
    return g


def grid_day_sharpes() -> pd.Series:
    """Sharpe of per-signal-day mean returns for every grid combo x market (sensitivity: trades on one day are not
    independent, so the day is the unit)."""
    t = pd.read_parquet(grid_dir() / "trades.parquet", columns=["combo", "market", "signal_date", "ret"])
    t["ret"] = t["ret"].astype(np.float64)
    t["market"] = t["market"].astype(str)
    d1 = t.groupby(["combo", "market", "signal_date"])["ret"].mean().groupby(["combo", "market"]).agg(["mean", "std"])
    d2 = t.groupby(["combo", "signal_date"])["ret"].mean().groupby("combo").agg(["mean", "std"])
    s = pd.concat([d1["mean"] / d1["std"], d2["mean"] / d2["std"]], ignore_index=True)
    return s[np.isfinite(s)]


def n_trials(cfg: dict) -> int:
    return int(sum(cfg["deep"]["trials"].values()))


def main() -> None:
    cfg = S.config()
    g = grid_sharpes()
    assert len(g) == 1440, len(g)
    v = float(g["sr"][np.isfinite(g["sr"])].var(ddof=1))
    v_kospi = float(g.loc[g["market"] == "KOSPI", "sr"].var(ddof=1))
    x = S.candidates("seq", S.rule_key(cfg))["ret"].to_numpy()
    n = n_trials(cfg)
    rows = [{"role": "판정", **deflated_sharpe(x, n, v)},
            {"role": "민감도 N 1,440", **deflated_sharpe(x, 1440, v)},
            {"role": "민감도 N 480 (코스피 그리드)", **deflated_sharpe(x, 480, v_kospi)},
            {"role": "민감도 V 코스피 그리드", **deflated_sharpe(x, n, v_kospi)}]
    seq = S.candidates("seq", S.rule_key(cfg))
    per_day = seq.groupby("signal_date")["ret"].mean().to_numpy(np.float64)
    gd = grid_day_sharpes()
    rows.append({"role": "민감도 신호일 단위", **deflated_sharpe(per_day, n, float(gd.var(ddof=1)))})
    df = pd.DataFrame(rows)
    df.to_csv(S.results_dir() / "dsr.csv", index=False)
    g.to_csv(S.results_dir() / "grid_sharpes.csv", index=False)
    print(df.to_string(index=False))
    print("rule rank by trade Sharpe:", int((g["sr"] > df["sr_trade"].iloc[0]).sum()) + 1, "of", len(g))


if __name__ == "__main__":
    main()
