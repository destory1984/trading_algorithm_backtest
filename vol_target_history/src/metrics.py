"""SPEC 4.4: metrics per run and stretch (full + the four eras), with the baselines built from the same assumption's
hold run (adapted from index_timing_backtest/src/metrics.py):
  risk  = w x hold + (1 - w) x cash, rebalanced daily, w by bisection so its max drawdown equals the rule's
  avgw  = the same with w = the rule's average weight
Run after run.py.
Usage: python -m src.metrics   -> results/metrics.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C
from src.run import assumptions


def cash_returns(idx: pd.DatetimeIndex, rate: float) -> pd.Series:
    days = np.concatenate([[0.0], np.diff(idx.to_numpy()).astype("timedelta64[D]").astype(np.float64)])
    return pd.Series((1 + rate) ** (days / 365.25) - 1, index=idx)


def risk_weight(hold: pd.Series, cash: pd.Series, target: float, iters: int = 100) -> float:
    if target <= C.mdd(hold):
        return 1.0
    if target >= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        w = (lo + hi) / 2
        lo, hi = (w, hi) if C.mdd(w * hold + (1 - w) * cash) > target else (lo, w)
    return (lo + hi) / 2


def stretches(cfg: dict) -> dict[str, tuple]:
    return {"전체": (cfg["data"]["start"], cfg["data"]["end"]), **{k: tuple(v) for k, v in cfg["data"]["eras"].items()}}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    weights = pd.read_parquet(out / "weights.parquet")
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    asm = assumptions(cfg)
    rows = []
    for k in daily.columns:
        rule, tgt, a = k.split("|")
        h_all = daily[f"HOLD||{a}"]
        c_all = cash_returns(daily.index, asm[a]["cash"])
        t_all = tr.loc[tr["run"] == k, "date"]
        for s, (lo, hi) in stretches(cfg).items():
            msk = (daily.index >= lo) & (daily.index <= hi)
            d, h, w, c = daily[k][msk], h_all[msk], weights[k][msk], c_all[msk]
            years = (d.index[-1] - d.index[0]).days / 365.25
            n_tr = int(((t_all >= d.index[0]) & (t_all <= d.index[-1])).sum())
            wr = risk_weight(h, c, C.mdd(d))
            wa = float(w.mean())
            rows.append({"rule": rule, "target": float(tgt) if tgt else None, "assumption": a, "run": k, "stretch": s,
                         "first_day": d.index[0].date(), "last_day": d.index[-1].date(),
                         "cagr": C.cagr(d), "mdd": C.mdd(d), "sharpe": C.sharpe(d),
                         "calmar": C.cagr(d) / abs(C.mdd(d)) if C.mdd(d) < 0 else np.nan,
                         "avg_weight": wa, "trades": n_tr, "trades_per_year": n_tr / years,
                         "underwater_days": C.longest_underwater(d),
                         "hold_cagr": C.cagr(h), "hold_mdd": C.mdd(h), "hold_sharpe": C.sharpe(h),
                         "risk_w": wr, "risk_cagr": C.cagr(wr * h + (1 - wr) * c),
                         "avgw_cagr": C.cagr(wa * h + (1 - wa) * c), "avgw_mdd": C.mdd(wa * h + (1 - wa) * c)})
    df = pd.DataFrame(rows)
    df["diff_hold"] = df["cagr"] - df["hold_cagr"]
    df["diff_risk"] = df["cagr"] - df["risk_cagr"]
    df["diff_avgw"] = df["cagr"] - df["avgw_cagr"]
    df["mdd_ratio"] = df["mdd"] / df["hold_mdd"]
    df.to_csv(out / "metrics.csv", index=False)
    show = df[df["rule"] == "T3"]
    print(show[["target", "assumption", "stretch", "cagr", "mdd", "hold_cagr", "hold_mdd", "mdd_ratio", "diff_hold",
                "diff_risk", "trades_per_year", "avg_weight"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
