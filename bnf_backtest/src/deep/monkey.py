"""SPEC2 4.3: the rule's candidates vs random KOSPI stocks bought on the same signal days.

Rule side: every signal as its own trade (kind=ind, the rule combo), data breaks dropped.
Random side (judged): for each of the rule's signal days d with k_d trades, draw k_d distinct stocks from the pool
of that day (every eligible KOSPI stock on d, kind=pool: next-day open entry, exit close >= MA25 -> next open,
else day-20 close, no stop), 2,000 times. Percentile = share of runs strictly below the rule.
Random side (reference only): the same total number of trades drawn from every eligible KOSPI stock-day.
Usage: python -m src.deep.monkey   -> results/deep/monkey.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.deep import shared as S


def pf_rows(m: np.ndarray) -> np.ndarray:
    """Profit factor of each row of a (runs x trades) return matrix."""
    gain = np.where(m > 0, m, 0).sum(axis=1)
    loss = -np.where(m < 0, m, 0).sum(axis=1)
    with np.errstate(divide="ignore"):
        return np.where(loss > 0, gain / loss, np.inf)


def same_day_draws(rule: pd.DataFrame, pool: pd.DataFrame, runs: int, rng: np.random.Generator) -> np.ndarray:
    """(runs x trades) returns: per signal day, k distinct pool stocks of that day."""
    k = rule.groupby("signal_date").size()
    by_day = {d: g["ret"].to_numpy(np.float64) for d, g in pool.groupby("signal_date")}
    cols = []
    for d, n in k.items():
        r = by_day[d]
        if n > len(r):
            raise ValueError(f"{d.date()}: {n} rule trades but {len(r)} pool stocks")
        idx = np.argsort(rng.random((runs, len(r))), axis=1)[:, :n]
        cols.append(r[idx])
    return np.concatenate(cols, axis=1)


def main() -> None:
    cfg = S.config()
    d = cfg["deep"]
    rng = np.random.default_rng(int(d["seed"]))
    rule = S.candidates("ind", S.rule_key(cfg))
    pool = S.candidates("pool", "pool")
    rr = rule["ret"].to_numpy(np.float64)
    pf, mean = float(pf_rows(rr[None, :])[0]), float(rr.mean())
    m = same_day_draws(rule, pool, int(d["monkey_runs"]), rng)
    pfs, means = pf_rows(m), m.mean(axis=1)
    pr = pool["ret"].to_numpy(np.float64)
    ref = np.stack([pr[rng.choice(len(pr), size=len(rr), replace=False)] for _ in range(int(d["monkey_runs"]))])
    rpfs, rmeans = pf_rows(ref), ref.mean(axis=1)
    days = pool[pool["signal_date"].isin(rule["signal_date"].unique())]
    rows = [
        {"test": "같은 날 무작위 종목 (판정)", "trades": len(rr), "signal_days": rule["signal_date"].nunique(),
         "pf": pf, "pf_pct": float((pfs < pf).mean()), "pf_rand_median": float(np.median(pfs)),
         "pf_rand_p95": float(np.quantile(pfs, 0.95)), "mean": mean, "mean_pct": float((means < mean).mean()),
         "mean_rand_median": float(np.median(means)), "mean_rand_p95": float(np.quantile(means, 0.95)),
         "pool_mean_same_days": float(days["ret"].mean()), "pool_size_same_days": len(days)},
        {"test": "아무 날 무작위 종목 (참고)", "trades": len(rr), "signal_days": np.nan,
         "pf": pf, "pf_pct": float((rpfs < pf).mean()), "pf_rand_median": float(np.median(rpfs)),
         "pf_rand_p95": float(np.quantile(rpfs, 0.95)), "mean": mean, "mean_pct": float((rmeans < mean).mean()),
         "mean_rand_median": float(np.median(rmeans)), "mean_rand_p95": float(np.quantile(rmeans, 0.95)),
         "pool_mean_same_days": float(pool["ret"].mean()), "pool_size_same_days": len(pool)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(S.results_dir() / "monkey.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
