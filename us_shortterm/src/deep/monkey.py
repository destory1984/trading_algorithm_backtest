"""SPEC2 4.3: random entries vs the S1 rule on the screen stretch, per ETF.

Each run: the rule's trade count, holding spans resampled with replacement from the rule's trades
(exit bar - entry bar), non-overlapping trades placed uniformly at random over the stretch, next-day open entry
and open exit, same costs. A trade that signals on bar s buys at o[s+1] and sells at o[s+1+span]; the next random
signal may fall on that exit bar (as the engine allows). Percentile = share of runs strictly below the rule.
Usage: python -m src.deep.monkey   -> results/deep/monkey.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from src import common, equity
from src.deep import base


def place(n_bars: int, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Start bars of non-overlapping blocks of `lengths` inside [0, n_bars - 1] (a block of length L starting at
    s ends on bar s + L), uniform over the arrangements of the gaps."""
    n = len(lengths)
    slack = n_bars - 1 - int(lengths.sum())
    if slack < 0:
        raise ValueError("trades do not fit")
    cuts = np.sort(rng.choice(slack + n, size=n, replace=False))
    gaps = np.diff(np.concatenate([[-1], cuts])) - 1
    return np.cumsum(gaps) + np.concatenate([[0], np.cumsum(lengths)[:-1]])


def random_run(o: np.ndarray, spans_pool: np.ndarray, n_trades: int, bc: float, sk: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signal bars, exit bars, trade returns) of one random run."""
    spans = rng.choice(spans_pool, size=n_trades, replace=True)
    s = place(len(o), 1 + spans, rng)
    e = s + 1
    x = e + spans
    return s, x, o[x] * sk / (o[e] * bc) - 1


def main() -> None:
    cfg = common.load_config()
    d = cfg["deep"]
    rng = np.random.default_rng(int(d["seed"]))
    bc, sk = cost_factors(cfg)
    _, fs = base.load(cfg, deep=False)
    rows = []
    for t in cfg["screen"]["tickers"]:
        f = fs[t]
        dates, _ = base.curve_inputs(f, cfg)
        o = f.loc[dates, "open"].to_numpy(np.float64)
        tr = base.rule_trades(f, cfg)
        ix = pd.Index(dates)
        spans = ix.get_indexer(pd.DatetimeIndex(tr["exit_date"])) - ix.get_indexer(pd.DatetimeIndex(tr["entry_date"]))
        years = (dates[-1] - dates[0]).days / 365.25
        ret = tr["ret"].to_numpy(np.float64)
        pf, cg = equity.profit_factor(ret), float(np.prod(1 + ret)) ** (1 / years) - 1
        pfs, cgs = np.empty(d["monkey_runs"]), np.empty(d["monkey_runs"])
        for k in range(d["monkey_runs"]):
            _, _, r = random_run(o, spans, len(tr), bc, sk, rng)
            pfs[k] = equity.profit_factor(r)
            cgs[k] = float(np.prod(1 + r)) ** (1 / years) - 1
        rows.append({"ticker": t, "trades": len(tr), "pf": pf, "pf_pct": float((pfs < pf).mean()),
                     "pf_rand_median": float(np.median(pfs)), "pf_rand_p95": float(np.quantile(pfs, 0.95)),
                     "cagr": cg, "cagr_pct": float((cgs < cg).mean()),
                     "cagr_rand_median": float(np.median(cgs)), "cagr_rand_p95": float(np.quantile(cgs, 0.95))})
    df = pd.DataFrame(rows)
    df.to_csv(common.results_dir("deep") / "monkey.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
