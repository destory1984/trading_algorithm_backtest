"""SPEC 4.5 / criterion 5: bear-version trades vs random entries on below-200-day-SMA days, per ticker.

Each run: the rule's trade count; holding spans (exit bar - entry bar) resampled with replacement from the rule's
trades; signal bars drawn in random order from the eligible days whose close is below the own 200-day SMA, each
accepted only if its block (signal bar through the bar before the exit) is free, until the count is reached;
next-day open entry, open exit, same costs. Trades never overlap and the next signal may fall on an exit bar, as in
the engine. Percentile = share of runs strictly below the rule. Run after run.py.
Usage: python -m src.monkey   -> results/monkey.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine
from krxbt.costs import cost_factors

from src import common as C
from src.equity import profit_factor


def random_run(o: np.ndarray, days: np.ndarray, spans_pool: np.ndarray, n: int, bc: float, sk: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signal bars, exit bars, returns). Raises if n trades cannot be placed."""
    busy = np.zeros(len(o), bool)
    sig, ext = [], []
    for s in rng.permutation(days):
        span = int(rng.choice(spans_pool))
        x = s + 1 + span
        if x >= len(o) or busy[s:x].any():
            continue
        busy[s:x] = True
        sig.append(s)
        ext.append(x)
        if len(sig) == n:
            break
    if len(sig) < n:
        raise ValueError(f"placed {len(sig)} of {n}")
    s, x = np.array(sig), np.array(ext)
    return s, x, o[x] * sk / (o[s + 1] * bc) - 1


def main() -> None:
    cfg = C.load_config()
    rng = np.random.default_rng(int(cfg["monkey"]["seed"]))
    runs = int(cfg["monkey"]["runs"])
    bc, sk = cost_factors(cfg)
    _, frames, _ = C.load_frames(cfg)
    judged = set(cfg["tickers"]["judged"])
    rows = []
    for t, f in frames.items():
        dates, _ = C.curve_inputs(f, cfg)
        g = f.loc[dates]
        o = g["open"].to_numpy(np.float64)
        s = C.signals(f, cfg)
        k = np.asarray(f.index >= dates[0])
        pool = np.flatnonzero(s["below"][k] & engine.arrays(f)["eligible"][k])
        tr = pd.read_parquet(C.results_dir() / f"trades_{t}_bear.parquet")
        ix = pd.Index(dates)
        spans = ix.get_indexer(pd.DatetimeIndex(tr["exit_date"])) - ix.get_indexer(pd.DatetimeIndex(tr["entry_date"]))
        ret = tr["ret"].to_numpy(np.float64)
        pf, mean = profit_factor(ret), float(ret.mean())
        pfs, means = np.empty(runs), np.empty(runs)
        for i in range(runs):
            _, _, r = random_run(o, pool, spans, len(tr), bc, sk, rng)
            pfs[i], means[i] = profit_factor(r), r.mean()
        rows.append({"ticker": t, "group": "판정" if t in judged else "참고", "trades": len(tr), "pool_days": len(pool),
                     "pf": pf, "pf_pct": float((pfs < pf).mean()), "pf_rand_median": float(np.median(pfs)),
                     "pf_rand_p95": float(np.quantile(pfs, 0.95)), "mean": mean,
                     "mean_pct": float((means < mean).mean()), "mean_rand_median": float(np.median(means))})
        print(rows[-1]["ticker"], round(rows[-1]["pf_pct"], 4))
    pd.DataFrame(rows).to_csv(C.results_dir() / "monkey.csv", index=False)


if __name__ == "__main__":
    main()
