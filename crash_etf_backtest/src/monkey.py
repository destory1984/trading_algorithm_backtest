"""SPEC 4.5 / criterion 2: random-date entries, 200 runs.

Per ticker: as many trades as the rule made, (holding length, exit kind) pairs resampled from that ticker's rule
trades, entry days drawn at random from buyable days (previous day's 20-day dollar volume at or above the floor and
MA25 present, entry day traded, on or after the curve start), no overlap inside a sleeve, same fills and cost.
Days are scanned in random order and a trade is placed where its whole span is free (as in ibs_bear_backtest).
Usage: python -m src.monkey   -> results/random.csv, random_check.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C
from src.run import CLOSE, OPEN, book


def buyable(fr: pd.DataFrame, cfg: dict) -> np.ndarray:
    ok = ((fr["dv"] >= cfg["rule"]["min_dollar_volume"]) & fr["ma"].notna()).shift(1, fill_value=False)
    return (ok & fr["tradable"] & (fr.index >= pd.Timestamp(cfg["data"]["start"]))).to_numpy()


def place(rng: np.random.Generator, ok: np.ndarray, tradable: np.ndarray, lengths: np.ndarray, kinds: np.ndarray):
    n = len(ok)
    free = np.ones(n, bool)
    order = rng.permutation(np.flatnonzero(ok))
    pick = rng.integers(0, len(lengths), len(lengths))
    spans, k = [], 0
    for e in order:
        if k == len(pick):
            break
        x = e + lengths[pick[k]] - 1
        if x >= n or not tradable[x] or not free[e:x + 1].all():
            continue
        free[e:x + 1] = False
        spans.append((e - 1, e, x, kinds[pick[k]]))
        k += 1
    return sorted(spans, key=lambda s: s[1])


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    trades = pd.read_parquet(out / "trades.parquet")
    t95 = trades[trades["variant"] == "95"]
    sl = pd.read_parquet(out / "sleeves.parquet")
    years = (sl.index[-1] - sl.index[0]).days / 365.25
    rng = np.random.default_rng(cfg["checks"]["seed"])
    frames = {t: C.frame(cfg, t) for t in sl.columns}
    rows, chk = [], {"runs": 0, "short": 0, "overlap": 0, "not_buyable": 0, "placed": 0}
    for run in range(cfg["pass"]["monkey_runs"]):
        final, rets = [], []
        for t, fr in frames.items():
            tt = t95[t95["ticker"] == t]
            ok, tr = buyable(fr, cfg), fr["tradable"].to_numpy()
            kinds = np.where(tt["kind"].to_numpy() == OPEN, OPEN, CLOSE)
            spans = place(rng, ok, tr, tt["hold_days"].to_numpy(), kinds)
            chk["short"] += len(tt) - len(spans)
            chk["placed"] += len(spans)
            chk["not_buyable"] += sum(not ok[e] for _, e, _, _ in spans)
            chk["overlap"] += sum(b[1] <= a[2] for a, b in zip(spans, spans[1:]))
            tb, r, _ = book(fr, spans, cfg["cost"])
            final.append(float(np.prod(1 + r[fr.index >= sl.index[0]])))
            rets.append(tb["ret"])
        chk["runs"] += 1
        eq = float(np.mean(final))
        rows.append({"run": run, "cagr": eq ** (1 / years) - 1, "mean": pd.concat(rets).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "random.csv", index=False)
    chk["rule_trades"] = len(t95)
    pd.DataFrame([chk]).to_csv(out / "random_check.csv", index=False)
    rule = C.cagr(pd.read_parquet(out / "daily.parquet")["95"].dropna())
    print(chk)
    print(f"rule cagr {rule:+.4f} mean {t95['ret'].mean():+.4f} | random cagr median {df['cagr'].median():+.4f} "
          f"p95 {df['cagr'].quantile(0.95):+.4f} | share of random below rule {(df['cagr'] < rule).mean():.3f} | "
          f"random mean trade median {df['mean'].median():+.4f}, below rule {(df['mean'] < t95['ret'].mean()).mean():.3f}")


if __name__ == "__main__":
    main()
