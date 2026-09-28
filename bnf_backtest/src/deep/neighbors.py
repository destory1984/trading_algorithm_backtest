"""SPEC2 4.4: 18 neighbouring combos (disparity x crash filter x hold, no stop) as 5-slot portfolios.
All 18 are grid combos (no new trials). Reported, never chosen from.
Usage: python -m src.deep.neighbors   -> results/deep/neighbors.csv
"""
from __future__ import annotations

import itertools

import pandas as pd

from src.deep import shared as S


def main() -> None:
    cfg = S.config()
    nb = cfg["deep"]["neighbors"]
    ind = S.candidates("ind")
    rows = []
    for thr, mf, hold in itertools.product(nb["threshold"], nb["market_filter"], nb["hold"]):
        key = S.combo_key(thr, None, hold, mf)
        tr = ind[ind["combo"] == key]
        eq, info, cmp = S.run(cfg, tr)
        full = next(r for r in cmp if r["stretch"] == "full")
        rows.append({"threshold": thr, "market_filter": mf, "hold": hold, "combo": key,
                     "cand_trades": len(tr), "cand_mean": float(tr["ret"].mean()), **full,
                     "trades_taken": info["trades_taken"]})
        print(key, f"cagr {full['cagr']:.4f} diff {full['diff_risk']:+.4f}")
    df = pd.DataFrame(rows)
    assert len(df) == 18
    df.to_csv(S.results_dir() / "neighbors.csv", index=False)


if __name__ == "__main__":
    main()
