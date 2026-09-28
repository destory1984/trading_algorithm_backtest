"""SPEC 4.3: the unfiltered IBS trades split by the signal day's regime (own close vs own 200-day SMA), and the
same split by the S&P 500 vs its 200-day SMA (sensitivity). Run after run.py.
Usage: python -m src.regime   -> results/regime.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C


def split(r: np.ndarray, below: np.ndarray) -> dict:
    out = {}
    for name, k in (("above", ~below), ("below", below)):
        x = r[k]
        sd = x.std(ddof=1) if len(x) > 1 else np.nan
        out |= {f"{name}_n": len(x), f"{name}_mean": x.mean() if len(x) else np.nan,
                f"{name}_t": x.mean() / (sd / np.sqrt(len(x))) if len(x) > 1 and sd > 0 else np.nan,
                f"{name}_win": (x > 0).mean() if len(x) else np.nan}
    return out


def main() -> None:
    cfg = C.load_config()
    _, _, spx = C.load_frames(cfg)
    spx_below = spx < C.sma(spx, int(cfg["rule"]["sma"]))
    judged = set(cfg["tickers"]["judged"])
    rows = []
    for t in C.tickers(cfg):
        tr = pd.read_parquet(C.results_dir() / f"trades_{t}_all.parquet")
        r = tr["ret"].to_numpy(np.float64)
        own = tr["below"].to_numpy(bool)
        mkt = spx_below.reindex(pd.DatetimeIndex(tr["signal_date"])).to_numpy(bool)
        for basis, b in (("자기 200일선", own), ("S&P500 200일선", mkt)):
            rows.append({"ticker": t, "group": "판정" if t in judged else "참고", "basis": basis, "trades": len(r),
                         **split(r, b)})
    df = pd.DataFrame(rows)
    df["below_gt_above"] = df["below_mean"] > df["above_mean"]
    df.to_csv(C.results_dir() / "regime.csv", index=False)
    print(df[["ticker", "basis", "above_n", "above_mean", "above_t", "below_n", "below_mean", "below_t"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
