"""BNF rule on top of the krxbt engine.

  signal   close disparity (close / MA25 * 100) <= threshold on an eligible
           day, plus the market filter: none / uptrend (index close > index
           MA120) / crashNN (index disparity on the signal day <= NN; plain
           "crash" uses config index_disparity_crash)
  target   close >= MA25
  entry, stop, max hold, costs: krxbt.engine

Usage:  python -m src.simulate 005930 [threshold] [stop] [hold] [none|uptrend|crash97|...]
        prints every trade with the bars around it, for hand checking.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from krxbt import engine
from krxbt.costs import net_return  # noqa: F401  (used by checks, divergence)
from krxbt.engine import run_kernel  # noqa: F401  (used by divergence)


def arrays(f: pd.DataFrame) -> dict:
    a = engine.arrays(f)
    with np.errstate(invalid="ignore"):
        a["target"] = a["close"] >= a["ma"]  # NaN compares False
    return a


def candidates(a: dict, threshold: float, market_filter: str) -> np.ndarray:
    return a["eligible"] & (a["disp"] <= threshold) & engine.market_mask(a, market_filter)


def simulate(f: pd.DataFrame, cfg: dict, threshold: float, stop: float | None, hold: int,
             market_filter: str = "none", a: dict | None = None,
             independent: bool = False) -> pd.DataFrame:
    a = a or arrays(f)
    return engine.run(a, cfg, candidates(a, threshold, market_filter), a["target"], stop, hold,
                      independent, extra={"disp": a["disp"]})


def main() -> None:
    from krxbt.frame import index_features, ticker_frame

    from .common import load_calendar, load_config
    from .grid import universe_tickers

    cfg = load_config()
    t = sys.argv[1] if len(sys.argv) > 1 else "005930"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 85
    stop = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "none" else None
    hold = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    info = universe_tickers(cfg).loc[t]
    f = ticker_frame(cfg, t, info["market"], load_calendar(cfg), index_features(cfg), bool(info["delisted"]))
    tr = simulate(f, cfg, thr, stop, hold, sys.argv[5] if len(sys.argv) > 5 else "none")
    pd.set_option("display.width", 200)
    print(f"{t} {info['market']} threshold={thr} stop={stop} hold={hold}: {len(tr)} trades")
    print(tr.round(4).to_string())
    for _, row in tr.head(3).iterrows():
        lo = f.index.get_loc(row["signal_date"])
        hi = f.index.get_loc(row["exit_date"])
        print(f"\n--- trade signal {row['signal_date'].date()} -> exit {row['exit_date'].date()} ({row['reason']})")
        print(f.iloc[lo: hi + 2][["open", "high", "low", "close", "ma", "disp", "valid"]].round(1).to_string())


if __name__ == "__main__":
    main()
