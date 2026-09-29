"""US prices: us_data (TQQQ, SOXL, SOXX, QQQ, SPY, EFA, TLT) and hist_data (SCZ, AGG, BIL, QQQ_1999), sealed at
us.end, dividend-adjusted like krxbt.us.ticker_frame (open / high / low x adj_close / close, close = adj_close).
Synthetic 3x QQQ (SPEC 2): close_t = close_{t-1} x (1 + 3 r_t - fee / 252) with r_t the adjusted QQQ return; the
open, high and low are the previous synthetic close x (1 + 3 x (QQQ open, high, low / previous QQQ close - 1)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C

HIST = {"SCZ", "AGG", "BIL", "QQQ_1999"}


def raw(cfg: dict, t: str) -> pd.DataFrame:
    u = cfg["us"]
    d = C.env_dir(u["hist_dir"], "HIST_DATA_DIR") if t in HIST else C.env_dir(u["dir"], "US_DATA_DIR") / "prices"
    df = pd.read_parquet(d / f"{t}.parquet")
    return C.seal(df[~df.index.duplicated(keep="last")].sort_index(), u["end"])


def adjusted(df: pd.DataFrame) -> pd.DataFrame:
    k = df["adj_close"] / df["close"]
    return pd.DataFrame({"open": df["open"] * k, "high": df["high"] * k, "low": df["low"] * k, "close": df["adj_close"]})


def prices(cfg: dict, t: str, tables: dict | None = None) -> pd.DataFrame:
    df = tables[t] if tables and t in tables else raw(cfg, t)
    return adjusted(df)


def synthetic(cfg: dict, tables: dict | None = None) -> pd.DataFrame:
    u = cfg["us"]
    q = prices(cfg, "QQQ_1999", tables)
    lev, fee = u["synth_leverage"], u["synth_fee"]
    r = q["close"].pct_change().fillna(0.0).to_numpy()
    close = np.cumprod(1 + lev * r - np.where(np.arange(len(r)) > 0, fee / 252, 0.0)) * 100.0
    prev_s = np.concatenate([[100.0], close[:-1]])
    prev_q = np.concatenate([[q["close"].iloc[0]], q["close"].to_numpy()[:-1]])
    rel = lambda col: np.maximum(prev_s * (1 + lev * (q[col].to_numpy() / prev_q - 1)), 1e-9)
    o, h, l = rel("open"), rel("high"), rel("low")
    out = pd.DataFrame({"open": o, "high": np.maximum.reduce([o, h, l, close]), "low": np.minimum.reduce([o, h, l, close]),
                        "close": close}, index=q.index)
    return out
