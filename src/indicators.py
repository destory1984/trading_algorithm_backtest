"""Stage 2: MA25, disparity, liquidity and market-regime columns per ticker.

Every ticker frame is laid on the market calendar (KOSPI index dates) from its
first to its last row. Days with volume 0 (trading halt) stay in the frame with
valid=False so that holding days keep counting, but nothing is ever priced or
exited on them.

Usage:  python -m src.indicators 005930   # prints a sample around a signal
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import load_calendar, load_config, load_index, load_prices


def index_features(cfg: dict) -> dict[str, pd.DataFrame]:
    ind = cfg["indicators"]
    out = {}
    for m in cfg["data"]["markets"]:
        ix = load_index(cfg, m)
        c = ix["close"]
        out[m] = pd.DataFrame({
            "idx_close": c,
            "idx_regime": c > c.rolling(ind["index_ma_window"], min_periods=ind["index_ma_window"]).mean(),
            "idx_disp": c / c.rolling(ind["ma_window"], min_periods=ind["ma_window"]).mean() * 100,
        })
    return out


def ticker_frame(cfg: dict, ticker: str, market: str, cal: pd.DatetimeIndex,
                 idx: dict[str, pd.DataFrame], delisted: bool = False) -> pd.DataFrame | None:
    px = load_prices(cfg, ticker)
    if px is None or px.empty:
        return None
    ind, uni = cfg["indicators"], cfg["universe"]
    px = px[~px.index.duplicated(keep="last")]
    valid = (px["volume"] > 0) & (px["open"] > 0) & (px["high"] > 0) & (px["low"] > 0) & (px["close"] > 0)
    v = px[valid].copy()
    if len(v) < ind["ma_window"]:
        return None
    w = ind["ma_window"]
    v["ma"] = v["close"].rolling(w, min_periods=w).mean()
    v["disp"] = v["close"] / v["ma"] * 100
    v["avg_value"] = (v["close"] * v["volume"]).rolling(uni["avg_value_window"], min_periods=uni["avg_value_window"]).mean()
    v["listing_days"] = np.arange(1, len(v) + 1)

    # Calendar rows between the first and last *traded* day. Trailing halt rows
    # (delisted names are often frozen for weeks) are dropped: the last trade
    # is the last real price.
    days = cal[(cal >= v.index[0]) & (cal <= v.index[-1])]
    f = v.reindex(days)
    f["valid"] = f["close"].notna()
    f["market"] = market
    f = f.join(idx[market][["idx_regime", "idx_disp"]], how="left")
    f["idx_regime"] = f["idx_regime"].fillna(False).astype(bool)

    liquid = f["avg_value"] >= uni["min_avg_value"]
    listed = f["listing_days"] >= uni["min_listing_days"]
    in_period = f.index >= pd.Timestamp(cfg["data"]["start"])
    f["eligible"] = f["valid"] & f["disp"].notna() & liquid & listed & in_period
    # Market days skipped (halted) right before each traded day.
    pos = f.index.get_indexer(v.index)
    halted_before = pd.Series(np.diff(pos, prepend=pos[0]) - 1, index=v.index).clip(lower=0)
    resumed = halted_before > uni.get("long_halt_days", 3)
    hit = resumed.copy()
    brk = uni.get("price_break_drop")
    if brk is not None:
        chg = v["close"] / v["close"].shift(1) - 1
        hit |= (chg < brk) & ~resumed
    # Block the break / resume day and the next ma_window-1 traded days, until
    # MA25 no longer mixes prices from both sides of the break.
    blocked = hit.astype(int).rolling(w, min_periods=1).max().astype(bool)
    f.loc[blocked.reindex(f.index, fill_value=False).to_numpy(), "eligible"] = False
    n_sale = uni.get("delisting_sale_days", 0)
    if delisted and n_sale:
        f.loc[f.index >= v.index[-min(n_sale, len(v))], "eligible"] = False
    return f


def main() -> None:
    cfg = load_config()
    t = sys.argv[1] if len(sys.argv) > 1 else "005930"
    cal = load_calendar(cfg)
    f = ticker_frame(cfg, t, "KOSPI", cal, index_features(cfg))
    cols = ["open", "close", "ma", "disp", "avg_value", "valid", "eligible", "idx_regime", "idx_disp"]
    print(f[cols].describe().T[["count", "mean", "min", "max"]])
    low = f[f["eligible"]]["disp"].idxmin()
    print(f"\nlowest disparity day {low.date()}:")
    print(f.loc[low - pd.Timedelta(days=5): low + pd.Timedelta(days=5), cols].round(2).to_string())


if __name__ == "__main__":
    main()
