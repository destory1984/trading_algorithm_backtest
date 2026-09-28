"""US data (krxbt.fetch_us) as ticker frames the engine can run on.

Same columns as krxbt.frame.ticker_frame, with US rules:
  - prices are dividend-adjusted: open/high/low/close are scaled by
    adj_close / close, so special dividends (KDP 2018: -82% on close) do not
    look like crashes
  - no daily price limit, no delisting sale, no break blocking: data_break is
    always False
  - valid = volume > 0 (zero-volume stretches are OTC quotes before a US
    listing, e.g. FER, POET)
  - eligible = valid, MA ready, listing_days >= universe.min_listing_days,
    avg traded value >= us.min_avg_value (USD), on or after data.start
  - market columns (idx_regime, idx_disp, idx_crash) come from us.market_index

Config section `us`: dir, market_index (NDX / DJI / SPX), min_avg_value.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def us_dir(cfg: dict) -> Path:
    env = os.environ.get("US_DATA_DIR")
    return (Path(env) if env else Path(cfg.get("_root", ".")) / cfg["us"]["dir"]).resolve()


def load_index(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(us_dir(cfg) / f"index_{name}.parquet")


def load_calendar(cfg: dict) -> pd.DatetimeIndex:
    """Trading calendar = dates of the S&P 500 index."""
    return pd.DatetimeIndex(load_index(cfg, "SPX").index)


def load_universe(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(us_dir(cfg) / "universe.parquet").set_index("ticker")


def load_prices(cfg: dict, ticker: str) -> pd.DataFrame | None:
    p = us_dir(cfg) / "prices" / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def index_features(cfg: dict) -> pd.DataFrame:
    ind = cfg["indicators"]
    c = load_index(cfg, cfg["us"]["market_index"])["close"]
    return pd.DataFrame({
        "idx_close": c,
        "idx_regime": c > c.rolling(ind["index_ma_window"], min_periods=ind["index_ma_window"]).mean(),
        "idx_disp": c / c.rolling(ind["ma_window"], min_periods=ind["ma_window"]).mean() * 100,
    })


def ticker_frame(cfg: dict, ticker: str, cal: pd.DatetimeIndex, idx: pd.DataFrame) -> pd.DataFrame | None:
    px = load_prices(cfg, ticker)
    if px is None or px.empty:
        return None
    ind, uni = cfg["indicators"], cfg["universe"]
    px = px[~px.index.duplicated(keep="last")]
    value = px["close"] * px["volume"]  # split-adjusted close x split-adjusted volume = dollars traded
    k = px["adj_close"] / px["close"]
    px = px.assign(open=px["open"] * k, high=px["high"] * k, low=px["low"] * k, close=px["adj_close"])
    valid = (px["volume"] > 0) & (px[["open", "high", "low", "close"]] > 0).all(axis=1)
    v = px[valid].copy()
    w = ind["ma_window"]
    if len(v) < w:
        return None
    v["ma"] = v["close"].rolling(w, min_periods=w).mean()
    v["disp"] = v["close"] / v["ma"] * 100
    v["avg_value"] = value[valid].rolling(uni["avg_value_window"], min_periods=uni["avg_value_window"]).mean()
    v["listing_days"] = np.arange(1, len(v) + 1)
    days = cal[(cal >= v.index[0]) & (cal <= v.index[-1])]
    f = v.reindex(days)
    f["valid"] = f["close"].notna()
    f["market"] = cfg["us"]["market_index"]
    f = f.join(idx[["idx_regime", "idx_disp"]], how="left")
    f["idx_regime"] = f["idx_regime"].fillna(False).astype(bool)
    f["idx_crash"] = (f["idx_disp"] <= ind["index_disparity_crash"]).fillna(False).astype(bool)
    f["eligible"] = (f["valid"] & f["disp"].notna() & (f["avg_value"] >= cfg["us"]["min_avg_value"])
                     & (f["listing_days"] >= uni["min_listing_days"]) & (f.index >= pd.Timestamp(cfg["data"]["start"])))
    f["data_break"] = False
    return f
