"""Read the collected data: indices, calendar, universe, prices."""
from __future__ import annotations

import pandas as pd

from .config import data_dir, price_dir


def load_index(cfg: dict, market: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir(cfg) / f"index_{market}.parquet")


def load_calendar(cfg: dict) -> pd.DatetimeIndex:
    """Trading calendar = dates of the KOSPI index."""
    return pd.DatetimeIndex(load_index(cfg, "KOSPI").index)


def load_universe(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(data_dir(cfg) / "universe.parquet")


def load_prices(cfg: dict, ticker: str) -> pd.DataFrame | None:
    p = price_dir(cfg) / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def universe_tickers(cfg: dict) -> pd.DataFrame:
    """One row per usable ticker with its latest market and delisting flag."""
    uni = load_universe(cfg)
    uni = uni[uni["excluded"].isna()].sort_values("snapshot")
    last = uni.groupby("ticker").tail(1).set_index("ticker")
    last["delisted"] = last["snapshot"] < uni["snapshot"].max()
    have = {p.stem for p in price_dir(cfg).glob("*.parquet")}
    return last[last.index.isin(have)][["market", "name", "delisted"]]
