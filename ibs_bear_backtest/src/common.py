"""Config, sealed data loading, the two rule versions and small helpers.

Copied from us_shortterm/src/common.py (seal, sealed_loaders, load_frames, with_rules, one_way, formatting) and
us_shortterm/src/indicators.py (sma, ibs). The seal cuts every price and index table at data.end right after it is
read and asserts the last date; krxbt.us.ticker_frame reads through the swapped loaders.

Rule (SPEC 3): signal IBS < rule.ibs_entry on an eligible day; version "bear" also needs close < own SMA(rule.sma)
on the signal day. Exit condition IBS > rule.ibs_exit. Next-day open fills for both.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import krxbt.config
import krxbt.us
import numpy as np
import pandas as pd
from krxbt import engine

ROOT = Path(__file__).resolve().parent.parent
ZERO_COSTS = {"buy_fee": 0.0, "sell_fee": 0.0, "sell_tax": 0.0, "slippage": 0.0}
VERSIONS = ("bear", "all")

_raw_prices = krxbt.us.load_prices
_raw_index = krxbt.us.load_index


def load_config(path: Path | str | None = None) -> dict:
    cfg = krxbt.config.load_config(path or ROOT / "config.yaml")
    cfg["costs"] = dict(cfg["us"]["costs"])  # the engine reads cfg["costs"]
    return cfg


def end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["data"]["end"])


def seal(df: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    out = df[df.index <= last]
    assert len(out) and out.index[-1] <= last, "seal failed"
    return out


@contextmanager
def sealed_loaders(cfg: dict):
    last = end(cfg)

    def prices(c, ticker):
        p = _raw_prices(c, ticker)
        return None if p is None else seal(p, last)

    def index(c, name):
        return seal(_raw_index(c, name), last)

    with mock.patch.object(krxbt.us, "load_prices", prices), mock.patch.object(krxbt.us, "load_index", index):
        yield


def raw_prices(cfg: dict, ticker: str) -> pd.DataFrame:
    return seal(_raw_prices(cfg, ticker), end(cfg))


def tickers(cfg: dict) -> list[str]:
    return list(cfg["tickers"]["judged"]) + list(cfg["tickers"]["reference"])


def load_frames(cfg: dict):
    """(calendar, {ticker: frame}, SPX close) for judged + reference tickers, sealed."""
    with sealed_loaders(cfg):
        cal = krxbt.us.load_calendar(cfg)
        idx = krxbt.us.index_features(cfg)
        frames = {t: krxbt.us.ticker_frame(cfg, t, cal, idx) for t in tickers(cfg)}
        spx = krxbt.us.load_index(cfg, "SPX")["close"]
    for t, f in frames.items():
        assert f is not None and f.index[-1] <= end(cfg), t
    return cal, frames, spx


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(close - low) / (high - low); NaN where high == low."""
    rng = high - low
    return ((close - low) / rng).where(rng != 0)


def signals(f: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    r = cfg["rule"]
    x = ibs(f["high"], f["low"], f["close"])
    below = (f["close"] < sma(f["close"], int(r["sma"]))).to_numpy()
    low = (x < r["ibs_entry"]).to_numpy()
    return {"ibs": x.to_numpy(), "below": below, "cand_all": low, "cand_bear": low & below,
            "target": (x > r["ibs_exit"]).to_numpy()}


def trades(f: pd.DataFrame, cfg: dict, version: str, a: dict | None = None) -> pd.DataFrame:
    a = a or engine.arrays(f)
    s = signals(f, cfg)
    cand = s[f"cand_{version}"] & a["eligible"]
    c = with_rules(cfg, False, False)
    return engine.run(a, c, cand, s["target"], None, int(cfg["rule"]["hold_max"]),
                      extra={"below": s["below"]})


def curve_inputs(f: pd.DataFrame, cfg: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    g = f[f.index >= pd.Timestamp(cfg["data"]["start"])]
    assert g["valid"].all(), "gap in an ETF's prices"
    return g.index, g["close"].to_numpy(np.float64)


def with_rules(cfg: dict, entry_on_close: bool, exit_on_close: bool) -> dict:
    c = copy.deepcopy(cfg)
    c["rules"]["entry_on_signal_close"] = entry_on_close
    c["rules"]["target_exit_on_close"] = exit_on_close
    return c


def one_way(bp: float) -> dict:
    return {**ZERO_COSTS, "buy_fee": bp / 1e4, "sell_fee": bp / 1e4}


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else ("inf" if np.isinf(v) else f"{v:.{d}f}")


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
