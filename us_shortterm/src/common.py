"""Config, sealed data loading and small shared helpers.

The seal (SPEC 2, principle 1): every price or index table is cut at data.screen_end right after it is read, and the
cut table's last date is asserted. krxbt.us.ticker_frame reads files through krxbt.us.load_prices / load_index, so
load_frames swaps those two for sealed versions while it builds the frames; the engine code itself is unchanged.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import krxbt.config
import krxbt.engine
import krxbt.us
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZERO_COSTS = {"buy_fee": 0.0, "sell_fee": 0.0, "sell_tax": 0.0, "slippage": 0.0}

_raw_prices = krxbt.us.load_prices
_raw_index = krxbt.us.load_index


def load_config(path: Path | str | None = None) -> dict:
    cfg = krxbt.config.load_config(path or ROOT / "config.yaml")
    cfg["costs"] = dict(cfg["us"]["costs"])  # the engine reads cfg["costs"]
    return cfg


def screen_end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["data"]["screen_end"])


def seal(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Drop every row after `end` and assert it."""
    out = df[df.index <= end]
    assert len(out) and out.index[-1] <= end, f"seal failed: last date {out.index[-1] if len(out) else None}"
    return out


@contextmanager
def sealed_loaders(cfg: dict, overrides: dict[str, pd.DataFrame] | None = None):
    """Inside the block krxbt.us reads sealed tables. `overrides` replaces a ticker's raw price table (checks.py)."""
    end = screen_end(cfg)
    overrides = overrides or {}

    def prices(c, ticker):
        px = overrides[ticker] if ticker in overrides else _raw_prices(c, ticker)
        return None if px is None else seal(px, end)

    def index(c, name):
        return seal(_raw_index(c, name), end)

    with mock.patch.object(krxbt.us, "load_prices", prices), mock.patch.object(krxbt.us, "load_index", index):
        yield


def raw_prices(cfg: dict, ticker: str) -> pd.DataFrame:
    """The sealed raw file (split-adjusted OHLC + adj_close), for hand checks."""
    return seal(_raw_prices(cfg, ticker), screen_end(cfg))


def load_frames(cfg: dict, overrides: dict[str, pd.DataFrame] | None = None):
    """(calendar, {ticker: frame}) for screen.tickers, all sealed."""
    end = screen_end(cfg)
    with sealed_loaders(cfg, overrides):
        cal = krxbt.us.load_calendar(cfg)
        idx = krxbt.us.index_features(cfg)
        frames = {t: krxbt.us.ticker_frame(cfg, t, cal, idx) for t in cfg["screen"]["tickers"]}
    assert cal[-1] <= end
    for t, f in frames.items():
        assert f is not None, f"no data for {t}"
        assert f.index[-1] <= end, f"{t} ends {f.index[-1]}"
    return cal, frames


def with_rules(cfg: dict, entry_on_close: bool, exit_on_close: bool) -> dict:
    c = copy.deepcopy(cfg)
    c["rules"]["entry_on_signal_close"] = entry_on_close
    c["rules"]["target_exit_on_close"] = exit_on_close
    return c


def with_costs(cfg: dict, costs: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["costs"] = dict(costs)
    return c


def one_way(bp: float) -> dict:
    """A single one-way cost of `bp` basis points on both sides (used for the breakeven search)."""
    return {**ZERO_COSTS, "buy_fee": bp / 1e4, "sell_fee": bp / 1e4}


def results_dir(sub: str | None = "screen") -> Path:
    d = ROOT / "results" / sub if sub else ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def share(v) -> str:
    return "" if pd.isna(v) else f"{v:.1%}"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else ("inf" if np.isinf(v) else f"{v:.{d}f}")


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
