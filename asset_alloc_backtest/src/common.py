"""Config, sealed price loading and helpers.

Prices: the five ETFs from us_data/prices, sealed at data.end right after reading (asserted), dividend-adjusted like
krxbt.us.ticker_frame: close = adj_close, open = open x adj_close / close. All five share the same trading days
(asserted), so they live in one frame per field.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
STRETCHES = ("full", "first", "second")
NAMES = {"full": "전체", "first": "전반 2008-05 → 2016", "second": "후반 2017 → 끝"}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seal(df: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    out = df[df.index <= last]
    assert len(out) and out.index[-1] <= last, "seal failed"
    return out


def data_dir(cfg: dict) -> Path:
    e = os.environ.get("US_DATA_DIR")
    return Path(e) if e else (ROOT / cfg["data"]["us_dir"]).resolve()


def raw(cfg: dict) -> dict[str, pd.DataFrame]:
    """The sealed raw files, one per ticker."""
    last = pd.Timestamp(cfg["data"]["end"])
    out = {}
    for t in cfg["data"]["tickers"]:
        df = pd.read_parquet(data_dir(cfg) / "prices" / f"{t}.parquet")
        out[t] = seal(df[~df.index.duplicated(keep="last")].sort_index(), last)
    return out


def prices(cfg: dict, tables: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(open, close) frames, columns = tickers. `tables` replaces the raw files (look-ahead check)."""
    r = raw(cfg) if tables is None else tables
    tick = cfg["data"]["tickers"]
    idx = r[tick[0]].index
    for t in tick:
        assert r[t].index.equals(idx), f"{t} trading days differ"
    k = {t: r[t]["adj_close"] / r[t]["close"] for t in tick}
    opn = pd.DataFrame({t: r[t]["open"] * k[t] for t in tick}, index=idx)
    cls = pd.DataFrame({t: r[t]["adj_close"] for t in tick}, index=idx)
    assert opn.notna().all().all() and cls.notna().all().all() and (opn > 0).all().all()
    return opn, cls


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Days whose next trading day is in another month. The last day of the data is never one (tomorrow unknown)."""
    nxt = idx[1:]
    cur = idx[:-1]
    return cur[(nxt.month != cur.month) | (nxt.year != cur.year)]


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cagr(d: pd.Series) -> float:
    last = float((1 + d).prod())
    years = (d.index[-1] - d.index[0]).days / 365.25
    return last ** (1 / years) - 1 if last > 0 else -1.0


def mdd(d: pd.Series) -> float:
    eq = (1 + d).cumprod()
    return float(min((eq / eq.cummax() - 1).min(), 0.0))


def sharpe(d: pd.Series) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def longest_underwater(d: pd.Series) -> int:
    """Longest stretch in calendar days from a peak until the curve is back at that peak (or the end)."""
    eq = (1 + d).cumprod()
    peak_day, best = eq.index[0], 0
    top = eq.iloc[0]
    for day, v in eq.items():
        if v >= top:
            top, peak_day = v, day
        else:
            best = max(best, (day - peak_day).days)
    return best


def stretch_mask(cfg: dict, idx: pd.DatetimeIndex, name: str) -> np.ndarray:
    split = pd.Timestamp(cfg["data"]["split"])
    if name == "first":
        return np.asarray(idx <= split)
    if name == "second":
        return np.asarray(idx > split)
    return np.ones(len(idx), bool)


def key(rule: str, param, cash: float, net: bool) -> str:
    return f"{rule}|{'' if param is None else param}|{cash}|{'net' if net else 'gross'}"


def parse(k: str) -> dict:
    r, p, c, n = k.split("|")
    return {"rule": r, "param": int(p) if p else None, "cash": float(c), "net": n == "net"}


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def pp(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%p"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
