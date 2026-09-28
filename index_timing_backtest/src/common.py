"""Config, sealed price loading and helpers.

Prices per market, sealed at markets.<m>.end right after reading (asserted):
  US  SPY from us_data, dividend-adjusted like krxbt.us.ticker_frame (open x adj_close / close, close = adj_close)
  KR  KOSPI price index from krx_data plus markets.KR.dividend spread evenly: the close level is multiplied by
      (1 + dividend)^(n / 252) after n trading days and the open by the previous day's factor (the dividend accrues
      at the close)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
MARKETS = ("US", "KR")
STRETCHES = ("full", "first", "second")
NAMES = {"full": "전체", "first": "전반 2008 → 2016", "second": "후반 2017 → 끝"}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def end(cfg: dict, m: str) -> pd.Timestamp:
    return pd.Timestamp(cfg["markets"][m]["end"])


def seal(df: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    out = df[df.index <= last]
    assert len(out) and out.index[-1] <= last, "seal failed"
    return out


def _dir(cfg: dict, key: str, env: str) -> Path:
    e = os.environ.get(env)
    return Path(e) if e else (ROOT / cfg["data"][key]).resolve()


def raw(cfg: dict, m: str) -> pd.DataFrame:
    """The sealed raw file: SPY prices (US) or the KOSPI index (KR)."""
    if m == "US":
        p = _dir(cfg, "us_dir", "US_DATA_DIR") / "prices" / "SPY.parquet"
    else:
        p = _dir(cfg, "kr_dir", "KRX_DATA_DIR") / "index_KOSPI.parquet"
    df = pd.read_parquet(p)
    return seal(df[~df.index.duplicated(keep="last")], end(cfg, m))


def prices(cfg: dict, m: str, dividend: float | None = "config", table: pd.DataFrame | None = None) -> pd.DataFrame:
    """open / close used for trading and valuation. `table` replaces the raw file (look-ahead check)."""
    r = raw(cfg, m) if table is None else table
    if m == "US":
        k = r["adj_close"] / r["close"]
        return pd.DataFrame({"open": r["open"] * k, "close": r["adj_close"]})
    dv = cfg["markets"]["KR"]["dividend"] if dividend == "config" else dividend
    f = (1 + (dv or 0.0)) ** (np.arange(len(r)) / 252)
    prev = np.concatenate([[1.0], f[:-1]])
    opn = r["open"].where(r["open"] > 0, r["close"].shift(1))
    return pd.DataFrame({"open": opn * prev, "close": r["close"] * f}, index=r.index)


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


def risk_weight(hold: pd.Series, target: float, iters: int = 100) -> float:
    if target <= mdd(hold):
        return 1.0
    if target >= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        w = (lo + hi) / 2
        lo, hi = (w, hi) if mdd(w * hold) > target else (lo, w)
    return (lo + hi) / 2


def stretch_mask(cfg: dict, idx: pd.DatetimeIndex, name: str) -> np.ndarray:
    split = pd.Timestamp(cfg["data"]["split"])
    if name == "first":
        return np.asarray(idx <= split)
    if name == "second":
        return np.asarray(idx > split)
    return np.ones(len(idx), bool)


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
