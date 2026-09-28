"""Price frames from us_data and the ^IRX cash rate.

frames  krxbt.us ticker frames (dividend-adjusted OHLC on the SPX calendar)
        cut at data.end, plus an `ibs` column (NaN where high == low) and
        `raw_open` / `raw_close` (unadjusted, for the claims gap test: the
        adjustment factor adj_close / close drifts ~1e-7 per day for dividend
        payers, which makes the adjusted open == previous close test miss
        real gap days).
irx     13-week T-bill yield (^IRX, quoted in percent) as an annual decimal.
        fetch_us keeps tradeable tickers only (an index has volume 0 and
        ticker_frame would drop every day), so ^IRX is downloaded here with
        yfinance and cached in results/irx.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import us as kus

from .common import ibs, results_dir


def frames(cfg: dict, tickers: list[str], end: str | None = None) -> dict[str, pd.DataFrame]:
    cal = kus.load_calendar(cfg)
    idx = kus.index_features(cfg)
    end = end or cfg["data"].get("end")
    out = {}
    for t in tickers:
        f = kus.ticker_frame(cfg, t, cal, idx)
        if f is None:
            raise FileNotFoundError(f"no us_data prices for {t}; run python -m krxbt.fetch_us --dir ../../us_data --tickers {t}")
        raw = kus.load_prices(cfg, t)
        if raw is not None:
            raw = raw[~raw.index.duplicated(keep="last")]
            f["raw_open"] = raw["open"].reindex(f.index)
            f["raw_close"] = raw["close"].reindex(f.index)
        if end:
            f = f[f.index <= pd.Timestamp(end)]
        f["ibs"] = ibs(f["high"], f["low"], f["close"])
        out[t] = f
    return out


def index_close(cfg: dict, name: str) -> pd.Series:
    return kus.load_index(cfg, name)["close"]


def irx(cfg: dict, refresh: bool = False) -> pd.Series:
    path = results_dir() / cfg["cash"]["irx_file"]
    if path.exists() and not refresh:
        return pd.read_parquet(path)["rate"]
    import yfinance as yf
    h = yf.Ticker("^IRX").history(start=cfg["cash"]["irx_start"], auto_adjust=False, actions=False)
    if h.empty:
        raise RuntimeError("yfinance returned no ^IRX rows")
    ix = pd.DatetimeIndex(h.index.tz_localize(None).normalize(), name="date")
    s = pd.Series(h["Close"].to_numpy(np.float64) / 100.0, ix, name="rate").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.to_frame().to_parquet(path)
    return s


def cash_factors(cal: pd.DatetimeIndex, rate: pd.Series | None, lag: int, days: int) -> np.ndarray:
    """Growth factor of idle cash on each day of `cal`, applied before that day's open.

    Day i earns rate / days, where rate is the ^IRX value `lag` trading days before cal[i]
    (lag 1 = the previous day's close: the spec; lag 0 = the same day: the original package,
    used only by the crosscheck). ^IRX holidays carry the last value forward. Day 0 is 1.0.
    rate None -> all 1.0 (no interest)."""
    f = np.ones(len(cal))
    if rate is None:
        return f
    r = rate.sort_index().reindex(cal, method="ffill").bfill().fillna(0.0)
    if lag:
        r = r.shift(lag).bfill()
    f[1:] = 1.0 + r.to_numpy()[1:] / days
    return f
