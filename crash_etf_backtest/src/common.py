"""Config, sealed price loading and helpers (metric helpers copied from vol_target_history/src/common.py).

Prices: Yahoo daily bars from hist_data/country, sealed at data.end right after reading (asserted).
Open/high/low are multiplied by adj_close / close (same as krxbt.us.ticker_frame); the signal uses adj_close.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> Path:
    e = os.environ.get("HIST_DATA_DIR")
    base = Path(e) if e else (ROOT / cfg["data"]["hist_dir"]).resolve()
    return base / cfg["data"]["sub"]


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw(cfg: dict, ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir(cfg) / f"{ticker}.parquet")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    last = pd.Timestamp(cfg["data"]["end"])
    out = df[df.index <= last]
    assert len(out) and out.index[-1] <= last, "seal failed"
    return out


def tickers(cfg: dict) -> list[str]:
    """Judged tickers: in the config list, file present, first bar not after latest_first_day (SPEC 2)."""
    lim = pd.Timestamp(cfg["data"]["latest_first_day"])
    out = []
    for t in cfg["tickers"]:
        p = data_dir(cfg) / f"{t}.parquet"
        if p.exists() and raw(cfg, t).index[0] <= lim:
            out.append(t)
    return out


def frame(cfg: dict, ticker: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adjusted bars plus the columns the rule needs. `df` lets checks pass perturbed raw bars."""
    r = cfg["rule"]
    df = raw(cfg, ticker) if df is None else df
    f = df["adj_close"] / df["close"]
    out = pd.DataFrame({"open": df["open"] * f, "close": df["adj_close"], "raw_close": df["close"],
                        "volume": df["volume"]}, index=df.index)
    out["ma"] = out["close"].rolling(r["ma"]).mean()
    out["disp"] = out["close"] / out["ma"]
    out["dv"] = (df["close"] * df["volume"]).rolling(r["dv_window"]).mean()
    out["tradable"] = (df["volume"] > 0) & np.isfinite(out["open"]) & (out["open"] > 0)
    return out


def cagr(d: pd.Series) -> float:
    last = float((1 + d).prod())
    years = (d.index[-1] - d.index[0]).days / 365.25
    return last ** (1 / years) - 1 if last > 0 else -1.0


def mdd(d: pd.Series) -> float:
    eq = (1 + d).cumprod()
    return float(min((eq / eq.cummax() - 1).min(), 0.0))


def sharpe(d: pd.Series, per_year: float = 252) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0


def cut(d: pd.Series, lo: str, hi: str) -> pd.Series:
    return d[(d.index >= pd.Timestamp(lo)) & (d.index <= pd.Timestamp(hi))]


def matched_weight(strategy: pd.Series, hold: pd.Series) -> float:
    """Weight w <= 1 so that w * hold (rebalanced daily, rest in cash, no cost) has the strategy's max drawdown."""
    target = mdd(strategy)
    if mdd(hold) >= target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mdd(hold * mid) < target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


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
