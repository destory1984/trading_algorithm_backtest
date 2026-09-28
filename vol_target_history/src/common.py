"""Config, sealed price loading and helpers (the metric helpers are copied from index_timing_backtest/src/common.py).

Prices: ^GSPC close from hist_data, sealed at data.end right after reading (asserted). Execution is at the next
day's close (SPEC 2.3), so only the close is used. The total-return approximation multiplies the close by
(1 + dividend)^(calendar days since the first row / 365.25) (SPEC 2.1).
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


def hist_dir(cfg: dict) -> Path:
    e = os.environ.get("HIST_DATA_DIR")
    return Path(e) if e else (ROOT / cfg["data"]["hist_dir"]).resolve()


def seal(df: pd.DataFrame, last: pd.Timestamp) -> pd.DataFrame:
    out = df[df.index <= last]
    assert len(out) and out.index[-1] <= last, "seal failed"
    return out


def raw(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(hist_dir(cfg) / cfg["data"]["file"])
    return seal(df[~df.index.duplicated(keep="last")].sort_index(), pd.Timestamp(cfg["data"]["end"]))


def total_return(close: pd.Series, dividend: float) -> pd.Series:
    days = (close.index - close.index[0]).days.to_numpy()
    return close * (1 + dividend) ** (days / 365.25)


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


def sharpe(d: pd.Series, per_year: float = 252) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0


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
