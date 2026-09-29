"""Config, the sealed calendar, per-ticker frames (krxbt data-cleaning rules), costs and metric helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from krxbt.config import load_config as _load
from krxbt.data import load_calendar, load_index, universe_tickers

ROOT = Path(__file__).resolve().parent.parent
STRETCHES = ("full", "first", "second")
NAMES = {"full": "전체", "first": "전반 2008-04 → 2016", "second": "후반 2017 → 끝"}


def load_config() -> dict:
    return _load(ROOT / "config.yaml")


def end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["data"]["end"])


def calendar(cfg: dict) -> pd.DatetimeIndex:
    cal = load_calendar(cfg)
    cal = cal[cal <= end(cfg)]
    assert cal[-1] <= end(cfg)
    return cal


def tickers(cfg: dict) -> pd.DataFrame:
    return universe_tickers(cfg)


def kospi_tr(cfg: dict, cal: pd.DatetimeIndex) -> pd.Series:
    """Daily returns of the KOSPI price index plus kospi_dividend spread over trading days."""
    c = load_index(cfg, "KOSPI")["close"].reindex(cal)
    r = c.pct_change().fillna(0.0)
    return r + (1 + cfg["kospi_dividend"]) ** (1 / 252) - 1


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sell_tax(cfg: dict, day: pd.Timestamp) -> float:
    cs = cfg["costs"]
    t = cs["sell_tax"]
    for since, rate in cs.get("sell_tax_schedule") or []:
        if day >= pd.Timestamp(since):
            t = rate
    return t


def tax_table(cfg: dict, cal: pd.DatetimeIndex) -> np.ndarray:
    return np.array([sell_tax(cfg, d) for d in cal], np.float64)


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
    """w in [0, 1] so that w x hold (rebalanced daily, cash 0) has max drawdown `target`."""
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


def variants(cfg: dict) -> list[tuple[str, int, bool]]:
    """(rule, entry days, is main) for the 2 rules and 4 neighbours."""
    out = []
    for r, s in cfg["rules"].items():
        out.append((r, s["entry"], True))
        out += [(r, n, False) for n in s["neighbors"]]
    return out


def vkey(rule: str, entry: int) -> str:
    return f"{rule}_{entry}"


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
