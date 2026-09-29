"""Config, paths, costs and metric helpers shared by the Korean and US parts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from krxbt.config import load_config as _load

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    return _load(ROOT / "config.yaml")


def results_dir(sub: str | None = None) -> Path:
    d = ROOT / "results" / sub if sub else ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def env_dir(cfg_value: str, env: str) -> Path:
    e = os.environ.get(env)
    return Path(e) if e else (ROOT / cfg_value).resolve()


def seal(df: pd.DataFrame, last) -> pd.DataFrame:
    out = df[df.index <= pd.Timestamp(last)]
    assert len(out) and out.index[-1] <= pd.Timestamp(last), "seal failed"
    return out


def sell_tax(cfg: dict, day) -> float:
    cs = cfg["costs"]
    t = cs["sell_tax"]
    for since, rate in cs.get("sell_tax_schedule") or []:
        if pd.Timestamp(day) >= pd.Timestamp(since):
            t = rate
    return t


def tax_table(cfg: dict, cal: pd.DatetimeIndex) -> np.ndarray:
    return np.array([sell_tax(cfg, d) for d in cal], np.float64)


def cagr(d: pd.Series) -> float:
    last = float((1 + d).prod())
    years = (d.index[-1] - d.index[0]).days / 365.25
    return last ** (1 / years) - 1 if last > 0 and years > 0 else -1.0


def mdd(d: pd.Series) -> float:
    eq = (1 + d).cumprod()
    return float(min((eq / eq.cummax() - 1).min(), 0.0))


def sharpe(d: pd.Series, per_year: float = 252) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(per_year)) if sd > 0 else 0.0


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


def from_equity(eq: pd.Series, capital: float) -> pd.Series:
    return eq.pct_change().fillna(eq.iloc[0] / capital - 1)


def pct(v, d=1) -> str:
    return "" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def pp(v, d=1) -> str:
    return "" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%p"


def num(v, d=2) -> str:
    return "" if v is None or pd.isna(v) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if (isinstance(v, float) and pd.isna(v)) or v is None else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
