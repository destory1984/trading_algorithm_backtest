"""Shared helpers: config, cost and execution variants, statistics, markdown.

Copied from ibs_lev_backtest/src/common.py: load_config, with_exec, results_dir, stats, trade_stats (extended with
pf_ex_best), daily_returns, capital_usd, pct, share, num, md_table. Copied from ibs_backtest/src/strategy.py:
in_position. New here: cost_set, with_costs, profit_factor, pf_ex_best, one.
"""
from __future__ import annotations

import copy
from pathlib import Path

import krxbt.config
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZERO_COSTS = {"buy_fee": 0.0, "sell_fee": 0.0, "sell_tax": 0.0, "slippage": 0.0}


def load_config(path: Path | str | None = None) -> dict:
    cfg = krxbt.config.load_config(path or ROOT / "config.yaml")
    cfg["costs"] = dict(cfg["us"]["costs"])  # the engine reads cfg["costs"]
    return cfg


def cost_set(cfg: dict, name: str) -> dict:
    """us = us.costs (commission 0.07% + slippage 0.05% one way); orig = the original's 5bp one way on the fill
    price, no commission; zero = no costs."""
    if name == "us":
        return dict(cfg["us"]["costs"])
    if name == "orig":
        return {**ZERO_COSTS, "slippage": float(cfg["original"]["slippage"])}
    if name == "zero":
        return dict(ZERO_COSTS)
    raise ValueError(f"unknown cost set {name!r}")


def with_costs(cfg: dict, name: str) -> dict:
    c = copy.deepcopy(cfg)
    c["costs"] = cost_set(cfg, name)
    return c


def with_exec(cfg: dict, mode: str) -> dict:
    """open = signal on day s, buy at s+1's open; exit condition on day d, sell at d+1's open (the original).
    close = buy at the signal day's close, sell at the exit-condition day's close."""
    if mode not in ("close", "open"):
        raise ValueError(mode)
    c = copy.deepcopy(cfg)
    c["rules"]["entry_on_signal_close"] = mode == "close"
    c["rules"]["target_exit_on_close"] = mode == "close"
    return c


def results_dir(sub: str | None = None) -> Path:
    d = ROOT / "results" / sub if sub else ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profit_factor(ret) -> float:
    """Sum of gains / sum of losses; inf without a losing trade, NaN without trades."""
    ret = np.asarray(ret, np.float64)
    if not len(ret):
        return np.nan
    loss = -ret[ret < 0].sum()
    return float(ret[ret > 0].sum() / loss) if loss > 0 else np.inf


def pf_ex_best(ret) -> float:
    """PF without the single largest trade (the original's pfnb)."""
    ret = np.asarray(ret, np.float64)
    if len(ret) < 2:
        return np.nan
    return profit_factor(np.delete(ret, int(np.argmax(ret))))


def stats(daily: pd.Series, exposure: float | None = None) -> dict:
    """CAGR over calendar days, Sharpe = mean x 252 / (std x sqrt(252)) with no risk-free rate, max drawdown of the
    compounded daily curve."""
    eq = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    total = eq.iloc[-1] - 1
    sd = daily.std()
    return {
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "mdd": float((eq / eq.cummax() - 1).min()),
        "sharpe": float(daily.mean() * 252 / (sd * np.sqrt(252))) if sd > 0 else 0.0,
        "total_return": float(total),
        "exposure": exposure if exposure is not None else 1.0,
    }


def trade_stats(ret) -> dict:
    ret = np.asarray(ret, np.float64)
    win, loss = ret[ret > 0], ret[ret < 0]
    return {
        "trades": len(ret),
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "expectancy": float(ret.mean()) if len(ret) else np.nan,
        "profit_factor": profit_factor(ret),
        "pf_ex_best": pf_ex_best(ret),
        "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan,
    }


def daily_returns(eq: pd.Series, capital: float) -> pd.Series:
    prev = eq.shift(1)
    prev.iloc[0] = capital
    return eq / prev - 1


def in_position(cal: pd.DatetimeIndex, tr: pd.DataFrame) -> pd.Series:
    """Days whose return depends on a position: entry day (after an open entry) or the day after a close entry,
    through the exit day."""
    pos = np.zeros(len(cal), bool)
    ix = {d: i for i, d in enumerate(cal)}
    for e, x, s in zip(tr["entry_date"], tr["exit_date"], tr["signal_date"]):
        i0 = ix.get(e)
        i1 = ix.get(x)
        if i0 is None or i1 is None:
            continue
        if e == s:  # bought at the close: exposed from the next day
            i0 += 1
        pos[i0:i1 + 1] = True
    return pd.Series(pos, cal)


def capital_usd(cfg: dict, krw: float | None = None) -> float:
    t = cfg["tax"]
    return float(krw if krw is not None else t["base_capital_krw"]) / float(t["fx"])


def one(df: pd.DataFrame, **eq) -> pd.Series | None:
    """The single row where every column equals its value, or None. Callers treat None as a failed criterion."""
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).to_numpy()
    x = df[m]
    return x.iloc[0] if len(x) == 1 else None


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def share(v) -> str:
    return "" if pd.isna(v) else f"{v:.1%}"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
