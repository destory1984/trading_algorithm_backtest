"""Shared helpers: config, IBS, daily-return statistics."""
from __future__ import annotations

import copy
from pathlib import Path

import krxbt.config
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    cfg = krxbt.config.load_config(path or ROOT / "config.yaml")
    cfg["costs"] = dict(cfg["us"]["costs"])  # the engine reads cfg["costs"]
    return cfg


def zero_costs(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["costs"] = {"buy_fee": 0.0, "sell_fee": 0.0, "sell_tax": 0.0, "slippage": 0.0}
    return c


def with_exec(cfg: dict, mode: str) -> dict:
    """close = signal-day close in, exit-signal-day close out; open = next day's open for both."""
    if mode not in ("close", "open"):
        raise ValueError(mode)
    c = copy.deepcopy(cfg)
    c["rules"]["entry_on_signal_close"] = mode == "close"
    c["rules"]["target_exit_on_close"] = mode == "close"
    return c


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ibs(high, low, close, flat: float) -> np.ndarray:
    """(close - low) / (high - low); `flat` where high == low. NaN stays NaN."""
    h, l, c = (np.asarray(x, np.float64) for x in (high, low, close))
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (c - l) / rng
    return np.where(rng == 0, flat, v)


def stats(daily: pd.Series, in_pos: pd.Series | None = None) -> dict:
    """Stats of a daily return series, as the original repo computes them:
    CAGR over calendar days, Sharpe = mean*252 / (std*sqrt(252)) with no
    risk-free rate, max drawdown of the compounded curve."""
    eq = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    total = eq.iloc[-1] - 1
    sd = daily.std()
    return {
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "mdd": float((eq / eq.cummax() - 1).min()),
        "sharpe": float(daily.mean() * 252 / (sd * np.sqrt(252))) if sd > 0 else 0.0,
        "total_return": float(total),
        "exposure": float(in_pos.mean()) if in_pos is not None else 1.0,
    }


def trade_stats(ret: np.ndarray) -> dict:
    ret = np.asarray(ret, np.float64)
    win, loss = ret[ret > 0], ret[ret < 0]
    return {
        "trades": len(ret),
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "expectancy": float(ret.mean()) if len(ret) else np.nan,
        # profit factor: sum of gains / sum of losses (the original's definition)
        "profit_factor": float(win.sum() / -loss.sum()) if len(loss) else np.inf,
        # payoff ratio: average gain / average loss
        "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan,
    }
