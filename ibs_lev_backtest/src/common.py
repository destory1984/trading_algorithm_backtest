"""Shared helpers: config, IBS, per-ticker costs, statistics, markdown."""
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


def ticker_costs(cfg: dict, ticker: str, with_costs: bool) -> dict:
    """Commission from us.costs plus the ticker's one-way slippage (us.slippage), or all zero."""
    c = copy.deepcopy(cfg)
    if with_costs:
        c["costs"]["slippage"] = float(cfg["us"]["slippage"][ticker])
    else:
        c["costs"] = dict(ZERO_COSTS)
    return c


def with_exec(cfg: dict, mode: str) -> dict:
    """open = previous day's IBS, filled at today's open (the original); close = signal-day close in,
    exit-signal-day close out (the same meaning as ibs_backtest)."""
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


def ibs(high, low, close) -> np.ndarray:
    """(close - low) / (high - low); NaN where high == low, so no signal (the original's rule)."""
    h, l, c = (np.asarray(x, np.float64) for x in (high, low, close))
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rng == 0, np.nan, (c - l) / rng)


def tstat(x) -> dict:
    """Mean, t value (mean / (std / sqrt(n)), ddof 1) and count of the finite values."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else np.nan, "t": np.nan, "n": n}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"mean": float(x.mean()), "t": float(x.mean() / se) if se > 0 else np.nan, "n": n}


def stats(daily: pd.Series, exposure: float | None = None) -> dict:
    """CAGR over calendar days, Sharpe = mean / std x sqrt(252) with no risk-free rate (the
    original's metrics.py), max drawdown of the compounded curve."""
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
        "profit_factor": float(win.sum() / -loss.sum()) if len(loss) else np.inf,
        "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan,
    }


def daily_returns(eq: pd.Series, capital: float) -> pd.Series:
    prev = eq.shift(1)
    prev.iloc[0] = capital
    return eq / prev - 1


def capital_usd(cfg: dict, krw: float | None = None) -> float:
    t = cfg["tax"]
    return float(krw if krw is not None else t["base_capital_krw"]) / float(t["fx"])


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


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format by column name: *_mean / mean -> % with 3 decimals; *_t, t, rank_corr, split_half,
    sharpe*, mean_corr, k* -> 2 decimals; cagr*, mdd*, *_share, exposure -> %."""
    out = df.copy()
    for c in out.columns:
        s = str(c)
        if s == "mean" or s.endswith("_mean"):
            out[c] = out[c].map(lambda v: pct(v, 3))
        elif s == "t" or s.endswith("_t") or s in ("rank_corr", "split_half", "mean_corr") or s.startswith(("sharpe", "k")):
            out[c] = out[c].map(num)
        elif s.startswith(("cagr", "mdd")) or s.endswith("_cagr") or s.endswith("_mdd"):
            out[c] = out[c].map(pct)
        elif s.endswith("_share") or s == "exposure":
            out[c] = out[c].map(share)
    return out
