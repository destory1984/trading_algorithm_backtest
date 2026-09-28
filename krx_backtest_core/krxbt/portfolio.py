"""Portfolio simulation over a list of candidate trades.

Input trades need: ticker, entry_date, exit_date, entry_px, ret, reason,
idx_disp and the rank column. Each trade is one signal simulated on its own
(krxbt.engine.run with independent=True).

Rules: equal weight, each new position gets 1/slots of the previous close's
equity, limited by cash. When more signals than free slots arrive on the same
day, the lowest rank value wins. A signal on a ticker already held is ignored.
Positions that exit on day d free their slot from day d+1. Equity is marked
to market at each close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_index, load_prices


def close_panel(cfg: dict, tickers, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Closes of traded days on the calendar, forward-filled over halts."""
    cols = {}
    for t in tickers:
        p = load_prices(cfg, t)
        c = p["close"].where(p["volume"] > 0)
        cols[t] = c[~c.index.duplicated(keep="last")]
    return pd.DataFrame(cols).reindex(cal).ffill()


def run_portfolio(cal: pd.DatetimeIndex, tr: pd.DataFrame, closes: pd.DataFrame, capital: float,
                  slots: int, rank_by: str, cooldown: int = 0,
                  crash_slots: int = 0, crash_level: float | None = None) -> tuple[pd.Series, dict]:
    """cooldown: after a stop-loss exit, skip that ticker for N trading days.
    crash_slots: a signal whose idx_disp <= crash_level may open a position
    while fewer than crash_slots are held, and gets 1/crash_slots of equity."""
    base = int(slots)
    wide = max(base, int(crash_slots or 0))
    day_no = {d: i for i, d in enumerate(cal)}
    blocked_until: dict[str, int] = {}  # ticker -> last day index still blocked after a stop
    ledger: list[dict] = []
    by_entry = {d: g.sort_values(rank_by) for d, g in tr.groupby("entry_date")}
    cash = float(capital)
    pos: dict[str, dict] = {}
    equity = np.empty(len(cal))
    prev_eq = cash
    taken = 0
    taken_rets: list[float] = []
    invested_days = 0
    max_held = 0
    exposure: list[float] = []  # share of equity in stocks, on days holding anything
    for i, d in enumerate(cal):
        # entries at today's open
        if d in by_entry and len(pos) < wide:
            for _, t in by_entry[d].iterrows():
                cap = wide if crash_level is not None and t["idx_disp"] <= crash_level else base
                if len(pos) >= cap:
                    continue
                tk = str(t["ticker"])
                if tk in pos:  # already holding it: ignore the signal
                    continue
                if i <= blocked_until.get(tk, -1):  # re-entry cooldown after a stop-loss
                    continue
                alloc = min(prev_eq / cap, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                ledger.append({"ticker": tk, "entry_date": d, "exit_date": t["exit_date"], "alloc": alloc,
                               "ret": float(t["ret"]), "reason": t["reason"]})
                pos[tk] = {"alloc": alloc, "entry_px": float(t["entry_px"]),
                           "exit_date": t["exit_date"], "ret": float(t["ret"]), "reason": t["reason"]}
                taken += 1
                taken_rets.append(float(t["ret"]))
        max_held = max(max_held, len(pos))
        # exits during today
        for tk in [k for k, p in pos.items() if p["exit_date"] <= d]:
            p = pos.pop(tk)
            cash += p["alloc"] * (1 + p["ret"])
            if cooldown and p["reason"] == "stop":
                blocked_until[tk] = day_no.get(p["exit_date"], i) + cooldown
        # mark to market at close
        row = closes.loc[d]
        val = 0.0
        for tk, p in pos.items():
            c = row.get(tk)
            val += p["alloc"] * (c / p["entry_px"] if pd.notna(c) else 1.0)
        prev_eq = cash + val
        equity[i] = prev_eq
        invested_days += bool(pos)
        if pos:
            exposure.append(val / prev_eq)
    eq = pd.Series(equity, index=cal, name="equity")
    eq.attrs["ledger"] = pd.DataFrame(ledger)
    return eq, {"trades_taken": taken, "trades_available": len(tr),
                "taken_mean_ret": float(np.mean(taken_rets)) if taken_rets else np.nan,
                "all_mean_ret": float(tr["ret"].mean()),
                "invested_share": invested_days / len(cal), "max_held": max_held,
                "exposure": float(np.mean(exposure)) if exposure else 0.0}


def stats(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "mdd": (eq / eq.cummax() - 1).min(),
    }


def index_curve(cfg: dict, cal: pd.DatetimeIndex, capital: float, market: str = "KOSPI") -> pd.Series:
    """Buy and hold the index, same capital."""
    c = load_index(cfg, market)["close"].reindex(cal).ffill()
    return c / c.iloc[0] * capital
