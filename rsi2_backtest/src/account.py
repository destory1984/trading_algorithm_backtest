"""Trades as a daily account with Korean capital-gains tax (config `tax`), for one or several tickers.

Copied from ibs_lev_backtest/src/sizing.py (account) with one change: an open position is marked with its own ticker's
close from a closes table (walk-forward trades switch tickers between years). Entry weight and cash factors stay in the
signature; this folder passes weight 1 and factor 1 (no sizing, no interest on idle cash).
Per day, in order: cash x factor; a tax payment day other than the last day: pay before any entry (an entry that day
is sized from post-tax cash); entry: alloc = weight x cash; exit: cash += alloc x (1 + ret), the gain booked in the exit
year; last day: pay after that day's exits (the final year's total is complete); mark to market at the close. Cash goes
negative when the tax falls due while a position is open (the tax is taken from equity); min_cash reports it.
With weight 1, factor 1 and no tax this is krxbt.portfolio.run_portfolio(slots=1) (checks.account_engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tax import tax_due_usd, tax_pay_days


def account(cal: pd.DatetimeIndex, tr: pd.DataFrame, closes: pd.DataFrame, cap: float, weight: np.ndarray,
            factors: np.ndarray, tax: dict | None = None) -> tuple[pd.Series, dict]:
    tr = tr.reset_index(drop=True)
    assert len(weight) == len(tr), "one weight per trade"
    assert len(factors) == len(cal), "one cash factor per calendar day"
    assert tr["entry_date"].is_unique, "entry dates must be unique"
    by_entry = {d: i for i, d in enumerate(tr["entry_date"])}
    px = closes.reindex(cal).ffill()
    col = {t: j for j, t in enumerate(px.columns)}
    pxv = px.to_numpy(np.float64)
    pay_on: dict[pd.Timestamp, list[int]] = {}
    if tax is not None:
        for y, day in tax_pay_days(cal, tax["pay_month"]).items():
            pay_on.setdefault(day, []).append(y)
    cash, pos = float(cap), None
    eq = np.empty(len(cal))
    realized: dict[int, float] = {}
    taxes: dict[int, float] = {}
    paid_on: dict[int, pd.Timestamp] = {}
    taken = held = 0
    min_cash, min_cash_date, min_cash_equity = cash, (cal[0] if len(cal) else None), cash
    last = cal[-1] if len(cal) else None

    def pay(d: pd.Timestamp) -> None:
        nonlocal cash
        for y in pay_on.get(d, []):
            due = tax_due_usd(realized.get(y, 0.0), tax)
            cash -= due
            taxes[y] = due
            paid_on[y] = d

    for i, d in enumerate(cal):
        cash *= factors[i]
        if d != last:
            pay(d)
        if pos is None and d in by_entry:
            j = by_entry[d]
            alloc = float(weight[j]) * cash
            cash -= alloc
            pos = {"alloc": alloc, "px": float(tr.at[j, "entry_px"]), "exit": tr.at[j, "exit_date"],
                   "ret": float(tr.at[j, "ret"]), "col": col[tr.at[j, "ticker"]]}
            taken += 1
        if pos is not None and pos["exit"] <= d:
            gain = pos["alloc"] * pos["ret"]
            cash += pos["alloc"] + gain
            realized[d.year] = realized.get(d.year, 0.0) + gain
            pos = None
        if d == last:
            pay(d)
        eq[i] = cash + (0.0 if pos is None else pos["alloc"] * pxv[i, pos["col"]] / pos["px"])
        if cash < min_cash:
            min_cash, min_cash_date, min_cash_equity = cash, d, eq[i]
        held += pos is not None
    return pd.Series(eq, cal, name="equity"), {
        "trades_taken": taken, "trades_available": len(tr), "invested_share": held / len(cal) if len(cal) else 0.0,
        "realized": realized, "taxes": taxes, "paid_on": paid_on, "min_cash": min_cash,
        "min_cash_date": min_cash_date, "min_cash_equity": min_cash_equity}
