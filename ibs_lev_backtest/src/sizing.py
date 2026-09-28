"""One ticker's trades as a daily account: entry weight, idle-cash interest, capital-gains tax.

The engine gives the trades (dates, prices, return after costs). Per day i, in order:
  1. idle cash grows by factors[i] (data.cash_factors: ^IRX / 252), before the open.
     Negative cash is multiplied by the same factor, so a shortfall is borrowed at the
     ^IRX rate (the debt grows by the T-bill rate) until an exit repays it.
  2. tax payment day other than the last day (tax.tax_pay_days: first trading day of May
     of the next year): cash -= tax on that tax year's realized gains (tax.tax_due_usd),
     before any entry, so an entry that day is sized from post-tax cash. Until then the
     money stays in the account (in cash earning interest, or invested).
  3. entry on entry_date: alloc = weight x cash (flat, so cash = equity); the rest stays
     in cash and keeps earning interest. The weight does not change while holding.
  4. exit on exit_date: cash += alloc x (1 + ret); the gain alloc x ret is booked in
     the exit year
  5. last day of the data: the tax of every year paid that day (the final year, whose May
     is outside the sample) is taken after that day's exits, so its realized total is complete.
  6. mark to market at the close: alloc x close / entry_px
Cash goes negative only when the tax falls due while a position opened earlier is still
open (the tax is taken from equity); min_cash reports it.
With weight 1, factors 1 and no tax this is krxbt.portfolio.run_portfolio(slots=1).
Invested share = days ending with a position open (the original's "time in market").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tax import tax_due_usd, tax_pay_days


def realized_vol(f: pd.DataFrame, window: int) -> pd.Series:
    """Annualized std (ddof 1) of the last `window` close-to-close returns up to each day's close."""
    c = f.loc[f["valid"], "close"]
    return c.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252)


def weights(tr: pd.DataFrame, vol: pd.Series, target: float | None) -> np.ndarray:
    """min(1, target / vol on the signal day); vol not ready or 0 -> 1 (the original); None -> all 1."""
    if target is None:
        return np.ones(len(tr))
    v = vol.reindex(pd.DatetimeIndex(tr["signal_date"])).to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.minimum(1.0, target / v)
    return np.where(np.isfinite(v) & (v > 0), w, 1.0)


def account(cal: pd.DatetimeIndex, tr: pd.DataFrame, close: pd.Series, cap: float, weight: np.ndarray,
            factors: np.ndarray, tax: dict | None = None) -> tuple[pd.Series, dict]:
    tr = tr.reset_index(drop=True)
    assert len(weight) == len(tr), "one weight per trade"
    assert len(factors) == len(cal), "one cash factor per calendar day"
    assert tr["entry_date"].is_unique, "entry dates must be unique"
    by_entry = {d: i for i, d in enumerate(tr["entry_date"])}
    px = close.reindex(cal).ffill().to_numpy(np.float64)
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
    min_cash = cash
    min_cash_date = cal[0] if len(cal) else None
    min_cash_equity = cash
    last = cal[-1] if len(cal) else None

    def pay(d: pd.Timestamp) -> None:
        nonlocal cash
        for y in pay_on.get(d, []):  # the tax year y is closed: its realized total is final
            due = tax_due_usd(realized.get(y, 0.0), tax)
            cash -= due
            taxes[y] = due
            paid_on[y] = d

    for i, d in enumerate(cal):
        cash *= factors[i]
        if d != last:
            pay(d)  # before the entry: an entry today is sized from post-tax cash
        if pos is None and d in by_entry:
            j = by_entry[d]
            alloc = float(weight[j]) * cash
            cash -= alloc
            pos = {"alloc": alloc, "px": float(tr.at[j, "entry_px"]), "exit": tr.at[j, "exit_date"],
                   "ret": float(tr.at[j, "ret"])}
            taken += 1
        if pos is not None and pos["exit"] <= d:
            gain = pos["alloc"] * pos["ret"]
            cash += pos["alloc"] + gain
            realized[d.year] = realized.get(d.year, 0.0) + gain
            pos = None
        if d == last:
            pay(d)  # after the last day's exits: the final year's realized total is complete
        eq[i] = cash + (0.0 if pos is None else pos["alloc"] * px[i] / pos["px"])
        if cash < min_cash:
            min_cash, min_cash_date, min_cash_equity = cash, d, eq[i]
        held += pos is not None
    return pd.Series(eq, cal, name="equity"), {
        "trades_taken": taken, "trades_available": len(tr), "invested_share": held / len(cal),
        "realized": realized, "taxes": taxes, "paid_on": paid_on, "min_cash": min_cash,
        "min_cash_date": min_cash_date, "min_cash_equity": min_cash_equity}
