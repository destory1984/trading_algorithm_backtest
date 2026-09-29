"""SPEC 4 (US): U1 infinite buying, U2 value rebalancing, U3 dual momentum, and the plain hold, on adjusted prices.
Costs: every buy x (1 + fee) x (1 + slippage), every sale x (1 - fee) x (1 - slippage). Shares are fractional. Cash
earns 0.

U1 (baseline): a cycle starts at a close with 1 unit = equity / splits bought at that close. On each later day,
orders placed before the open with yesterday's average price avg: sell everything at avg x (1 + target) if the high
reaches it (fill max(open, limit)); the cycle then ends, the day's LOC buys are cancelled and a new cycle starts at the
next day's close. Otherwise half a unit is bought at the close if close <= avg and half a unit if close <= avg x
(1 + target) (LOC), while the round count T = cumulative buys / unit is below splits. When T reaches splits, a quarter
of the shares is sold at that close and the cumulative buys (so T) shrink by a quarter (the average stays).
U2 (baseline): buy start_stock of the capital at the first close, the rest is the pool; every `period` trading days
the target V grows by (1 + growth)^(period / 252); if the stock value is below V x (1 - band) buy up to V from the pool,
above V x (1 + band) sell down to V, at that close.
U3: at each month-end close pick the asset (G: higher 12-month return of SPY / EFA, AGG if that return is below BIL's;
A: higher mean of 1 / 3 / 6-month returns of SPY / SCZ, the safe asset if both are negative); switch at the next open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def costs(cfg: dict, mult: float = 1.0) -> tuple[float, float]:
    u = cfg["us"]
    f, s = u["fee"] * mult, u["slippage"] * mult
    return (1 + f) * (1 + s), (1 - f) * (1 - s)


def hold(cfg, px: pd.DataFrame, mult=1.0) -> pd.Series:
    bk, _ = costs(cfg, mult)
    cap = cfg["us"]["capital"]
    sh = cap / (px["close"].iloc[0] * bk)
    return pd.Series(sh * px["close"].to_numpy(), index=px.index)


def infinite(cfg, px: pd.DataFrame, splits: int, target: float, mult=1.0) -> dict:
    bk, sk = costs(cfg, mult)
    o, h, c = px["open"].to_numpy(), px["high"].to_numpy(), px["close"].to_numpy()
    cash = float(cfg["us"]["capital"])
    shares = basis = cum = unit = 0.0
    in_cycle = False
    eq = np.empty(len(c))
    log, cycles, exhausts = [], [], []
    cyc_start, exhaust_open = None, None
    for t in range(len(c)):
        if not in_cycle:
            unit = cash / splits
            q = unit / (c[t] * bk)
            shares, basis, cum, cash = q, q * c[t], unit, cash - unit
            in_cycle, cyc_start = True, t
            log.append((t, "buy_start", c[t], q))
        else:
            avg = basis / shares
            lim = avg * (1 + target)
            if h[t] >= lim:
                fill = max(o[t], lim)
                cash += shares * fill * sk
                log.append((t, "sell_all", fill, shares))
                cycles.append((cyc_start, t))
                if exhaust_open is not None:
                    exhausts[-1] = (exhausts[-1][0], t)
                    exhaust_open = None
                shares = basis = cum = 0.0
                in_cycle = False
                eq[t] = cash
                continue
            for cond, tag in ((c[t] <= avg, "loc_avg"), (c[t] <= lim, "loc_target")):
                if cum / unit < splits - 1e-9 and cond:
                    amt = min(unit / 2, cash)
                    if amt > 0:
                        q = amt / (c[t] * bk)
                        shares, basis, cum, cash = shares + q, basis + q * c[t], cum + amt, cash - amt
                        log.append((t, tag, c[t], q))
            if cum / unit >= splits - 1e-9:
                q = shares / 4
                cash += q * c[t] * sk
                shares -= q
                basis *= 0.75
                cum *= 0.75
                log.append((t, "quarter_sell", c[t], q))
                if exhaust_open is None:
                    exhausts.append((t, None))
                    exhaust_open = t
        eq[t] = cash + shares * c[t]
    return {"equity": pd.Series(eq, index=px.index), "log": pd.DataFrame(log, columns=["t", "kind", "px", "shares"]),
            "cycles": cycles, "exhausts": exhausts}


def value_rebalance(cfg, px: pd.DataFrame, band: float, mult=1.0) -> dict:
    u = cfg["us"]["U2"]
    bk, sk = costs(cfg, mult)
    c = px["close"].to_numpy()
    cap = float(cfg["us"]["capital"])
    shares = cap * u["start_stock"] / (c[0] * bk)
    pool = cap * (1 - u["start_stock"])
    v = shares * c[0]
    g = (1 + u["growth"]) ** (u["period"] / 252)
    eq = np.empty(len(c))
    log = []
    for t in range(len(c)):
        if t > 0 and t % u["period"] == 0:
            v *= g
            e = shares * c[t]
            if e < v * (1 - band) and pool > 0:
                amt = min(v - e, pool)
                shares += amt / (c[t] * bk)
                pool -= amt
                log.append((t, "buy", c[t], amt))
            elif e > v * (1 + band):
                q = (e - v) / c[t]
                shares -= q
                pool += q * c[t] * sk
                log.append((t, "sell", c[t], q * c[t]))
        eq[t] = pool + shares * c[t]
    return {"equity": pd.Series(eq, index=px.index), "log": pd.DataFrame(log, columns=["t", "kind", "px", "amount"])}


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    cur, nxt = idx[:-1], idx[1:]
    return cur[(cur.month != nxt.month) | (cur.year != nxt.year)]


def momentum_picks(cfg, px: dict[str, pd.DataFrame], kind: str, lookback=None, safe=None) -> pd.Series:
    """Month-end decision date -> asset (NaN where the look-back is not full)."""
    cal = px["SPY"].index
    me = month_ends(cal)
    cl = pd.DataFrame({t: p["close"].reindex(cal) for t, p in px.items()}).loc[me]
    if kind == "G":
        n = lookback
        r = cl / cl.shift(n) - 1
        pick = np.where(r["SPY"] >= r["EFA"], "SPY", "EFA")
        best = np.where(pick == "SPY", r["SPY"], r["EFA"])
        out = np.where(best < r["BIL"], "AGG", pick)
        ok = r[["SPY", "EFA", "BIL", "AGG"]].notna().all(axis=1).to_numpy()
    else:
        lbs = cfg["us"]["U3"]["A"]["lookbacks"]
        sc = sum(cl / cl.shift(k) - 1 for k in lbs) / len(lbs)
        pick = np.where(sc["SPY"] >= sc["SCZ"], "SPY", "SCZ")
        both_neg = (sc["SPY"] < 0) & (sc["SCZ"] < 0)
        out = np.where(both_neg, safe, pick)
        ok = sc[["SPY", "SCZ"]].notna().all(axis=1).to_numpy() & cl[safe].notna().to_numpy()
    return pd.Series(np.where(ok, out, None), index=me).dropna()


def switcher(cfg, px: dict[str, pd.DataFrame], picks: pd.Series, mult=1.0) -> dict:
    """100% in the picked asset; a new pick is traded at the next open (sell the old, buy the new)."""
    bk, sk = costs(cfg, mult)
    cal = px["SPY"].index
    first = cal.searchsorted(picks.index[0]) + 1
    days = cal[first:]
    want = picks.reindex(cal).ffill().shift(1).reindex(days)   # decided at yesterday's (month-end) close
    cash, asset, sh = float(cfg["us"]["capital"]), None, 0.0
    eq = np.empty(len(days))
    log = []
    for k, d in enumerate(days):
        w = want.iloc[k]
        if w != asset:
            if asset is not None:
                cash += sh * px[asset].loc[d, "open"] * sk
            op = px[w].loc[d, "open"]
            sh = cash / (op * bk)
            cash = 0.0
            log.append((d, asset, w, op))
            asset = w
        eq[k] = cash + sh * px[asset].loc[d, "close"]
    return {"equity": pd.Series(eq, index=days), "log": pd.DataFrame(log, columns=["date", "from", "to", "open"])}
