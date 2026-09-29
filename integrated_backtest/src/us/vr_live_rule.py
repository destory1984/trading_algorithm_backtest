"""SPEC2 2: value rebalancing with next-open execution, shared by the gate (live_gate) and the forward log (live_vr).

Bar 0 is the first bar of `px`: start_stock of the capital buys TQQQ at bar 0's open; V = shares x that open.
Every `period` bars after that (bar 10, 20, ...), at the close: V grows by (1 + growth)^(period / 252), E = shares x
close; below V x (1 - band) an order to buy min(V - E, pool) dollars, above V x (1 + band) an order to sell E - V
dollars. The order fills at the next bar's open (a buy spends that dollar amount, a sale sells that dollar amount of
shares, capped at the shares held). An order decided on the last bar is returned but not filled.
Costs as us.fee / us.slippage (buy x (1 + fee)(1 + slippage), sale x (1 - fee)(1 - slippage)); pool earns 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.us.strategies import costs


def run(cfg: dict, px: pd.DataFrame, band: float, mult: float = 1.0) -> dict:
    u = cfg["us"]["U2"]
    bk, sk = costs(cfg, mult)
    o, c = px["open"].to_numpy(), px["close"].to_numpy()
    idx = px.index
    cap = float(cfg["us"]["capital"])
    amt0 = cap * u["start_stock"]
    shares = amt0 / (o[0] * bk)
    pool = cap - amt0
    v = shares * o[0]
    g = (1 + u["growth"]) ** (u["period"] / 252)
    eq = np.empty(len(c))
    orders, pending = [], None
    for t in range(len(c)):
        if pending is not None:
            side, amt, k = pending
            if side == "buy":
                shares += amt / (o[t] * bk)
                pool -= amt
            else:
                q = min(amt / o[t], shares)
                shares -= q
                pool += q * o[t] * sk
            orders[k]["filled"] = str(idx[t].date())
            orders[k]["fill_open"] = float(o[t])
            pending = None
        if t > 0 and t % u["period"] == 0:
            v *= g
            e = shares * c[t]
            side, amt = None, 0.0
            if e < v * (1 - band) and pool > 0:
                side, amt = "buy", min(v - e, pool)
            elif e > v * (1 + band):
                side, amt = "sell", e - v
            orders.append({"decision": str(idx[t].date()), "bar": t, "close": float(c[t]), "V": v, "E": e, "pool": pool,
                           "side": side or "", "amount": amt, "filled": "", "fill_open": np.nan})
            if side:
                pending = (side, amt, len(orders) - 1)
        eq[t] = pool + shares * c[t]
    return {"equity": pd.Series(eq, index=idx), "orders": pd.DataFrame(orders),
            "state": {"shares": shares, "pool": pool, "V": v}}


def hold(cfg: dict, px: pd.DataFrame) -> pd.Series:
    """TQQQ bought with the whole capital at bar 0's open."""
    bk, _ = costs(cfg)
    sh = float(cfg["us"]["capital"]) / (px["open"].iloc[0] * bk)
    return pd.Series(sh * px["close"].to_numpy(), index=px.index)
