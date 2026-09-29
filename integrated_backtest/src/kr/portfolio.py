"""SPEC 5 (Korea): the 5-slot account over precomputed trades.

One day t:
  1. positions whose exit is at today's open (reason LINE: exit line decided at yesterday's close, and every C exit)
     are sold first and free their slot
  2. entries of the day, highest priority first (A2: lowest RSI, B: highest value multiple, C: the 5 largest previous
     20-day values chosen before the open, bought only if the target was reached), skipping stocks held; size =
     20% of equity at the previous close (equal, judged) or floor(equity x 1% / (2 x ATR20)) capped at 20% (turtle,
     sensitivity); bought only if the cash covers it
  3. every other exit of the day (stop, max hold, data end, void at a data break, still held at the end) at its price
  4. equity = cash + shares x last close
Costs: buy x (1 + slippage) x (1 + fee), sale x (1 - slippage) x (1 - fee - that day's sell tax). Cash earns 0.
Random entry: on each day the rule opened k positions, k stocks drawn from the random table rows of that day (every
buyable open with the same exit rules), skipping stocks held.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import common as C
from src.kr.kernel import LINE


class Panel:
    def __init__(self, in_ram: bool = True):
        d = C.results_dir("kr/panel")
        mm = None if in_ram else "r"
        self.close = np.load(d / "close.npy", mmap_mode=mm)
        self.valid = np.load(d / "valid.npy", mmap_mode=mm)
        m = pd.read_parquet(d / "meta.parquet").set_index("tid")
        n_t = int(m.index.max()) + 1
        self.off = np.full(n_t, -1, np.int64)
        self.n = np.zeros(n_t, np.int64)
        self.pos = np.zeros(n_t, np.int64)
        self.off[m.index], self.n[m.index], self.pos[m.index] = m["off"], m["n"], m["pos"]
        self.ticker = pd.Series(m["ticker"])
        self.market = pd.Series(m["market"])
        self.cal = pd.DatetimeIndex(pd.read_parquet(d / "calendar.parquet")["date"])

    def close_at(self, tid: int, t: int) -> float | None:
        r = t - self.off[tid]
        if r < 0 or r >= self.n[tid]:
            return None
        i = self.pos[tid] + r
        return float(self.close[i]) if self.valid[i] else None


def c_orders(cand: pd.DataFrame, orders: int) -> pd.DataFrame:
    """C: the day's order list is the `orders` rows with the largest previous 20-day value; keep those that hit."""
    top = cand.sort_values(["entry_i", "prio"], ascending=[True, False], kind="stable").groupby("entry_i").head(orders)
    return top[top["hit"]]


def simulate(cfg: dict, P: Panel, cand: pd.DataFrame | None, slippage: float, sizing: str = "equal",
             random_k: np.ndarray | None = None, rand_table: pd.DataFrame | None = None, seed: int = 0) -> dict:
    kr = cfg["kr"]
    cs = cfg["costs"]
    buy_k = (1 + slippage) * (1 + cs["buy_fee"])
    tax = C.tax_table(cfg, P.cal)
    sell_k = (1 - slippage) * (1 - cs["sell_fee"] - tax)
    t0 = int(P.cal.searchsorted(pd.Timestamp(cfg["data"]["start"])))
    nd = len(P.cal)
    slots = kr["slots"]
    share = 1.0 / slots
    if cand is not None:
        src = cand.sort_values(["entry_i", "prio"], ascending=[True, False], kind="stable")
    else:
        src = rand_table.sort_values("entry_i", kind="stable")
        rng = np.random.default_rng(seed)
    ent = src["entry_i"].to_numpy()
    starts = np.searchsorted(ent, np.arange(nd + 1))
    cols = {k: src[k].to_numpy() for k in ("tid", "exit_i", "entry_px", "exit_px", "reason")}
    atr = src["atr"].to_numpy() if "atr" in src else np.full(len(src), np.nan)
    cash = float(kr["capital"])
    eq_prev = cash
    held: dict[int, dict] = {}
    eq = np.empty(nd - t0)
    inv = np.empty(nd - t0)
    trades, n_entries = [], np.zeros(nd, np.int64)

    def sell(tid, t):
        nonlocal cash
        p = held.pop(tid)
        proceeds = p["shares"] * p["exit_px"] * sell_k[t]
        cash += proceeds
        trades.append({"tid": tid, "entry_i": p["entry_i"], "exit_i": t, "reason": p["reason"], "shares": p["shares"],
                       "entry_px": p["entry_px"], "exit_px": p["exit_px"], "cost": p["cost"], "proceeds": proceeds})

    for k, t in enumerate(range(t0, nd)):
        for tid in [x for x, p in held.items() if p["exit_i"] == t and p["reason"] == LINE]:
            sell(tid, t)
        lo, hi = starts[t], starts[t + 1]
        if lo < hi:
            if cand is not None:
                rows = range(lo, hi)
            else:
                want = int(random_k[t]) if random_k is not None else 0
                pool = [j for j in range(lo, hi) if int(cols["tid"][j]) not in held]
                rows = rng.choice(pool, size=min(want, len(pool)), replace=False) if want and pool else []
            for j in rows:
                if len(held) >= slots:
                    break
                tid = int(cols["tid"][j])
                if tid in held:
                    continue
                px = float(cols["entry_px"][j])
                budget = eq_prev * share
                if sizing == "turtle" and np.isfinite(atr[j]) and atr[j] > 0:
                    sh = min(math.floor(eq_prev * kr["turtle_risk"] / (2 * atr[j])), math.floor(budget / (px * buy_k)))
                else:
                    sh = math.floor(budget / (px * buy_k))
                cost = sh * px * buy_k
                if sh <= 0 or cost > cash:
                    continue
                cash -= cost
                held[tid] = {"entry_i": t, "exit_i": int(cols["exit_i"][j]), "exit_px": float(cols["exit_px"][j]),
                             "reason": int(cols["reason"][j]), "shares": sh, "entry_px": px, "cost": cost, "last": px}
                n_entries[t] += 1
        for tid in [x for x, p in held.items() if p["exit_i"] <= t]:
            sell(tid, t)
        value = 0.0
        for tid, p in held.items():
            cl = P.close_at(tid, t)
            if cl is not None:
                p["last"] = cl
            value += p["shares"] * p["last"]
        eq[k] = cash + value
        inv[k] = value / eq[k]
        eq_prev = eq[k]
        assert cash >= -1e-6
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["ret"] = tr["proceeds"] / tr["cost"] - 1
        tr["pnl"] = tr["proceeds"] - tr["cost"]
    dates = P.cal[t0:]
    return {"equity": pd.Series(eq, index=dates), "invested": pd.Series(inv, index=dates), "trades": tr,
            "entries": n_entries}
