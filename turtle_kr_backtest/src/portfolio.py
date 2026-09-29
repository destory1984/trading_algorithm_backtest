"""SPEC 3 / 4.2: the Turtle account on Korean stocks, long only.

One day, in this order (SPEC 3, all ties resolved against the account):
  1. Held stocks. Halted: nothing (valued at the last close). Data break today: sold at the previous close (void).
     Exit pending from a locked day: sold at the open. Locked (high == low): if the stop or the exit channel is
     touched, the sale moves to the next traded day's open. Otherwise, if the low touches the stop or goes under the
     exit channel, all units are sold at min(open, the higher of the touched levels). Otherwise units are added at
     max(open, last buy + N/2) while the high reaches that level (eligible and not locked days only, 4 per stock,
     12 in all, cash permitting); every add lifts the stop to (add price - 2N), and if the low then touches the new
     stop the stock is sold at it the same day.
  2. New entries (candidates of the day, largest 20-day trading value first; or random picks at the open for the
     random baseline): shares = floor(equity at the previous close x 1% / N), bought at max(open, level) if the cash
     covers it and fewer than 12 units are held. Not a stock sold today. If the low that day touches the stop or goes
     under the exit channel, it is sold the same day at the higher touched level. No adds on the entry day.
  3. A stock whose data ends today (delisted or halted to the end) is sold at today's close.
  4. Equity = cash + shares x last close.
N is the value known before the first entry's open and stays fixed for the position. Buy price x (1 + slippage) x
(1 + fee); sale x (1 - slippage) x (1 - fee - that day's sell tax). Cash earns nothing and is never negative.
Usage: python -m src.portfolio   -> results/equity.parquet, trades.parquet, units.parquet
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import common as C
from src import signals as SG


class Panel:
    """Array access by (ticker id, calendar day)."""

    def __init__(self, p: dict, in_ram: bool = True):
        a = p["arrays"]
        get = (lambda k: np.asarray(a[k])) if in_ram else (lambda k: a[k])
        self.o, self.h, self.l, self.c = get("open"), get("high"), get("low"), get("close")
        self.ll = {10: get("ll10"), 20: get("ll20")}
        self.valid, self.elig, self.locked, self.brk = get("valid"), get("eligible"), get("locked"), get("data_break")
        self.n_prev = get("n_prev")
        m = p["meta"]
        self.off, self.n, self.pos = m["off"].to_numpy(), m["n"].to_numpy(), m["pos"].to_numpy()
        self.ticker = m["ticker"].to_numpy()
        self.cal = p["cal"]
        self.last_day = self.off + self.n - 1

    def row(self, tid: int, t: int) -> int:
        return int(self.pos[tid] + t - self.off[tid])


def by_day(df: pd.DataFrame, n_days: int) -> tuple[np.ndarray, ...]:
    d = df["day"].to_numpy()
    start = np.searchsorted(d, np.arange(n_days + 1))
    return start, df["tid"].to_numpy(), df["level"].to_numpy(), df["n_prev"].to_numpy()


def simulate(cfg: dict, P: Panel, exit_days: int, cands: pd.DataFrame | None = None, slippage: float | None = None,
             random_k: np.ndarray | None = None, elig_days: np.ndarray | None = None, seed: int = 0) -> dict:
    tc = cfg["turtle"]
    cs_cfg = cfg["costs"]
    slip = cs_cfg["slippage"] if slippage is None else slippage
    buy_k = (1 + slip) * (1 + cs_cfg["buy_fee"])
    tax = C.tax_table(cfg, P.cal)
    sell_k = (1 - slip) * (1 - cs_cfg["sell_fee"] - tax)
    cal_n = len(P.cal)
    t0 = int(P.cal.searchsorted(pd.Timestamp(cfg["data"]["curve_start"])))
    ll = P.ll[exit_days]
    risk, stop_n, add_n = tc["risk_per_unit"], tc["stop_n"], tc["add_n"]
    max_u, max_all = tc["max_units_per_stock"], tc["max_units_total"]
    if cands is not None:
        c_start, c_tid, c_lvl, c_n = by_day(cands, cal_n)
    else:
        rng = np.random.default_rng(seed)
        e_day = elig_days[:, 0]
        e_start = np.searchsorted(e_day, np.arange(cal_n + 1))
        e_tid = elig_days[:, 1]
    cash = float(tc["capital"])
    held: dict[int, dict] = {}
    units_all = 0
    eq = np.empty(cal_n - t0)
    units = np.zeros(cal_n - t0, np.int64)
    invested = np.empty(cal_n - t0)
    trades, entries = [], np.zeros(cal_n, np.int64)
    eq_prev = cash

    def sell(tid: int, t: int, px: float, reason: str, when: int):
        nonlocal cash, units_all
        p = held.pop(tid)
        proceeds = p["shares"] * px * sell_k[when]
        cash += proceeds
        units_all -= p["units"]
        trades.append({"tid": tid, "ticker": P.ticker[tid], "entry_day": P.cal[p["entry_t"]], "exit_day": P.cal[t],
                       "reason": reason, "units": p["units"], "shares": p["shares"], "n": p["n"],
                       "cost": p["cost"], "proceeds": proceeds, "exit_px": px, "fills": p["fills"]})

    def buy(tid: int, t: int, px: float, n: float, first: bool) -> bool:
        nonlocal cash, units_all
        sh = math.floor(eq_prev * risk / n)
        cost = sh * px * buy_k
        if sh <= 0 or cost > cash:
            return False
        cash -= cost
        units_all += 1
        if first:
            held[tid] = {"entry_t": t, "n": n, "units": 0, "shares": 0, "cost": 0.0, "fills": [], "pending": None,
                         "last_close": px}
        p = held[tid]
        p["units"] += 1
        p["shares"] += sh
        p["cost"] += cost
        p["fills"].append((str(P.cal[t].date()), px, sh))
        p["last"] = px
        p["stop"] = px - stop_n * n
        p["next_add"] = px + add_n * n
        return True

    for k, t in enumerate(range(t0, cal_n)):
        sold_today = set()
        for tid in list(held):
            p = held[tid]
            r = P.row(tid, t)
            if t > P.last_day[tid] or not P.valid[r]:
                continue
            o, h, l, c = P.o[r], P.h[r], P.l[r], P.c[r]
            if P.brk[r]:
                sell(tid, t, p["last_close"], "break_void", t)
                sold_today.add(tid)
                continue
            if p["pending"] is not None:
                sell(tid, t, o, "pending_" + p["pending"], t)
                sold_today.add(tid)
                continue
            s_hit = l <= p["stop"]
            c_hit = np.isfinite(ll[r]) and l < ll[r]
            if P.locked[r]:
                if s_hit or c_hit:
                    p["pending"] = "stop" if s_hit and not (c_hit and ll[r] > p["stop"]) else "channel"
                p["last_close"] = c
                continue
            if s_hit or c_hit:
                use_c = c_hit and (not s_hit or ll[r] > p["stop"])
                lvl = ll[r] if use_c else p["stop"]
                sell(tid, t, min(o, lvl), "channel" if use_c else "stop", t)
                sold_today.add(tid)
                continue
            added = False
            if P.elig[r]:
                while p["units"] < max_u and units_all < max_all and h >= p["next_add"]:
                    if not buy(tid, t, max(o, p["next_add"]), p["n"], False):
                        break
                    added = True
            if added and l <= p["stop"]:
                sell(tid, t, p["stop"], "stop_after_add", t)
                sold_today.add(tid)
                continue
            p["last_close"] = c

        if cands is not None:
            lo_, hi_ = c_start[t], c_start[t + 1]
            picks = ((c_tid[j], c_lvl[j], c_n[j]) for j in range(lo_, hi_))
        else:
            want = int(random_k[t]) if random_k is not None else 0
            pool = e_tid[e_start[t]: e_start[t + 1]]
            pool = pool[[x not in held and x not in sold_today for x in pool]] if want and len(pool) else pool[:0]
            ch = rng.choice(pool, size=min(want, len(pool)), replace=False) if want and len(pool) else []
            picks = ((int(x), np.nan, P.n_prev[P.row(int(x), t)]) for x in ch)
        for tid, lvl, n in picks:
            if units_all >= max_all:
                break
            tid = int(tid)
            if tid in held or tid in sold_today:
                continue
            r = P.row(tid, t)
            px = P.o[r] if np.isnan(lvl) else max(P.o[r], lvl)
            if not buy(tid, t, px, float(n), True):
                continue
            entries[t] += 1
            p = held[tid]
            s_hit = P.l[r] <= p["stop"]
            c_hit = np.isfinite(ll[r]) and P.l[r] < ll[r]
            if s_hit or c_hit:
                use_c = c_hit and (not s_hit or ll[r] > p["stop"])
                sell(tid, t, ll[r] if use_c else p["stop"], "same_day_" + ("channel" if use_c else "stop"), t)
                sold_today.add(tid)
            else:
                p["last_close"] = P.c[r]

        for tid in list(held):
            if t == P.last_day[tid] and P.last_day[tid] < cal_n - 1:
                r = P.row(tid, t)
                sell(tid, t, P.c[r], "data_end", t)

        value = sum(p["shares"] * p["last_close"] for p in held.values())
        eq[k] = cash + value
        invested[k] = value / eq[k]
        units[k] = units_all
        eq_prev = eq[k]
        assert cash >= -1e-6, "negative cash"

    t_end = cal_n - 1
    for tid in list(held):
        p = held[tid]
        sell(tid, t_end, p["last_close"], "open_at_end", t_end)
    dates = P.cal[t0:]
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["ret"] = tr["proceeds"] / tr["cost"] - 1
        tr["pnl"] = tr["proceeds"] - tr["cost"]
    return {"equity": pd.Series(eq, index=dates), "units": pd.Series(units, index=dates),
            "invested": pd.Series(invested, index=dates), "trades": tr, "entries": entries}


def main() -> None:
    cfg = C.load_config()
    p = SG.load(mmap=False)
    P = Panel(p)
    out = C.results_dir()
    eqs, units, inv, trs, ent = {}, {}, {}, [], {}
    for r, n, main_ in C.variants(cfg):
        k = C.vkey(r, n)
        res = simulate(cfg, P, cfg["rules"][r]["exit"], p["cands"][k])
        eqs[k], units[k], inv[k] = res["equity"], res["units"], res["invested"]
        res["trades"].insert(0, "variant", k)
        trs.append(res["trades"])
        ent[k] = res["entries"]
        e = res["equity"]
        print(f"{k}: final {e.iloc[-1] / 1e8:.2f}x, CAGR {C.cagr(e.pct_change().fillna(e.iloc[0] / cfg['turtle']['capital'] - 1)):+.2%}, "
              f"trades {len(res['trades'])}, avg units {res['units'].mean():.1f}, invested {res['invested'].mean():.0%}")
    pd.DataFrame(eqs).to_parquet(out / "equity.parquet")
    pd.DataFrame(units).to_parquet(out / "units.parquet")
    pd.DataFrame(inv).to_parquet(out / "invested.parquet")
    t = pd.concat(trs, ignore_index=True)
    t["fills"] = t["fills"].map(lambda f: ";".join(f"{d}@{px:.4f}x{sh}" for d, px, sh in f))
    t.to_parquet(out / "trades.parquet")
    pd.DataFrame(ent, index=P.cal).to_parquet(out / "entries.parquet")


if __name__ == "__main__":
    main()
