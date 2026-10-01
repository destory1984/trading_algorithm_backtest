"""SPEC 3 / 4.3: trades and daily curves for every ticker and variant, and the 1/n-sleeve basket.

Rule: signal when close / MA25 <= threshold (adjusted close) and the 20-day average dollar volume is at least the
floor; buy at the next open (signal dropped if that day has no volume). Sell at the next open after a close at or
above MA25, or at the close of holding day 20 (entry day = day 1), whichever comes first; exits on a no-volume day
move to the next traded day. Signals are ignored while the sleeve holds; after a close exit the next signal day is
the following day, after an open exit the same day's close can signal again.

Usage: python -m src.run   -> results/trades.parquet, daily.parquet, sleeves.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C

OPEN, CLOSE, FORCED = "ma", "max", "forced"


def find_trades(fr: pd.DataFrame, thr: float, cfg: dict, start: str, min_dv: float) -> list[tuple[int, int, int, str]]:
    """(signal index, entry index, exit index, kind) per trade."""
    n = len(fr)
    c, ma, tr = fr["close"].to_numpy(), fr["ma"].to_numpy(), fr["tradable"].to_numpy()
    sig = ((fr["disp"] <= thr) & (fr["dv"] >= min_dv) & fr["ma"].notna()).to_numpy()
    hold = cfg["rule"]["max_hold"]
    first = int(fr.index.searchsorted(pd.Timestamp(start)))
    out, i = [], max(first - 1, 0)
    while i < n - 1:
        if not sig[i] or not tr[i + 1]:
            i += 1
            continue
        e, x, kind = i + 1, None, FORCED
        for j in range(e, n):
            if j - e + 1 >= hold:
                x = next((k for k in range(j, n) if tr[k]), None)
                kind = CLOSE
                break
            if c[j] >= ma[j]:
                x = next((k for k in range(j + 1, n) if tr[k]), None)
                kind = OPEN
                break
        if x is None:
            x, kind = n - 1, FORCED
        out.append((i, e, x, kind))
        i = x if kind == OPEN else x + 1
    return out


def book(fr: pd.DataFrame, spans: list[tuple[int, int, int, str]], cost: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Trade rows, the sleeve's daily returns and an in-position mask from (signal, entry, exit, kind) spans."""
    o, c = fr["open"].to_numpy(), fr["close"].to_numpy()
    r, held = np.zeros(len(fr)), np.zeros(len(fr), bool)
    rows = []
    for s, e, x, kind in spans:
        buy = o[e] * (1 + cost)
        sell = (o[x] if kind == OPEN else c[x]) * (1 - cost)
        if x == e:                                   # forced on the entry day (data ends)
            r[e] = sell / buy - 1
        else:
            r[e] = c[e] / buy - 1
            r[e + 1:x] = c[e + 1:x] / c[e:x - 1] - 1
            r[x] = sell / c[x - 1] - 1
        held[e:x + 1] = True
        rows.append({"signal_day": fr.index[s], "entry_day": fr.index[e], "exit_day": fr.index[x],
                     "entry": o[e], "exit": o[x] if kind == OPEN else c[x], "kind": kind,
                     "hold_days": x - e + 1, "disp": fr["disp"].iloc[s], "ret": sell / buy - 1})
    return pd.DataFrame(rows), r, held


def calendar(cfg: dict, tickers: list[str], start: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex([])
    for t in tickers:
        idx = idx.union(C.raw(cfg, t).index)
    return idx[idx >= pd.Timestamp(start)]


def basket(sleeves: pd.DataFrame) -> pd.Series:
    """Daily return of sleeves run separately with equal starting money (no rebalancing)."""
    eq = (1 + sleeves).cumprod().mean(axis=1)
    return (eq / eq.shift(1).fillna(1.0) - 1).rename("basket")


def hold_sleeve(fr: pd.DataFrame, cal: pd.DatetimeIndex, ma: int) -> pd.Series:
    """Buy at the close of the first calendar day on which MA25 exists, then hold. Cash before that."""
    c = fr["close"].where(fr["ma"].notna()).reindex(cal).ffill()
    return c.pct_change(fill_method=None).fillna(0.0)


def run_variant(cfg: dict, thr: float, cost: float | None = None, cash: float | None = None,
                min_dv: float | None = None, start: str | None = None, frames: dict | None = None):
    cost = cfg["cost"] if cost is None else cost
    cash = cfg["cash"] if cash is None else cash
    min_dv = cfg["rule"]["min_dollar_volume"] if min_dv is None else min_dv
    start = cfg["data"]["start"] if start is None else start
    tk = C.tickers(cfg)
    frames = frames or {t: C.frame(cfg, t) for t in tk}
    cal = calendar(cfg, tk, start)
    daily_cash = (1 + cash) ** (1 / 252) - 1
    trades, sl, hd, held = [], {}, {}, {}
    for t in tk:
        fr = frames[t]
        tr, r, h = book(fr, find_trades(fr, thr, cfg, start, min_dv), cost)
        if len(tr):
            trades.append(tr.assign(ticker=t))
        r = r + np.where(h, 0.0, daily_cash)
        sl[t] = pd.Series(r, fr.index).reindex(cal).fillna(daily_cash)
        held[t] = pd.Series(h, fr.index).reindex(cal, fill_value=False)
        hd[t] = hold_sleeve(fr, cal, cfg["rule"]["ma"])
    sl, hd = pd.DataFrame(sl), pd.DataFrame(hd)
    return {"trades": pd.concat(trades, ignore_index=True), "sleeves": sl, "held": pd.DataFrame(held),
            "basket": basket(sl), "hold": basket(hd).rename("hold"), "hold_sleeves": hd}


def variants(cfg: dict) -> dict[str, dict]:
    r, s = cfg["rule"], cfg["sens"]
    v = {f"{round(th * 100)}": {"thr": th} for th in [r["main"], *r["neighbors"]]}
    m = r["main"]
    v["95|비용 2배"] = {"thr": m, "cost": cfg["pass"]["cost_x2"]}
    v["95|비용 0"] = {"thr": m, "cost": 0.0}
    v["95|현금 2%"] = {"thr": m, "cash": s["cash"]}
    v["95|하한 없음"] = {"thr": m, "min_dv": 0.0}
    v["95|2004 시작"] = {"thr": m, "start": s["late_start"]}
    return v


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    frames = {t: C.frame(cfg, t) for t in C.tickers(cfg)}
    trades, daily = [], {}
    for name, kw in variants(cfg).items():
        res = run_variant(cfg, frames=frames, **kw)
        trades.append(res["trades"].assign(variant=name))
        daily[name] = res["basket"]
        daily[f"{name}|hold"] = res["hold"]
        if name == "95":
            res["sleeves"].to_parquet(out / "sleeves.parquet")
            res["hold_sleeves"].to_parquet(out / "hold_sleeves.parquet")
            res["held"].to_parquet(out / "held.parquet")
        print(f"{name:14s} trades {len(res['trades']):5d}  mean {res['trades']['ret'].mean():+.4f}  "
              f"cagr {C.cagr(res['basket'].dropna()):+.4f}  mdd {C.mdd(res['basket'].dropna()):+.4f}  "
              f"hold cagr {C.cagr(res['hold'].dropna()):+.4f} mdd {C.mdd(res['hold'].dropna()):+.4f}")
    pd.concat(trades, ignore_index=True).to_parquet(out / "trades.parquet")
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")


if __name__ == "__main__":
    main()
