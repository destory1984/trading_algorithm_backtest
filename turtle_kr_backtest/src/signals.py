"""SPEC 4.1: per-ticker arrays and the entry candidates of every variant.

For each ticker the krxbt frame (data-cleaning rules, ma_window 55) is laid on the sealed calendar from its first to
its last traded day. Indicators are computed on traded (valid) rows only and put back on the calendar rows:
  N      Wilder true-range average: first value = mean of the first n_window TRs, then (19 N + TR) / 20
  n_prev N as of the previous traded day (what is known before today's open; used for sizing and stops)
  hh_k   highest high of the previous k traded days (entry levels), ll_k lowest low (exit levels)
Candidates (one row per ticker-day): valid, eligible, not locked (high == low), on or after data.curve_start, and the
high crosses hh_entry. S1 (filter) drops a 20-day breakout when the previous hypothetical S1 trade was a winner,
unless the high also crosses hh_55 (failsafe; the level is then hh_55). New highs made while a hypothetical trade is
open belong to the breakout that opened it and take its decision. The hypothetical trade: one unit bought at
max(open, level) on every breakout while none is open, sold at the 2N stop or the exit channel, whichever is hit first
(fill min(open, level)); a winner if it sold above its buy price.

Outputs in results/panel/: concatenated arrays (*.npy) with per-ticker offsets, candidates_<variant>.parquet,
skipped_<variant>.parquet (S1 signals dropped by the filter, with the hypothetical trade that caused it),
eligible_days.npy (day, ticker) pairs for the random baseline, meta.parquet.
Usage: python -m src.signals
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd
from krxbt.frame import index_features, ticker_frame

from src import common as C

ARR_F = ("open", "high", "low", "close", "n_prev", "ll10", "ll20", "avg_value")
ARR_B = ("valid", "eligible", "locked", "data_break")


@numba.njit(cache=True)
def wilder(tr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(tr), np.nan)
    if len(tr) < n + 1:
        return out
    # tr[0] has no previous close; the first N uses tr[1 .. n]
    out[n] = tr[1: n + 1].mean()
    for i in range(n + 1, len(tr)):
        out[i] = ((n - 1) * out[i - 1] + tr[i]) / n
    return out


@numba.njit(cache=True)
def hypothetical(o, h, l, n_prev, hh, llx, stop_n):
    """S1 filter on traded rows. skip[i] is meaningful on breakout rows; prev_* describe the hypothetical trade that
    decided skip[i] (-1 when there was none)."""
    m = len(o)
    skip = np.zeros(m, np.bool_)
    prev_in = np.full(m, -1, np.int64)
    prev_out = np.full(m, -1, np.int64)
    prev_buy = np.full(m, np.nan)
    prev_sell = np.full(m, np.nan)
    in_trade = False
    last_win = False
    cur_skip = False  # the decision of the breakout that opened the hypothetical trade in progress
    li, lo_, lb, ls = -1, -1, np.nan, np.nan
    pi_, po_, pb_, ps_ = -1, -1, np.nan, np.nan
    ent_i, ent_px, stop = -1, 0.0, 0.0
    for i in range(m):
        if in_trade:
            s_hit = l[i] <= stop
            c_hit = np.isfinite(llx[i]) and l[i] < llx[i]
            if s_hit or c_hit:
                lvl = -np.inf
                if s_hit:
                    lvl = stop
                if c_hit and llx[i] > lvl:
                    lvl = llx[i]
                px = min(o[i], lvl)
                last_win = px > ent_px
                li, lo_, lb, ls = ent_i, i, ent_px, px
                in_trade = False
            elif np.isfinite(hh[i]) and h[i] > hh[i]:
                # a new high while the hypothetical trade is open belongs to the same signal
                skip[i] = cur_skip
                prev_in[i], prev_out[i], prev_buy[i], prev_sell[i] = pi_, po_, pb_, ps_
            continue
        if np.isfinite(hh[i]) and np.isfinite(n_prev[i]) and h[i] > hh[i]:
            skip[i] = last_win
            cur_skip = last_win
            prev_in[i], prev_out[i], prev_buy[i], prev_sell[i] = li, lo_, lb, ls
            pi_, po_, pb_, ps_ = li, lo_, lb, ls
            ent_i, ent_px = i, max(o[i], hh[i])
            stop = ent_px - stop_n * n_prev[i]
            in_trade = True
            s_hit = l[i] <= stop
            c_hit = np.isfinite(llx[i]) and l[i] < llx[i]
            if s_hit or c_hit:  # same-day exit (loss by construction)
                lvl = stop if s_hit else -np.inf
                if c_hit and llx[i] > lvl:
                    lvl = llx[i]
                last_win = False
                li, lo_, lb, ls = ent_i, i, ent_px, lvl
                in_trade = False
    return skip, prev_in, prev_out, prev_buy, prev_sell


def prior_max(x: pd.Series, k: int) -> np.ndarray:
    return x.shift(1).rolling(k, min_periods=k).max().to_numpy()


def prior_min(x: pd.Series, k: int) -> np.ndarray:
    return x.shift(1).rolling(k, min_periods=k).min().to_numpy()


def build(cfg: dict, load=None) -> dict:
    """All arrays and candidates. `load` replaces krxbt's price loader (look-ahead check)."""
    import krxbt.frame as KF
    old = KF.load_prices
    if load is not None:
        KF.load_prices = load
    try:
        return _build(cfg)
    finally:
        KF.load_prices = old


def _build(cfg: dict) -> dict:
    cal = C.calendar(cfg)
    idx = index_features(cfg)
    uni = C.tickers(cfg)
    t_cfg = cfg["turtle"]
    cs = int(cal.searchsorted(pd.Timestamp(cfg["data"]["curve_start"])))
    entries = sorted({n for _, n, _ in C.variants(cfg)} | {t_cfg["failsafe_breakout"]})
    exits = sorted({s["exit"] for s in cfg["rules"].values()})
    cols_f = {k: [] for k in ARR_F}
    cols_b = {k: [] for k in ARR_B}
    meta, cands, skipped, elig = [], {C.vkey(r, n): [] for r, n, _ in C.variants(cfg)}, {}, []
    pos = 0
    for tid, (tk, row) in enumerate(uni.iterrows()):
        f = ticker_frame(cfg, tk, row["market"], cal, idx, bool(row["delisted"]))
        if f is None or not len(f):
            meta.append((tk, row["market"], row["name"], bool(row["delisted"]), -1, 0, pos))
            continue
        f = f[f.index <= C.end(cfg)]
        off = int(cal.get_loc(f.index[0]))
        n_rows = len(f)
        v = f[f["valid"]]
        vpos = f.index.get_indexer(v.index)          # traded rows inside the frame
        pc = v["close"].shift(1)
        tr = np.maximum.reduce([(v["high"] - v["low"]).to_numpy(), (v["high"] - pc).abs().to_numpy(),
                                (pc - v["low"]).abs().to_numpy()])
        tr[0] = np.nan
        nn = wilder(np.nan_to_num(tr, nan=0.0), t_cfg["n_window"])
        n_prev = np.concatenate([[np.nan], nn[:-1]])
        hh = {k: prior_max(v["high"], k) for k in entries}
        ll = {k: prior_min(v["low"], k) for k in exits}

        def on_cal(x, fill=np.nan):
            out = np.full(n_rows, fill, dtype=np.asarray(x).dtype if fill is not np.nan else np.float64)
            out[vpos] = x
            return out

        arr = {"open": f["open"].to_numpy(np.float64), "high": f["high"].to_numpy(np.float64),
               "low": f["low"].to_numpy(np.float64), "close": f["close"].to_numpy(np.float64),
               "n_prev": on_cal(n_prev), "ll10": on_cal(ll.get(10, np.full(len(v), np.nan))),
               "ll20": on_cal(ll.get(20, np.full(len(v), np.nan))), "avg_value": f["avg_value"].to_numpy(np.float64)}
        valid = f["valid"].to_numpy(bool)
        locked = valid & (arr["high"] == arr["low"])
        eligible = f["eligible"].to_numpy(bool) & valid
        brk = f["data_break"].to_numpy(bool)
        for k in ARR_F:
            cols_f[k].append(arr[k])
        for k, x in (("valid", valid), ("eligible", eligible), ("locked", locked), ("data_break", brk)):
            cols_b[k].append(x)
        meta.append((tk, row["market"], row["name"], bool(row["delisted"]), off, n_rows, pos))
        pos += n_rows

        cal_i = off + vpos
        ok = eligible[vpos] & ~locked[vpos] & (cal_i >= cs) & np.isfinite(n_prev)
        base = np.flatnonzero(ok & np.isfinite(arr["n_prev"][vpos]))
        elig.append(np.stack([cal_i[base], np.full(len(base), tid)], axis=1))
        vo, vh, vl = v["open"].to_numpy(np.float64), v["high"].to_numpy(np.float64), v["low"].to_numpy(np.float64)
        av = v["avg_value"].to_numpy(np.float64)
        for r, n, _ in C.variants(cfg):
            spec = cfg["rules"][r]
            brk_up = ok & np.isfinite(hh[n]) & (vh > hh[n])
            level = hh[n].copy()
            if spec["filter"]:
                fs = t_cfg["failsafe_breakout"]
                sk, p_in, p_out, p_buy, p_sell = hypothetical(vo, vh, vl, n_prev, hh[n], ll[spec["exit"]],
                                                              t_cfg["stop_n"])
                fail = np.isfinite(hh[fs]) & (vh > hh[fs])
                dropped = brk_up & sk & ~fail
                use55 = brk_up & sk & fail
                level = np.where(use55, hh[fs], level)
                for i in np.flatnonzero(dropped):
                    skipped.setdefault(C.vkey(r, n), []).append(
                        (cal_i[i], tid, hh[n][i], cal_i[p_in[i]] if p_in[i] >= 0 else -1,
                         cal_i[p_out[i]] if p_out[i] >= 0 else -1, p_buy[i], p_sell[i]))
                brk_up = brk_up & ~dropped
            for i in np.flatnonzero(brk_up):
                cands[C.vkey(r, n)].append((cal_i[i], tid, level[i], av[i], n_prev[i]))
    out = {"cal": cal, "arrays": {k: np.concatenate(cols_f[k]) for k in ARR_F} |
                                 {k: np.concatenate(cols_b[k]) for k in ARR_B},
           "meta": pd.DataFrame(meta, columns=["ticker", "market", "name", "delisted", "off", "n", "pos"]),
           "cands": {k: pd.DataFrame(v, columns=["day", "tid", "level", "avg_value", "n_prev"])
                     .sort_values(["day", "avg_value"], ascending=[True, False], kind="stable").reset_index(drop=True)
                     for k, v in cands.items()},
           "skipped": {k: pd.DataFrame(v, columns=["day", "tid", "level", "prev_in", "prev_out", "prev_buy", "prev_sell"])
                       for k, v in skipped.items()},
           "eligible_days": np.concatenate(elig) if elig else np.zeros((0, 2), int)}
    return out


def save(b: dict) -> None:
    d = C.results_dir() / "panel"
    d.mkdir(parents=True, exist_ok=True)
    for k, a in b["arrays"].items():
        np.save(d / f"{k}.npy", a)
    b["meta"].to_parquet(d / "meta.parquet")
    pd.Series(b["cal"]).to_frame("date").to_parquet(d / "calendar.parquet")
    for k, df in b["cands"].items():
        df.to_parquet(d / f"candidates_{k}.parquet")
    for k, df in b["skipped"].items():
        df.to_parquet(d / f"skipped_{k}.parquet")
    e = b["eligible_days"]
    np.save(d / "eligible_days.npy", e[np.lexsort((e[:, 1], e[:, 0]))])


def load(mmap: bool = True) -> dict:
    d = C.results_dir() / "panel"
    arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r" if mmap else None) for k in ARR_F + ARR_B}
    cal = pd.DatetimeIndex(pd.read_parquet(d / "calendar.parquet")["date"])
    cands = {p.stem.removeprefix("candidates_"): pd.read_parquet(p) for p in d.glob("candidates_*.parquet")}
    skipped = {p.stem.removeprefix("skipped_"): pd.read_parquet(p) for p in d.glob("skipped_*.parquet")}
    return {"cal": cal, "arrays": arrays, "meta": pd.read_parquet(d / "meta.parquet"), "cands": cands,
            "skipped": skipped, "eligible_days": np.load(d / "eligible_days.npy")}


def main() -> None:
    cfg = C.load_config()
    b = build(cfg)
    save(b)
    m = b["meta"]
    print(f"{(m['n'] > 0).sum()} tickers with frames of {len(m)}; {len(b['arrays']['close'])} rows; "
          f"calendar {b['cal'][0].date()} -> {b['cal'][-1].date()}")
    for k, df in b["cands"].items():
        print(k, len(df), "candidates;", len(b["skipped"].get(k, [])), "skipped by the S1 filter")


if __name__ == "__main__":
    main()
