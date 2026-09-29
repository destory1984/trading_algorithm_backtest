"""SPEC 4-6 (Korea): one pass over every ticker that produces everything the Korean part needs.

Pass 1 (cross-section): each ticker's 250-day return, 20-day average trading value and eligibility on the calendar
  -> RS  = percentile (0 -> 100) of the 250-day return among the tickers eligible that day
  -> top = rank of the previous day's 20-day average value among the tickers eligible that previous day <= top_n (C)
Pass 2 (per ticker, 6 processes): krxbt frame (ma_window 250), indicators on traded rows, then
  - grid trades in seq mode for every A2 / B / C grid combo (SPEC 6)            -> grid_trades.parquet
  - portfolio candidates in all mode for the judged combos and their neighbours -> cand_<strategy>.parquet
  - random-entry tables: the trade from every buyable open with the judged exit rules (A2, B) or next-open exit (C)
                                                                                 -> rand_<strategy>.parquet
  - close / valid arrays for valuing held positions                             -> panel/*.npy, meta.parquet
Indicators (traded rows): MA n; RSI(2) Wilder (ewm alpha 1/2); hh_n / ll_n = max high / min low of the previous n
rows; value multiple = close x volume / previous 20-row mean; ATR20 = mean true range of 20 rows; 250-row return;
52-week low / high = 250-row min low / max high; noise = 1 - |open - close| / (high - low), 20-row mean.
Usage: python -m src.kr.build
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from krxbt.data import load_calendar, universe_tickers
from krxbt.frame import index_features, ticker_frame

from src import common as C
from src.kr import kernel as K

OUT = "kr"


# ---------------------------------------------------------------- combos

def a2_grid(cfg):
    g = cfg["kr"]["A2"]["grid"]
    return [dict(rsi=r, stop=s, regime=rg) for r, s, rg in itertools.product(g["rsi"], g["stop"], g["regime"])]


def b_grid(cfg):
    g = cfg["kr"]["B"]["grid"]
    return [dict(n=n, mult=m, stop=s, exit=x, filter=f, regime=rg)
            for n, m, s, x, f, rg in itertools.product(g["n"], g["mult"], g["stop"], g["exit"], g["filter"], g["regime"])]


def c_grid(cfg):
    g = cfg["kr"]["C"]["grid"]
    return [dict(k=k, filter=f, regime=rg) for k, f, rg in itertools.product(g["k"], g["filter"], g["regime"])]


def judged(cfg) -> dict[str, list[tuple[str, dict]]]:
    """strategy -> [(label, combo)] : the judged combo first, then its neighbours (SPEC 1, 7.1-4)."""
    k = cfg["kr"]
    out = {}
    m = {x: v for x, v in k["A2"]["main"].items() if x != "market"}
    out["A2"] = [("main", m)] + [(f"nb_{list(nb)[0]}_{list(nb.values())[0]}", {**m, **nb}) for nb in k["A2"]["neighbors"]]
    for key, name in (("main_O", "BO"), ("main_M", "BM")):
        m = {x: v for x, v in k["B"][key].items() if x != "market"}
        out[name] = [("main", m)] + [(f"nb_{list(nb)[0]}_{list(nb.values())[0]}", {**m, **nb}) for nb in k["B"]["neighbors"]]
    m = {x: v for x, v in k["C"]["main"].items() if x != "market"}
    out["C"] = [("main", m)] + [(f"nb_{list(nb)[0]}_{list(nb.values())[0]}", {**m, **nb}) for nb in k["C"]["neighbors"]]
    return out


def combo_id(grid: list[dict], combo: dict) -> int:
    for i, g in enumerate(grid):
        if g == combo:
            return i
    raise KeyError(combo)


# ---------------------------------------------------------------- shared state

def calendar(cfg) -> pd.DatetimeIndex:
    cal = load_calendar(cfg)
    return cal[cal <= pd.Timestamp(cfg["data"]["end"])]


def limit_ratio(cfg, days: pd.DatetimeIndex) -> np.ndarray:
    k = cfg["kr"]
    return np.where(days < pd.Timestamp(k["price_limit_change"]), k["price_limit_pre"], k["price_limit"])


_G = {}


def _init():
    cfg = C.load_config()
    _G["cfg"] = cfg
    _G["cal"] = calendar(cfg)
    _G["idx"] = index_features(cfg)
    _G["uni"] = universe_tickers(cfg)
    d = C.results_dir(OUT)
    if (d / "rs.npy").exists():
        _G["rs"] = np.load(d / "rs.npy", mmap_mode="r")
        _G["top"] = np.load(d / "top.npy", mmap_mode="r")


def _frame(tid: int):
    cfg, cal, uni = _G["cfg"], _G["cal"], _G["uni"]
    tk = uni.index[tid]
    row = uni.iloc[tid]
    f = ticker_frame(cfg, tk, row["market"], cal, _G["idx"], bool(row["delisted"]))
    if f is None or not len(f):
        return None
    return f[f.index <= pd.Timestamp(cfg["data"]["end"])]


def _pass1(tid: int):
    f = _frame(tid)
    if f is None:
        return tid, None
    cal = _G["cal"]
    v = f[f["valid"]]
    r250 = (v["close"] / v["close"].shift(250) - 1).reindex(f.index)
    off = int(cal.get_loc(f.index[0]))
    return tid, (off, f["eligible"].to_numpy(bool), r250.to_numpy(np.float32), f["avg_value"].to_numpy(np.float32))


def pass1(cfg) -> None:
    cal = calendar(cfg)
    uni = universe_tickers(cfg)
    nd, nt = len(cal), len(uni)
    ret = np.full((nd, nt), np.nan, np.float32)
    av = np.full((nd, nt), np.nan, np.float32)
    el = np.zeros((nd, nt), bool)
    with ProcessPoolExecutor(cfg["kr"]["workers"], initializer=_init) as ex:
        for tid, res in ex.map(_pass1, range(nt), chunksize=16):
            if res is None:
                continue
            off, e, r, a = res
            ret[off: off + len(e), tid], av[off: off + len(e), tid], el[off: off + len(e), tid] = r, a, e
    r_ok = np.where(el, ret, np.nan)
    rs = pd.DataFrame(r_ok).rank(axis=1, pct=True).to_numpy(np.float32) * 100
    a_ok = pd.DataFrame(np.where(el, av, np.nan)).rank(axis=1, ascending=False, method="first").to_numpy()
    top = np.zeros((nd, nt), bool)
    top[1:] = a_ok[:-1] <= cfg["kr"]["C"]["top_n"]          # yesterday's rank decides today's universe
    d = C.results_dir(OUT)
    np.save(d / "rs.npy", rs)
    np.save(d / "top.npy", top)
    print(f"pass 1: {nt} tickers x {nd} days; eligible ticker-days {int(el.sum())}")


# ---------------------------------------------------------------- pass 2

def indicators(f: pd.DataFrame) -> dict[str, np.ndarray]:
    v = f[f["valid"]]
    o, h, l, c = v["open"], v["high"], v["low"], v["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (pc - l).abs()], axis=1).max(axis=1)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi = rsi.where(dn > 0, 100.0)
    value = c * v["volume"]
    rng = h - l
    noise = (1 - (o - c).abs() / rng.where(rng > 0))
    out = {"ma5": c.rolling(5).mean(), "ma20": c.rolling(20).mean(), "ma50": c.rolling(50).mean(),
           "ma150": c.rolling(150).mean(), "ma200": c.rolling(200).mean(), "rsi2": rsi,
           "hh20": h.shift(1).rolling(20).max(), "hh60": h.shift(1).rolling(60).max(),
           "ll10": l.shift(1).rolling(10).min(), "ll20": l.shift(1).rolling(20).min(),
           "mult": value / value.shift(1).rolling(20).mean(), "atr20": tr.rolling(20).mean(),
           "lo52": l.rolling(250).min(), "hi52": h.rolling(250).max(), "noise20": noise.rolling(20, min_periods=20).mean(),
           "rng": rng}
    out["ma200_20"] = out["ma200"].shift(20)
    return {k: s.reindex(f.index).to_numpy(np.float64) for k, s in out.items()}


def _trades_df(res, tid, extra: dict | None = None) -> pd.DataFrame:
    s, e, x, ep, xp, why = res
    d = {"tid": np.full(len(s), tid, np.int32), "sig_i": s, "entry_i": e, "exit_i": x, "entry_px": ep,
         "exit_px": xp, "reason": why.astype(np.int8)}
    if extra:
        d.update({k: v[s] for k, v in extra.items()})
    return pd.DataFrame(d)


def _pass2(tid: int):
    cfg, cal = _G["cfg"], _G["cal"]
    f = _frame(tid)
    if f is None:
        return tid, None
    kr = cfg["kr"]
    off = int(cal.get_loc(f.index[0]))
    k0 = int(cal.searchsorted(pd.Timestamp(cfg["data"]["keep_from"])))
    start = int(cal.searchsorted(pd.Timestamp(cfg["data"]["start"])))
    ind = indicators(f)
    n = len(f)
    o, h, l, c = (f[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    valid = f["valid"].to_numpy(bool)
    locked = valid & (h == l)
    brk = f["data_break"].to_numpy(bool)
    elig = f["eligible"].to_numpy(bool) & valid
    regime = f["idx_regime"].to_numpy(bool)
    ends_early = off + n - 1 < len(cal) - 1
    rs = np.asarray(_G["rs"][off: off + n, tid], np.float64)
    top = np.asarray(_G["top"][off: off + n, tid], bool)
    st = max(start - off, 0)
    cl = np.nan_to_num
    g = lambda a, v: np.nan_to_num(a, nan=v)
    out = {"grid": [], "cand": {}, "rand": {}}
    mk = 0 if f["market"].iloc[-1] == "KOSPI" else 1

    def run(sig, stop, exit_cond, max_hold, all_mode):
        if stop is None:
            sm, sp = 0, 0.0
        elif stop == "atr2":
            sm, sp = 2, 0.0
        else:
            sm, sp = 1, float(stop)
        return K.run_trades(o, h, l, c, valid, locked, brk, sig, sm, sp, g(ind["atr20"], np.inf), exit_cond,
                            max_hold, st, all_mode, ends_early)

    # ---- A2
    a = kr["A2"]
    a_base = elig & (c > g(ind["ma200"], np.inf))
    a_exit = c > g(ind["ma5"], np.inf)

    def a_sig(cb):
        s = a_base & (g(ind["rsi2"], 999) <= cb["rsi"])
        return s & regime if cb["regime"] else s

    for ci, cb in enumerate(a2_grid(cfg)):
        df = _trades_df(run(a_sig(cb), cb["stop"], a_exit, a["max_hold"], False), tid)
        df.insert(0, "combo", ci)
        df.insert(0, "strategy", "A2")
        out["grid"].append(df)

    # ---- B
    b = kr["B"]
    hh = {20: g(ind["hh20"], np.inf), 60: g(ind["hh60"], np.inf)}
    mult = g(ind["mult"], 0)
    ma = {k: g(ind[k], np.nan) for k in ("ma50", "ma150", "ma200", "ma200_20")}
    tmpl = (c > ma["ma50"]) & (ma["ma50"] > ma["ma150"]) & (ma["ma150"] > ma["ma200"]) & (ma["ma200"] > ma["ma200_20"]) \
        & (c >= g(ind["lo52"], np.inf) * 1.3) & (c >= g(ind["hi52"], np.inf) * 0.75) & (g(rs, 0) >= 70)
    filt = {"none": np.ones(n, bool), "rs80": g(rs, 0) >= 80, "template": tmpl}
    exits = {"ll10": c < g(ind["ll10"], -np.inf), "ll20": c < g(ind["ll20"], -np.inf), "ma20": c < g(ind["ma20"], -np.inf)}

    def b_sig(cb):
        s = elig & (c > hh[cb["n"]]) & (mult >= cb["mult"]) & filt[cb["filter"]]
        return s & regime if cb["regime"] else s

    for ci, cb in enumerate(b_grid(cfg)):
        df = _trades_df(run(b_sig(cb), cb["stop"], exits[cb["exit"]], b["max_hold"], False), tid)
        df.insert(0, "combo", ci)
        df.insert(0, "strategy", "B")
        out["grid"].append(df)

    # ---- C (trades on day i; conditions from day i-1)
    cc = kr["C"]
    prev = lambda x: np.concatenate([[np.nan if x.dtype.kind == "f" else False], x[:-1]])
    p_elig = prev(elig.astype(float)) == 1
    p_close = prev(c)
    p_ma5 = prev(g(ind["ma5"], np.nan))
    p_noise = prev(g(ind["noise20"], np.nan))
    p_reg = prev(regime.astype(float)) == 1
    # previous *traded* row's range (the row before a traded row is traded when p_elig holds)
    p_rng = prev(g(ind["rng"], np.nan))
    upper = p_close * (1 + limit_ratio(cfg, f.index))
    base_c = p_elig & top & valid & ~locked
    base_c[:st] = False
    c_filt = {"none": np.ones(n, bool), "ma5": p_close > p_ma5, "noise": p_noise <= cc["noise_max"],
              "ma5noise": (p_close > p_ma5) & (p_noise <= cc["noise_max"])}
    c_exit = {}
    for i in np.flatnonzero(base_c):
        c_exit[i] = K.next_open_exit(o, c, valid, locked, brk, i, ends_early)
    av_prev = prev(f["avg_value"].to_numpy(np.float64))

    def c_rows(cb):
        kk = p_noise if cb["k"] == "var" else np.full(n, float(cb["k"]))
        tgt = o + p_rng * kk
        ok = base_c & c_filt[cb["filter"]] & np.isfinite(tgt) & (tgt < cc["limit_guard"] * upper)
        if cb["regime"]:
            ok &= p_reg
        return ok, tgt

    for ci, cb in enumerate(c_grid(cfg)):
        ok, tgt = c_rows(cb)
        hit = ok & (h >= tgt)
        idx = np.flatnonzero(hit)
        if len(idx):
            ex = np.array([c_exit[i] for i in idx])
            df = pd.DataFrame({"strategy": "C", "combo": ci, "tid": np.int32(tid), "sig_i": idx - 1, "entry_i": idx,
                               "exit_i": ex[:, 0].astype(np.int64), "entry_px": np.maximum(o[idx], tgt[idx]),
                               "exit_px": ex[:, 1], "reason": ex[:, 2].astype(np.int8)})
            out["grid"].append(df)

    # ---- portfolio candidates (all mode) and random tables
    for strat, lst in judged(cfg).items():
        frames = []
        for label, cb in lst:
            if strat == "A2":
                res = run(a_sig(cb), cb["stop"], a_exit, a["max_hold"], True)
                df = _trades_df(res, tid, {"prio": -g(ind["rsi2"], 999), "atr": g(ind["atr20"], np.nan)})
            elif strat in ("BO", "BM"):
                res = run(b_sig(cb), cb["stop"], exits[cb["exit"]], b["max_hold"], True)
                df = _trades_df(res, tid, {"prio": mult, "atr": g(ind["atr20"], np.nan)})
            else:
                ok, tgt = c_rows(cb)
                idx = np.flatnonzero(ok)
                if not len(idx):
                    continue
                ex = np.array([c_exit[i] for i in idx])
                df = pd.DataFrame({"tid": np.int32(tid), "sig_i": idx - 1, "entry_i": idx, "exit_i": ex[:, 0].astype(np.int64),
                                   "entry_px": np.maximum(o[idx], tgt[idx]), "exit_px": ex[:, 1],
                                   "reason": ex[:, 2].astype(np.int8), "prio": av_prev[idx],
                                   "hit": h[idx] >= tgt[idx], "atr": prev(g(ind["atr20"], np.nan))[idx]})
            df.insert(0, "label", label)
            frames.append(df)
        if frames:
            out["cand"][strat] = pd.concat(frames, ignore_index=True)
    # random tables: buy at the open of any buyable row
    sig_any = elig.copy()
    res = run(sig_any, a["main"]["stop"], a_exit, a["max_hold"], True)
    out["rand"]["A2"] = _trades_df(res, tid)
    bm = b["main_O"]
    res = run(sig_any, bm["stop"], exits[bm["exit"]], b["max_hold"], True)
    out["rand"]["B"] = _trades_df(res, tid)
    idx = np.flatnonzero(base_c)
    if len(idx):
        ex = np.array([c_exit[i] for i in idx])
        out["rand"]["C"] = pd.DataFrame({"tid": np.int32(tid), "sig_i": idx - 1, "entry_i": idx,
                                         "exit_i": ex[:, 0].astype(np.int64), "entry_px": o[idx], "exit_px": ex[:, 1],
                                         "reason": ex[:, 2].astype(np.int8)})
    # shift row indices to calendar indices
    for part in [out["grid"], list(out["cand"].values()), list(out["rand"].values())]:
        for df in part:
            for col in ("sig_i", "entry_i", "exit_i"):
                df[col] = df[col] + off
    keep = slice(max(k0 - off, 0), n)
    arrays = {"close": c[keep].astype(np.float64), "valid": valid[keep]}
    meta = (tid, _G["uni"].index[tid], mk, bool(ends_early), max(off, k0), len(c[keep]))
    out["grid"] = pd.concat(out["grid"], ignore_index=True) if out["grid"] else None
    return tid, (out, arrays, meta)


def pass2(cfg) -> None:
    uni = universe_tickers(cfg)
    d = C.results_dir(OUT)
    grid, cand, rand, closes, valids, meta = [], {}, {}, [], [], []
    pos = 0
    with ProcessPoolExecutor(cfg["kr"]["workers"], initializer=_init) as ex:
        for tid, res in ex.map(_pass2, range(len(uni)), chunksize=8):
            if res is None:
                continue
            out, arr, m = res
            if out["grid"] is not None:
                grid.append(out["grid"])
            for k, v in out["cand"].items():
                cand.setdefault(k, []).append(v)
            for k, v in out["rand"].items():
                rand.setdefault(k, []).append(v)
            closes.append(arr["close"])
            valids.append(arr["valid"])
            meta.append((*m, pos))
            pos += len(arr["close"])
    g = pd.concat(grid, ignore_index=True)
    g["strategy"] = g["strategy"].astype("category")
    g.to_parquet(d / "grid_trades.parquet")
    for k, v in cand.items():
        pd.concat(v, ignore_index=True).to_parquet(d / f"cand_{k}.parquet")
    for k, v in rand.items():
        pd.concat(v, ignore_index=True).to_parquet(d / f"rand_{k}.parquet")
    pdir = C.results_dir(OUT + "/panel")
    np.save(pdir / "close.npy", np.concatenate(closes))
    np.save(pdir / "valid.npy", np.concatenate(valids))
    pd.DataFrame(meta, columns=["tid", "ticker", "market", "ends_early", "off", "n", "pos"]).to_parquet(pdir / "meta.parquet")
    pd.Series(calendar(cfg)).to_frame("date").to_parquet(pdir / "calendar.parquet")
    print(f"pass 2: grid trades {len(g)}; " + ", ".join(f"cand_{k} {sum(len(x) for x in v)}" for k, v in cand.items())
          + "; " + ", ".join(f"rand_{k} {sum(len(x) for x in v)}" for k, v in rand.items()))


def main() -> None:
    cfg = C.load_config()
    pass1(cfg)
    pass2(cfg)


if __name__ == "__main__":
    main()
