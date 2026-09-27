"""Stage 3: trade-by-trade simulation for one ticker.

Rules (config.yaml `rules`, `costs`):
  signal   close disparity <= threshold on an eligible day, plus the market
           filter: none / uptrend (index close > index MA120) /
           crashNN (index disparity on the signal day <= NN; plain "crash"
           uses config index_disparity_crash)
  entry    next trading day's open (option: signal-day close). If the stock is
           halted on that day the signal is skipped.
  exits    checked in this order each priced day, first hit wins:
           1. target reached on the previous priced day -> exit at this open
           2. open <= stop price -> exit at the open (gap through the stop)
           3. low  <= stop price -> exit at the stop price
           4. close >= MA25 -> target: exit next priced day's open
              (option: this close)
           5. holding day >= max hold -> exit at this close
  Holding day 1 is the entry day. Halted days count as holding days but no
  exit happens on them; a due time exit waits for the next priced day.
  If the data ends while holding (delisting or long halt) the trade is closed
  at the last traded close, reason "forced".
  While a position is open, new signals on that ticker are ignored.

Usage:  python -m src.simulate 005930 [threshold] [stop] [hold] [none|uptrend|crash97|...]
        prints every trade with the bars around it, for hand checking.
"""
from __future__ import annotations

import sys

import numba
import numpy as np
import pandas as pd

REASONS = np.array(["target", "stop", "time", "forced"])
R_TARGET, R_STOP, R_TIME, R_FORCED = 0, 1, 2, 3


@numba.njit(cache=True)
def run_kernel(o, h, l, c, ma, valid, cand, stop, hold, entry_on_close, target_on_close,
               buy_cost, sell_keep, independent=False):
    """Returns (sig, ent, ext, entry_px, exit_px, ret, reason, rebound).

    independent=True turns every candidate signal into its own trade, even
    while an earlier trade on the ticker is still open. The portfolio uses
    this and applies "ignore signals while holding" to its real holdings.
    """
    n = len(c)
    sig = np.empty(n, np.int32)
    ent = np.empty(n, np.int32)
    ext = np.empty(n, np.int32)
    epx = np.empty(n, np.float64)
    xpx = np.empty(n, np.float64)
    ret = np.empty(n, np.float64)
    rsn = np.empty(n, np.int8)
    reb = np.empty(n, np.bool_)
    k = 0
    next_ok = 0
    for s in range(n):
        if not cand[s] or s < next_ok:
            continue
        if entry_on_close:
            e = s
            entry = c[s]
        else:
            e = s + 1
            if e >= n or not valid[e]:
                continue
            entry = o[e]
        stop_px = entry * (1.0 + stop) if stop < 0 else -1.0
        pending = False
        time_due = False
        last_valid = e
        x = -1
        xp = 0.0
        r = -1
        at_open = False
        d = e
        while True:
            if d >= n:
                x = last_valid
                xp = c[last_valid]
                r = R_FORCED
                break
            held = d - e + 1
            if held >= hold:
                time_due = True
            if valid[d]:
                intraday = not (entry_on_close and d == e)
                if intraday:
                    if pending:
                        x, xp, r, at_open = d, o[d], R_TARGET, True
                        break
                    if o[d] <= stop_px:
                        x, xp, r, at_open = d, o[d], R_STOP, True
                        break
                    if l[d] <= stop_px:
                        x, xp, r = d, stop_px, R_STOP
                        break
                if c[d] >= ma[d]:
                    if target_on_close:
                        x, xp, r = d, c[d], R_TARGET
                        break
                    if not time_due:
                        pending = True
                if time_due:
                    x, xp, r = d, c[d], R_TIME
                    break
                last_valid = d
            d += 1
        # rebound: any close >= MA25 within the max-hold window (independent of stops)
        rb = False
        for j in range(e, min(e + hold, n)):
            if valid[j] and c[j] >= ma[j]:
                rb = True
                break
        sig[k] = s
        ent[k] = e
        ext[k] = x
        epx[k] = entry
        xpx[k] = xp
        ret[k] = xp * sell_keep / (entry * buy_cost) - 1.0
        rsn[k] = r
        reb[k] = rb
        k += 1
        if not independent:
            next_ok = x if at_open else x + 1
    return sig[:k], ent[:k], ext[:k], epx[:k], xpx[:k], ret[:k], rsn[:k], reb[:k]


def cost_factors(cfg: dict) -> tuple[float, float]:
    cs = cfg["costs"]
    buy_cost = (1 + cs["slippage"]) * (1 + cs["buy_fee"])
    sell_keep = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - cs["sell_tax"])
    return buy_cost, sell_keep


def net_return(cfg: dict, entry_px, exit_px, exit_dates) -> np.ndarray:
    """Return after fees, slippage and the sell tax in force on each exit date."""
    cs = cfg["costs"]
    buy_cost, _ = cost_factors(cfg)
    dates = pd.DatetimeIndex(exit_dates)
    tax = np.full(len(dates), cs["sell_tax"], dtype=np.float64)
    for since, rate in cs.get("sell_tax_schedule") or []:
        tax[dates >= pd.Timestamp(since)] = rate
    keep = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - tax)
    return np.asarray(exit_px, np.float64) * keep / (np.asarray(entry_px, np.float64) * buy_cost) - 1.0


def arrays(f: pd.DataFrame) -> dict:
    return {k: f[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "ma", "disp", "idx_disp")} | {
        "valid": f["valid"].to_numpy(np.bool_),
        "eligible": f["eligible"].to_numpy(np.bool_),
        "uptrend": f["idx_regime"].to_numpy(np.bool_),
        "crash": f["idx_crash"].to_numpy(np.bool_),
        "break_cum": np.cumsum(f["data_break"].to_numpy(np.int32)) if "data_break" in f else np.zeros(len(f), np.int32),
        "idx_disp_raw": f["idx_disp"].to_numpy(np.float64),
    }


def candidates(a: dict, threshold: float, market_filter: str) -> np.ndarray:
    cand = a["eligible"] & (a["disp"] <= threshold)
    if market_filter == "uptrend" or market_filter == "crash":
        cand &= a[market_filter]
    elif market_filter.startswith("crash"):
        cand &= a["idx_disp_raw"] <= float(market_filter[5:])  # NaN compares False
    elif market_filter != "none":
        raise ValueError(f"unknown market_filter {market_filter!r}")
    return cand


def simulate(f: pd.DataFrame, cfg: dict, threshold: float, stop: float | None, hold: int,
             market_filter: str = "none", a: dict | None = None,
             independent: bool = False) -> pd.DataFrame:
    a = a or arrays(f)
    cand = candidates(a, threshold, market_filter)
    buy_cost, sell_keep = cost_factors(cfg)
    ru = cfg["rules"]
    sig, ent, ext, epx, xpx, ret, rsn, reb = run_kernel(
        a["open"], a["high"], a["low"], a["close"], a["ma"], a["valid"], cand,
        stop if stop is not None else 0.0, int(hold),
        bool(ru["entry_on_signal_close"]), bool(ru["target_exit_on_close"]), buy_cost, sell_keep, independent)
    dates = f.index.to_numpy()
    return pd.DataFrame({
        "signal_date": dates[sig], "entry_date": dates[ent], "exit_date": dates[ext],
        "entry_px": epx, "exit_px": xpx, "ret": net_return(cfg, epx, xpx, dates[ext]),
        "hold_days": (ext - ent + 1).astype(np.int16),
        "reason": REASONS[rsn], "rebound": reb,
        "disp": a["disp"][sig], "idx_disp": a["idx_disp"][sig],
        # a price-data break on any day after entry up to the exit
        "data_break": a["break_cum"][ext] > a["break_cum"][ent],
    })


def main() -> None:
    from .common import load_config, load_calendar
    from .indicators import index_features, ticker_frame

    cfg = load_config()
    t = sys.argv[1] if len(sys.argv) > 1 else "005930"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 85
    stop = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "none" else None
    hold = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    from .grid import universe_tickers
    info = universe_tickers(cfg).loc[t]
    f = ticker_frame(cfg, t, info["market"], load_calendar(cfg), index_features(cfg), bool(info["delisted"]))
    tr = simulate(f, cfg, thr, stop, hold, sys.argv[5] if len(sys.argv) > 5 else "none")
    pd.set_option("display.width", 200)
    print(f"{t} {info['market']} threshold={thr} stop={stop} hold={hold}: {len(tr)} trades")
    print(tr.round(4).to_string())
    for _, row in tr.head(3).iterrows():
        lo = f.index.get_loc(row["signal_date"])
        hi = f.index.get_loc(row["exit_date"])
        print(f"\n--- trade signal {row['signal_date'].date()} -> exit {row['exit_date'].date()} ({row['reason']})")
        print(f.iloc[lo: hi + 2][["open", "high", "low", "close", "ma", "disp", "valid"]].round(1).to_string())


if __name__ == "__main__":
    main()
