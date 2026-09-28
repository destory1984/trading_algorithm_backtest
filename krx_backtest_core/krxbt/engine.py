"""Trade-by-trade simulation for one ticker.

The algorithm decides two boolean arrays on the ticker frame:
  cand     buy signal on this day's close
  target   the take-profit condition holds at this day's close

The engine turns them into trades (config `rules`, `costs`):
  entry    next trading day's open (option rules.entry_on_signal_close: the
           signal-day close). If the stock is halted that day the signal is
           skipped.
  exits    checked in this order each priced day, first hit wins:
           1. target held on the previous priced day -> exit at this open
           2. open <= stop price -> exit at the open (gap through the stop)
           3. low  <= stop price -> exit at the stop price
           4. target holds -> exit next priced day's open
              (option rules.target_exit_on_close: this close)
           5. holding day >= max hold -> exit at this close
  Holding day 1 is the entry day. Halted days count as holding days but no
  exit happens on them; a due time exit waits for the next priced day.
  If the data ends while holding (delisting or long halt) the trade is closed
  at the last traded close, reason "forced".
  While a position is open, new signals on that ticker are ignored, unless
  independent=True (every signal becomes its own trade; a portfolio then
  applies "ignore while holding" to its real holdings).
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

from .costs import cost_factors, net_return

REASONS = np.array(["target", "stop", "time", "forced"])
R_TARGET, R_STOP, R_TIME, R_FORCED = 0, 1, 2, 3


@numba.njit(cache=True)
def run_kernel(o, h, l, c, target, valid, cand, stop, hold, entry_on_close, target_on_close,
               buy_cost, sell_keep, independent=False):
    """Returns (sig, ent, ext, entry_px, exit_px, ret, reason, rebound) as bar indices / values.

    rebound: target held on any priced day within the max-hold window,
    whatever the stop did.
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
                if target[d]:
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
        rb = False
        for j in range(e, min(e + hold, n)):
            if valid[j] and target[j]:
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


def arrays(f: pd.DataFrame) -> dict:
    """Numpy views of a krxbt.frame.ticker_frame, computed once per ticker."""
    return {k: f[k].to_numpy(np.float64) for k in ("open", "high", "low", "close", "ma", "disp", "idx_disp")} | {
        "valid": f["valid"].to_numpy(np.bool_),
        "eligible": f["eligible"].to_numpy(np.bool_),
        "uptrend": f["idx_regime"].to_numpy(np.bool_),
        "crash": f["idx_crash"].to_numpy(np.bool_),
        "break_cum": np.cumsum(f["data_break"].to_numpy(np.int32)) if "data_break" in f else np.zeros(len(f), np.int32),
        "dates": f.index.to_numpy(),
    }


def market_mask(a: dict, market_filter: str) -> np.ndarray:
    """none / uptrend (index close > index MA) / crash (index disparity <=
    indicators.index_disparity_crash) / crashNN (index disparity <= NN)."""
    if market_filter == "none":
        return np.ones(len(a["valid"]), np.bool_)
    if market_filter in ("uptrend", "crash"):
        return a[market_filter]
    if market_filter.startswith("crash"):
        return a["idx_disp"] <= float(market_filter[5:])  # NaN compares False
    raise ValueError(f"unknown market_filter {market_filter!r}")


def run(a: dict, cfg: dict, cand: np.ndarray, target: np.ndarray, stop: float | None, hold: int,
        independent: bool = False, extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Trades as a DataFrame. `extra` arrays are sampled on the signal day
    (e.g. {"disp": a["disp"]}) and placed before idx_disp."""
    buy_cost, sell_keep = cost_factors(cfg)
    ru = cfg["rules"]
    sig, ent, ext, epx, xpx, _, rsn, reb = run_kernel(
        a["open"], a["high"], a["low"], a["close"], target, a["valid"], cand,
        stop if stop is not None else 0.0, int(hold),
        bool(ru["entry_on_signal_close"]), bool(ru["target_exit_on_close"]), buy_cost, sell_keep, independent)
    dates = a["dates"]
    return pd.DataFrame({
        "signal_date": dates[sig], "entry_date": dates[ent], "exit_date": dates[ext],
        "entry_px": epx, "exit_px": xpx, "ret": net_return(cfg, epx, xpx, dates[ext]),
        "hold_days": (ext - ent + 1).astype(np.int16),
        "reason": REASONS[rsn], "rebound": reb,
        **{k: v[sig] for k, v in (extra or {}).items()},
        "idx_disp": a["idx_disp"][sig],
        # a price-data break on any day after entry up to the exit
        "data_break": a["break_cum"][ext] > a["break_cum"][ent],
    })
