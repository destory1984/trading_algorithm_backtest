"""Trade kernels on one ticker's calendar rows (halted rows have valid=False).

run_trades: a signal at the close of row i buys at the open of row i+1 (dropped if that row is halted or locked).
Then, row by row from the entry (halted rows are skipped but counted as holding days, as in bnf):
  - a data break (krxbt data_break) after the entry voids the trade at the previous traded close   reason BRK
  - an exit decided at a close (exit_cond) sells at the next traded, non-locked open                  reason LINE
  - the low touching the stop sells at min(open, stop); on a locked row the sale waits for the next
    traded, non-locked open                                                                         reason STOP
  - holding day max_hold (entry = day 1) sells at that close (the next traded close if halted)      reason HOLD
  - the ticker's last row sells at its close: END if its data stops before the calendar end (delisted or
    halted to the end), else OPEN (still held when the data ends)
The stop is checked before the exit line; on the entry row the stop can hit (the open is the entry, so the fill is
the stop). seq mode skips signals until the previous trade has been sold (signal row >= exit row); all mode makes a
trade for every signal (the portfolio later drops signals on stocks it holds).
stop_mode: 0 none, 1 entry x (1 + stop_pct), 2 entry - 2 x atr[signal row].

c_trades: volatility breakout on day i (conditions from day i-1 are prepared by the caller): bought at the target
(max(open, target)) when the high reaches it, sold at the next traded, non-locked open (BRK / END / OPEN as above).
"""
from __future__ import annotations

import numba
import numpy as np

LINE, STOP, HOLD, END, BRK, OPEN = 0, 1, 2, 3, 4, 5
REASONS = {LINE: "line", STOP: "stop", HOLD: "max_hold", END: "data_end", BRK: "break_void", OPEN: "open_at_end"}


@numba.njit(cache=True)
def run_trades(o, h, l, c, valid, locked, brk, sig, stop_mode, stop_pct, atr, exit_cond, max_hold, start_i,
               all_mode, ends_early):
    n = len(o)
    cap = n if not all_mode else int(sig.sum()) + 1
    s_i = np.empty(cap, np.int64)
    e_i = np.empty(cap, np.int64)
    x_i = np.empty(cap, np.int64)
    e_px = np.empty(cap, np.float64)
    x_px = np.empty(cap, np.float64)
    why = np.empty(cap, np.int64)
    k = 0
    free = 0
    for i in range(start_i, n - 1):
        if not sig[i] or (not all_mode and i < free):
            continue
        e = i + 1
        if not valid[e] or locked[e]:
            continue
        entry = o[e]
        if stop_mode == 1:
            stop = entry * (1.0 + stop_pct)
        elif stop_mode == 2:
            stop = entry - 2.0 * atr[i]
        else:
            stop = -1.0
        pend_open = False
        pend_why = LINE
        last_valid = e
        done = False
        xi, xp, wy = -1, 0.0, -1
        for j in range(e, n):
            if not valid[j]:
                continue
            if j > e and brk[j]:
                xi, xp, wy = last_valid, c[last_valid], BRK
                done = True
                break
            if pend_open:
                if locked[j]:
                    last_valid = j
                    if j == n - 1:
                        xi, xp, wy = j, c[j], END if ends_early else OPEN
                        done = True
                        break
                    continue
                xi, xp, wy = j, o[j], pend_why
                done = True
                break
            if stop_mode > 0 and l[j] <= stop:
                if locked[j]:
                    pend_open = True
                    pend_why = STOP
                else:
                    xi, xp, wy = j, min(o[j], stop), STOP
                    done = True
                    break
            if not pend_open and exit_cond[j]:
                pend_open = True
                pend_why = LINE
            if not pend_open and j - e + 1 >= max_hold:
                xi, xp, wy = j, c[j], HOLD
                done = True
                break
            last_valid = j
            if j == n - 1:
                xi, xp, wy = j, c[j], END if ends_early else OPEN
                done = True
                break
        if not done:
            xi, xp, wy = last_valid, c[last_valid], END if ends_early else OPEN
        s_i[k], e_i[k], x_i[k], e_px[k], x_px[k], why[k] = i, e, xi, entry, xp, wy
        k += 1
        free = xi
    return s_i[:k], e_i[:k], x_i[:k], e_px[:k], x_px[:k], why[:k]


@numba.njit(cache=True)
def next_open_exit(o, c, valid, locked, brk, i, ends_early):
    """Sell a position bought on row i at the next traded, non-locked open."""
    n = len(o)
    last_valid = i
    for j in range(i + 1, n):
        if not valid[j]:
            continue
        if brk[j]:
            return last_valid, c[last_valid], BRK
        if locked[j]:
            last_valid = j
            continue
        return j, o[j], LINE
    return last_valid, c[last_valid], END if ends_early else OPEN
