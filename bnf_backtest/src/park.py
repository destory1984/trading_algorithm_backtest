"""Park idle cash in the index while the bnf portfolio waits for signals.

Same rules as krxbt.portfolio.run_portfolio (slots, 1/slots of the previous
close's equity, lowest disparity first, re-entry cooldown, crash slots).
Only the cash changes: with a park frame, cash left at a close buys the
index at that close, and an entry that needs more cash than is on hand
sells the index at the index open of the entry day. Index trades pay the
park costs. park=None runs the engine's bookkeeping unchanged.

Usage:  python -m src.park          # Korea, results/park.csv
        python -m src.park --us     # US, results/us/park.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors
from krxbt.portfolio import stats


def park_frame(ix: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Index open/close on the trading calendar. Missing closes are
    forward-filled; a missing open falls back to that filled close."""
    close = ix["close"].reindex(cal).ffill()
    opn = ix["open"].reindex(cal).where(lambda s: s > 0).fillna(close)
    return pd.DataFrame({"open": opn, "close": close})


def park_portfolio(cal: pd.DatetimeIndex, tr: pd.DataFrame, closes: pd.DataFrame, capital: float,
                   slots: int, rank_by: str, cooldown: int = 0, crash_slots: int = 0,
                   crash_level: float | None = None, park: pd.DataFrame | None = None,
                   costs: tuple[float, float] = (1.0, 1.0)) -> tuple[pd.Series, dict]:
    buy_cost, sell_keep = costs
    base = int(slots)
    wide = max(base, int(crash_slots or 0))
    day_no = {d: i for i, d in enumerate(cal)}
    blocked_until: dict[str, int] = {}
    ledger: list[dict] = []
    by_entry = {d: g.sort_values(rank_by) for d, g in tr.groupby("entry_date")} if len(tr) else {}
    cash = float(capital)
    units = 0.0
    park_buys = park_sells = 0
    pos: dict[str, dict] = {}
    equity = np.empty(len(cal))
    book = np.zeros((len(cal), 4))
    prev_eq = cash
    taken = 0
    taken_rets: list[float] = []
    invested_days = 0
    max_held = 0
    exposure: list[float] = []
    for i, d in enumerate(cal):
        if d in by_entry and len(pos) < wide:
            p_open = float(park.at[d, "open"]) if park is not None else 0.0
            for _, t in by_entry[d].iterrows():
                cap = wide if crash_level is not None and t["idx_disp"] <= crash_level else base
                if len(pos) >= cap:
                    continue
                tk = str(t["ticker"])
                if tk in pos:
                    continue
                if i <= blocked_until.get(tk, -1):
                    continue
                avail = cash + units * p_open * sell_keep
                alloc = min(prev_eq / cap, avail)
                if alloc <= 0:
                    break
                if alloc > cash:  # sell just enough index at today's index open
                    need = alloc - cash
                    sold = min(units, need / (p_open * sell_keep))
                    units -= sold
                    cash += sold * p_open * sell_keep
                    park_sells += 1
                    alloc = min(alloc, cash)
                cash -= alloc
                ledger.append({"ticker": tk, "entry_date": d, "exit_date": t["exit_date"], "alloc": alloc,
                               "ret": float(t["ret"]), "reason": t["reason"]})
                pos[tk] = {"alloc": alloc, "entry_px": float(t["entry_px"]),
                           "exit_date": t["exit_date"], "ret": float(t["ret"]), "reason": t["reason"]}
                taken += 1
                taken_rets.append(float(t["ret"]))
        max_held = max(max_held, len(pos))
        for tk in [k for k, p in pos.items() if p["exit_date"] <= d]:
            p = pos.pop(tk)
            cash += p["alloc"] * (1 + p["ret"])
            if cooldown and p["reason"] == "stop":
                blocked_until[tk] = day_no.get(p["exit_date"], i) + cooldown
        p_close = float(park.at[d, "close"]) if park is not None else 0.0
        if park is not None and cash > 0:  # idle cash buys the index at today's close
            units += cash / (p_close * buy_cost)
            cash = 0.0
            park_buys += 1
        row = closes.loc[d]
        val = 0.0
        for tk, p in pos.items():
            c = row.get(tk)
            val += p["alloc"] * (c / p["entry_px"] if pd.notna(c) else 1.0)
        pv = units * p_close
        prev_eq = cash + pv + val
        equity[i] = prev_eq
        book[i] = (cash, units, pv, val)
        invested_days += bool(pos)
        if pos:
            exposure.append(val / prev_eq)
    eq = pd.Series(equity, index=cal, name="equity")
    eq.attrs["ledger"] = pd.DataFrame(ledger)
    eq.attrs["book"] = pd.DataFrame(book, index=cal, columns=["cash", "units", "park_value", "stock_value"])
    return eq, {"trades_taken": taken, "trades_available": len(tr),
                "taken_mean_ret": float(np.mean(taken_rets)) if taken_rets else np.nan,
                "all_mean_ret": float(tr["ret"].mean()) if len(tr) else np.nan,
                "invested_share": invested_days / len(cal), "max_held": max_held,
                "exposure": float(np.mean(exposure)) if exposure else 0.0,
                "park_buys": park_buys, "park_sells": park_sells}


def park_costs(cost_cfg: dict) -> tuple[float, float]:
    return cost_factors({"costs": cost_cfg})


def _row(eq: pd.Series, te: pd.Timestamp) -> dict:
    return {**stats(eq), "cagr_train": stats(eq[eq.index <= te])["cagr"],
            "cagr_test": stats(eq[eq.index > te])["cagr"]}


def _checks(eq: pd.Series, eng: pd.Series | None) -> dict:
    b = eq.attrs["book"]
    # book.sum(axis=1) is exactly how eq is built inside park_portfolio (equity[i] = cash+pv+val),
    # so this "ledger" sum can never disagree with eq; it is kept only as a column, not a check.
    out = {"ledger_diff": float((b[["cash", "park_value", "stock_value"]].sum(axis=1) - eq).abs().max()),
           "min_cash": float(b["cash"].min()), "min_units": float(b["units"].min())}
    if eng is not None:
        out["engine_diff"] = float((eq - eng).abs().max())
    return out


def _conserve(cal, tr, closes, cap, slots, cooldown, off) -> float:
    """A real conservation check for the parked path: park in an index whose price never moves
    (open = close = 1) with no cost (buy/sell factor 1.0). Buying and selling a fixed-price,
    free asset around trades cannot create or destroy money, so this parked curve must equal
    the cash-idle curve (off) to the won."""
    const = pd.DataFrame({"open": 1.0, "close": 1.0}, index=cal)
    cst, _ = park_portfolio(cal, tr, closes, cap, slots, "disp", cooldown=cooldown, park=const, costs=(1.0, 1.0))
    return float((cst - off).abs().max())


def _pair(cal, tr, closes, cap, slots, cooldown, pf, costs, te) -> tuple[list[dict], dict]:
    """Cash-idle (checked against the engine) and index-parked runs of one trade list."""
    from krxbt import portfolio as kp
    eng, _ = kp.run_portfolio(cal, tr, closes, cap, slots, rank_by="disp", cooldown=cooldown)
    off, info0 = park_portfolio(cal, tr, closes, cap, slots, "disp", cooldown=cooldown)
    on, info1 = park_portfolio(cal, tr, closes, cap, slots, "disp", cooldown=cooldown, park=pf, costs=costs)
    conserve_diff = _conserve(cal, tr, closes, cap, slots, cooldown, off)
    rows = [{"parking": "현금", **_row(off, te), **info0, **_checks(off, eng), "conserve_diff": conserve_diff},
            {"parking": "지수", **_row(on, te), **info1, **_checks(on, None), "conserve_diff": conserve_diff}]
    return rows, {"현금": off, "지수": on}


def _nosignal(cal, closes, cap, slots, pf, costs) -> float:
    """No trades: the parked curve must be index buy-and-hold less one buy cost."""
    empty = pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "entry_px", "ret", "reason",
                                  "idx_disp", "disp"])
    eq, _ = park_portfolio(cal, empty, closes, cap, slots, "disp", park=pf, costs=costs)
    want = cap / (pf["close"].iloc[0] * costs[0]) * pf["close"]
    return float((eq - want).abs().max())


def run_kr() -> None:
    from krxbt.data import load_index
    from .common import load_config, results_dir
    from .portfolio import _cal, _close_panel, candidate_trades, combo_trades, compare_selection, filter_label
    cfg = load_config()
    st, pk = cfg["stage2"], cfg["park"]
    te = pd.Timestamp(st["train_end"])
    cap, slots, mk = st["initial_capital"], int(st["max_positions"]), st["compare_combo"]["market"]
    sel = compare_selection(cfg)
    cand = candidate_trades(cfg, list(sel.index))
    cal = _cal(cfg)
    closes = _close_panel(cfg, cand["ticker"].astype(str).unique(), cal)
    pf = park_frame(load_index(cfg, pk["kr_index"]), cal)
    costs = park_costs(pk["kr_costs"])
    rd = results_dir()
    old = pd.read_csv(rd / "portfolio_filters.csv")
    old = old[old["method"] == "portfolio"].set_index(["filter", "cooldown"])["cagr"]
    rows, curves = [], {}
    for cid, c in sel.iterrows():
        tr = combo_trades(cand, int(cid), mk)
        for cd in st.get("reentry_cooldowns", [0]):
            pair, eqs = _pair(cal, tr, closes, cap, slots, int(cd), pf, costs, te)
            for r in pair:
                r.update({"market": mk, "filter": filter_label(c["market_filter"]), "cooldown": int(cd)})
                key = (c["market_filter"], int(cd))
                if r["parking"] == "현금" and key in old.index:
                    r["old_cagr_diff"] = abs(r["cagr"] - float(old.loc[key]))
                rows.append(r)
                curves[f"{c['market_filter']}_cd{cd}_{r['parking']}"] = eqs[r["parking"]]
            print(f"{c['market_filter']} cd {cd}: cash {pair[0]['cagr']:+.3f}/{pair[0]['mdd']:.3f} "
                  f"index {pair[1]['cagr']:+.3f}/{pair[1]['mdd']:.3f}")
    bh = cap * pf["close"] / pf["close"].iloc[0]
    rows.append({"market": mk, "filter": f"{pk['kr_index']} 보유", "parking": "-", **_row(bh, te)})
    rows.append({"market": mk, "filter": "-", "parking": "신호 없음 점검",
                 "nosignal_diff": _nosignal(cal, closes, cap, slots, pf, costs)})
    curves[f"{pk['kr_index']} 보유"] = bh
    pd.DataFrame(rows).to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")


def run_us() -> None:
    from krxbt import us
    from .common import results_dir
    from .portfolio import filter_label
    from .us import us_config
    from .walkforward import us_candidates
    cfg = us_config()
    u, pk = cfg["us"], cfg["park"]
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    cap, slots = u["initial_capital"], int(cfg["stage2"]["max_positions"])
    rd = results_dir() / "us"
    s = pd.read_csv(rd / "grid_summary.csv")
    top = s[(s["kind"] == "stock") & (s["trades"] >= u["min_trades"])].head(u["portfolio_top_n"])
    pick = pd.concat([top, top.assign(kind="ALL")], ignore_index=True)
    cand, closes = us_candidates(cfg, [(int(c), "") for c in pick["combo"]])
    cal = us.load_calendar(cfg)
    cal = cal[cal >= pd.Timestamp(cfg["data"]["start"])]
    closes = closes.reindex(cal).ffill()
    n_tickers = int(closes.shape[1])
    pf = park_frame(us.load_index(cfg, pk["us_index"]), cal)
    costs = park_costs(u["costs"])
    rows, curves = [], {}
    for _, c in pick.iterrows():
        tr = cand[cand["combo"] == int(c["combo"])]
        if c["kind"] != "ALL":
            tr = tr[tr["kind"] == c["kind"]]
        pair, eqs = _pair(cal, tr, closes, cap, slots, 0, pf, costs, te)
        kind = "주식" if c["kind"] == "stock" else "주식+ETF"
        for r in pair:
            r.update({"market": "US", "kind": kind, "filter": filter_label(c["market_filter"]), "cooldown": 0,
                      "n_tickers": n_tickers})
            rows.append(r)
            curves[f"{kind}_{c['market_filter']}_{r['parking']}"] = eqs[r["parking"]]
        print(f"{kind} {c['market_filter']}: cash {pair[0]['cagr']:+.3f}/{pair[0]['mdd']:.3f} "
              f"index {pair[1]['cagr']:+.3f}/{pair[1]['mdd']:.3f}")
    bh = cap * pf["close"] / pf["close"].iloc[0]
    rows.append({"market": "US", "filter": f"{pk['us_index']} 보유", "parking": "-", **_row(bh, te)})
    rows.append({"market": "US", "filter": "-", "parking": "신호 없음 점검",
                 "nosignal_diff": _nosignal(cal, closes, cap, slots, pf, costs)})
    curves[f"{pk['us_index']} 보유"] = bh
    pd.DataFrame(rows).to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true")
    run_us() if ap.parse_args().us else run_kr()


if __name__ == "__main__":
    main()
