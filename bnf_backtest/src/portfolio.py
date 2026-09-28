"""Stage 5: portfolio simulation for the top combos.

Capital 100M KRW, at most 5 positions, equal weight (each new position gets
1/5 of the previous close's equity, limited by cash). When more signals than
free slots arrive on the same day, the lowest disparity wins. Positions that
exit on day d free their slot from day d+1.

Every signal is simulated as its own trade (simulate independent=True), and
the portfolio ignores a signal only while it really holds that ticker, as the
spec says. (The grid's per-ticker trade list also ignores signals while a
trade the portfolio never took is open; that version is kept only for the
before/after table.)

Optional re-entry cooldown: after a stop-loss exit on a ticker, skip new
entries on it for N trading days (stage2.reentry_cooldowns).

Usage:  python -m src.portfolio          # top combos from grid_summary.csv
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from krxbt import portfolio as kp
from krxbt.portfolio import stats

from .common import load_calendar, load_config, results_dir


def filter_label(v: str) -> str:
    if v == "none":
        return "필터없음"
    if v == "uptrend":
        return "상승장만"
    return f"동반급락≤{v[5:]}" if v[5:] else "동반급락"


def top_combos(cfg: dict, s: pd.DataFrame | None = None) -> pd.DataFrame:
    s = s if s is not None else pd.read_csv(results_dir() / "grid_summary.csv")
    s = s[s["trades"] >= cfg["stage2"]["min_trades"]]
    return s.sort_values("expectancy", ascending=False).head(cfg["stage2"]["top_n"])


def combo_trades(trades: pd.DataFrame, combo: int, market: str) -> pd.DataFrame:
    t = trades[trades["combo"] == combo]
    if market != "ALL":
        t = t[t["market"] == market]
    return t


_W: dict = {}


def _init(cfg, combo_rows):
    from .grid import _init as grid_init
    grid_init(cfg)
    _W["rows"] = combo_rows


def _candidates_for_ticker(args):
    from .grid import _W as G
    from krxbt.frame import ticker_frame
    from .simulate import arrays, simulate
    ticker, market, delisted = args
    cfg = G["cfg"]
    f = ticker_frame(cfg, ticker, market, G["cal"], G["idx"], delisted)
    if f is None:
        return None
    a = arrays(f)
    out = []
    for cid, c in _W["rows"]:
        stop = None if pd.isna(c["stop"]) else float(c["stop"])
        t = simulate(f, cfg, c["threshold"], stop, int(c["hold"]), c["market_filter"], a, independent=True)
        t = t[~t["data_break"]]
        if len(t):
            t = t.drop(columns="data_break")
            t.insert(0, "combo", cid)
            out.append(t)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df["ticker"], df["market"] = ticker, market
    return df


def candidate_trades(cfg: dict, combo_ids: list[int]) -> pd.DataFrame:
    """Every signal of the given combos on every ticker, each as its own trade."""
    from .grid import combos, universe_tickers
    cmb = combos(cfg)
    rows = [(int(c), cmb.loc[c].to_dict()) for c in sorted(set(combo_ids))]
    tk = universe_tickers(cfg)
    jobs = list(zip(tk.index, tk["market"], tk["delisted"]))
    with ProcessPoolExecutor(cfg["grid"]["workers"], initializer=_init, initargs=(cfg, rows)) as ex:
        parts = [p for p in ex.map(_candidates_for_ticker, jobs, chunksize=8) if p is not None]
    return pd.concat(parts, ignore_index=True)


def _cal(cfg: dict) -> pd.DatetimeIndex:
    cal = load_calendar(cfg)
    return cal[cal >= pd.Timestamp(cfg["data"]["start"])]


def _close_panel(cfg: dict, tickers, cal: pd.DatetimeIndex) -> pd.DataFrame:
    return kp.close_panel(cfg, tickers, cal)


def run_portfolio(cfg: dict, tr: pd.DataFrame, cooldown: int = 0,
                  closes: pd.DataFrame | None = None, slots: int | None = None,
                  crash_slots: int = 0, crash_level: float | None = None) -> tuple[pd.Series, dict]:
    """krxbt.portfolio.run_portfolio with this repo's stage2 settings. Lowest
    disparity wins when signals outnumber free slots. slots overrides
    stage2.max_positions."""
    st = cfg["stage2"]
    cal = _cal(cfg)
    if closes is None:
        closes = _close_panel(cfg, tr["ticker"].astype(str).unique(), cal)
    return kp.run_portfolio(cal, tr, closes, st["initial_capital"], int(slots or st["max_positions"]),
                            rank_by="disp", cooldown=cooldown, crash_slots=crash_slots, crash_level=crash_level)


def kospi_curve(cfg: dict, cal: pd.DatetimeIndex, capital: float) -> pd.Series:
    return kp.index_curve(cfg, cal, capital, "KOSPI")


def _label(rank, c) -> str:
    stop = "손절없음" if pd.isna(c["stop"]) else format(c["stop"], ".0%")
    return f"#{rank} {c['market']} ≤{c['threshold']} {stop} {int(c['hold'])}일 {filter_label(c['market_filter'])}"


def compare_selection(cfg: dict) -> pd.DataFrame:
    from .grid import combos
    cc = cfg["stage2"].get("compare_combo")
    if not cc:
        return pd.DataFrame()
    cmb = combos(cfg)
    stop = cc.get("stop")
    return cmb[(cmb["threshold"] == cc["threshold"]) & (cmb["hold"] == cc["hold"])
               & (cmb["stop"].isna() if stop is None else np.isclose(cmb["stop"], stop))]


def _period_row(eq: pd.Series, te: pd.Timestamp) -> dict:
    return {**stats(eq), "cagr_train": stats(eq[eq.index <= te])["cagr"],
            "cagr_test": stats(eq[eq.index > te])["cagr"]}


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    st2 = cfg["stage2"]
    te = pd.Timestamp(st2["train_end"])
    top = top_combos(cfg)
    sel = compare_selection(cfg)
    cand = candidate_trades(cfg, list(top["combo"].astype(int)) + list(sel.index))
    cal = _cal(cfg)
    closes = _close_panel(cfg, cand["ticker"].astype(str).unique(), cal)
    print(f"candidates: {len(cand):,} trades over {cand['ticker'].nunique()} tickers")

    curves, rows = {}, []
    for rank, (_, c) in enumerate(top.iterrows(), 1):
        label = _label(rank, c)
        eq, info = run_portfolio(cfg, combo_trades(cand, int(c["combo"]), c["market"]), closes=closes)
        curves[label] = eq
        rows.append({"combo": int(c["combo"]), "market": c["market"], **stats(eq), **info})
        print(f"{label}: cagr {rows[-1]['cagr']:.3f} mdd {rows[-1]['mdd']:.3f}")
    ks = kospi_curve(cfg, cal, st2["initial_capital"])
    curves["KOSPI"] = ks
    rows.append({"combo": -1, "market": "KOSPI buy&hold", **stats(ks)})
    pd.DataFrame(curves).to_csv(rd / "portfolio_equity.csv")
    pd.DataFrame(rows).to_csv(rd / "portfolio_summary.csv", index=False, encoding="utf-8-sig")

    if sel.empty:
        return
    mk = st2["compare_combo"]["market"]
    old = pd.read_parquet(rd / "trades.parquet", filters=[("combo", "in", list(map(int, sel.index)))])
    out = []
    for cid, c in sel.iterrows():
        # before the fix: per-ticker sequential list from the grid
        eq, info = run_portfolio(cfg, combo_trades(old, int(cid), mk))
        out.append({"filter": c["market_filter"], "method": "grid_list", "cooldown": 0,
                    **_period_row(eq, te), **info})
        for cd in st2.get("reentry_cooldowns", [0]):
            eq, info = run_portfolio(cfg, combo_trades(cand, int(cid), mk), cooldown=int(cd), closes=closes)
            out.append({"filter": c["market_filter"], "method": "portfolio", "cooldown": int(cd),
                        **_period_row(eq, te), **info})
            print(f"compare {c['market_filter']} cooldown {cd}: cagr {out[-1]['cagr']:.3f} mdd {out[-1]['mdd']:.3f}")
    out.append({"filter": "KOSPI buy&hold", "method": "index", "cooldown": 0, **_period_row(ks, te)})
    pd.DataFrame(out).to_csv(rd / "portfolio_filters.csv", index=False, encoding="utf-8-sig")

    # more slots on crash days: [normal slots, crash-day slots (0 = same)]
    lvl = st2.get("crash_slots_level")
    out = []
    for cid, c in sel.iterrows():
        tr = combo_trades(cand, int(cid), mk)
        for cd in st2.get("slot_cooldowns", [0]):
            for base, wide in st2.get("slot_variants", []):
                eq, info = run_portfolio(cfg, tr, cooldown=int(cd), closes=closes, slots=int(base),
                                         crash_slots=int(wide), crash_level=lvl if wide else None)
                out.append({"filter": c["market_filter"], "cooldown": int(cd), "slots": int(base),
                            "crash_slots": int(wide), **_period_row(eq, te), **info})
                print(f"slots {c['market_filter']} cd {cd} {base}/{wide}: "
                      f"cagr {out[-1]['cagr']:.3f} mdd {out[-1]['mdd']:.3f}")
    if out:
        pd.DataFrame(out).to_csv(rd / "portfolio_slots.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
