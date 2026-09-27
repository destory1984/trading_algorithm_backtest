"""Stage 5: portfolio simulation for the top combos.

Capital 100M KRW, at most 5 positions, equal weight (each new position gets
1/5 of the previous close's equity, limited by cash). When more signals than
free slots arrive on the same day, the lowest disparity wins. Positions that
exit on day d free their slot from day d+1.

The per-ticker trade list comes from the grid run, so "ignore signals while
holding" is applied per ticker even when the portfolio skipped the trade.

Usage:  python -m src.portfolio          # top combos from grid_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import load_calendar, load_config, load_index, load_prices, results_dir


def top_combos(cfg: dict, s: pd.DataFrame | None = None) -> pd.DataFrame:
    s = s if s is not None else pd.read_csv(results_dir() / "grid_summary.csv")
    s = s[s["trades"] >= cfg["stage2"]["min_trades"]]
    return s.sort_values("expectancy", ascending=False).head(cfg["stage2"]["top_n"])


def combo_trades(trades: pd.DataFrame, combo: int, market: str) -> pd.DataFrame:
    t = trades[trades["combo"] == combo]
    if market != "ALL":
        t = t[t["market"] == market]
    return t


def _close_panel(cfg: dict, tickers, cal: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        p = load_prices(cfg, t)
        c = p["close"].where(p["volume"] > 0)
        cols[t] = c[~c.index.duplicated(keep="last")]
    return pd.DataFrame(cols).reindex(cal).ffill()


def run_portfolio(cfg: dict, tr: pd.DataFrame) -> tuple[pd.Series, dict]:
    st = cfg["stage2"]
    cal = load_calendar(cfg)
    cal = cal[cal >= pd.Timestamp(cfg["data"]["start"])]
    closes = _close_panel(cfg, tr["ticker"].astype(str).unique(), cal)
    by_entry = {d: g.sort_values("disp") for d, g in tr.groupby("entry_date")}
    cash, slots = float(st["initial_capital"]), st["max_positions"]
    pos: dict[str, dict] = {}
    equity = np.empty(len(cal))
    prev_eq = cash
    taken = 0
    taken_rets: list[float] = []
    invested_days = 0
    for i, d in enumerate(cal):
        # entries at today's open
        if d in by_entry and len(pos) < slots:
            for _, t in by_entry[d].iterrows():
                if len(pos) >= slots:
                    break
                tk = str(t["ticker"])
                if tk in pos:
                    continue
                alloc = min(prev_eq / slots, cash)
                if alloc <= 0:
                    break
                cash -= alloc
                pos[tk] = {"alloc": alloc, "entry_px": float(t["entry_px"]),
                           "exit_date": t["exit_date"], "ret": float(t["ret"])}
                taken += 1
                taken_rets.append(float(t["ret"]))
        # exits during today
        for tk in [k for k, p in pos.items() if p["exit_date"] <= d]:
            p = pos.pop(tk)
            cash += p["alloc"] * (1 + p["ret"])
        # mark to market at close
        row = closes.iloc[i]
        val = 0.0
        for tk, p in pos.items():
            c = row.get(tk)
            val += p["alloc"] * (c / p["entry_px"] if pd.notna(c) else 1.0)
        prev_eq = cash + val
        equity[i] = prev_eq
        invested_days += bool(pos)
    eq = pd.Series(equity, index=cal, name="equity")
    return eq, {"trades_taken": taken, "trades_available": len(tr),
                "taken_mean_ret": float(np.mean(taken_rets)) if taken_rets else np.nan,
                "all_mean_ret": float(tr["ret"].mean()),
                "invested_share": invested_days / len(cal)}


def stats(eq: pd.Series) -> dict:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "mdd": (eq / eq.cummax() - 1).min(),
    }


def kospi_curve(cfg: dict, cal: pd.DatetimeIndex, capital: float) -> pd.Series:
    c = load_index(cfg, "KOSPI")["close"].reindex(cal).ffill()
    return c / c.iloc[0] * capital


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    trades = pd.read_parquet(rd / "trades.parquet")
    top = top_combos(cfg)
    curves, rows = {}, []
    for rank, (_, c) in enumerate(top.iterrows(), 1):
        label = f"#{rank} {c['market']} ≤{c['threshold']} "                 f"{'손절없음' if pd.isna(c['stop']) else format(c['stop'], '.0%')} {int(c['hold'])}일"
        tr = combo_trades(trades, int(c["combo"]), c["market"])
        eq, info = run_portfolio(cfg, tr)
        curves[label] = eq
        rows.append({"combo": int(c["combo"]), "market": c["market"], **stats(eq), **info})
        print(f"{label}: {rows[-1]}")
    cal = next(iter(curves.values())).index
    ks = kospi_curve(cfg, cal, cfg["stage2"]["initial_capital"])
    curves["KOSPI"] = ks
    rows.append({"combo": -1, "market": "KOSPI buy&hold", **stats(ks)})
    pd.DataFrame(curves).to_csv(rd / "portfolio_equity.csv")
    pd.DataFrame(rows).to_csv(rd / "portfolio_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
