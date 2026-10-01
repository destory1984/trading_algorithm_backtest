"""SPEC 4.4: trade and curve metrics per ticker and for the basket, whole period and the two halves.
Usage: python -m src.metrics   -> results/metrics_basket.csv, metrics_ticker.csv, metrics_extra.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C


def compare(strategy: pd.Series, hold: pd.Series) -> dict:
    w = C.matched_weight(strategy, hold)
    return {"cagr": C.cagr(strategy), "mdd": C.mdd(strategy), "sharpe": C.sharpe(strategy),
            "hold_cagr": C.cagr(hold), "hold_mdd": C.mdd(hold), "hold_sharpe": C.sharpe(hold),
            "w": w, "matched_cagr": C.cagr(hold * w), "diff": C.cagr(strategy) - C.cagr(hold * w)}


def trade_stats(t: pd.DataFrame) -> dict:
    r = t["ret"]
    win, loss = r[r > 0], r[r <= 0]
    return {"trades": len(r), "win": (r > 0).mean(), "mean": r.mean(), "median": r.median(),
            "payoff": win.mean() / -loss.mean() if len(loss) and loss.mean() < 0 else np.nan,
            "pf": win.sum() / -loss.sum() if loss.sum() < 0 else np.nan, "hold_days": t["hold_days"].mean()}


def breakeven(t: pd.DataFrame) -> float:
    """One-way cost at which the mean trade return is zero (bisection, sign kept)."""
    def f(c: float) -> float:
        return float((t["exit"] * (1 - c) / (t["entry"] * (1 + c)) - 1).mean())
    lo, hi = -0.1, 0.1
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    trades = pd.read_parquet(out / "trades.parquet")
    periods = {"전체": [cfg["data"]["start"], cfg["data"]["end"]], **cfg["data"]["periods"]}
    rows = []
    for v in [c for c in daily.columns if not c.endswith("|hold")]:
        s, h = daily[v].dropna(), daily[f"{v}|hold"].dropna()
        tv = trades[trades["variant"] == v]
        for pname, (lo, hi) in periods.items():
            sp, hp = C.cut(s, lo, hi), C.cut(h, lo, hi)
            if pname != "전체" and v.endswith("2004 시작"):
                continue
            tp = tv[(tv["exit_day"] >= lo) & (tv["exit_day"] <= hi)]
            rows.append({"variant": v, "period": pname, **compare(sp, hp), **trade_stats(tp)})
    mb = pd.DataFrame(rows)
    mb.to_csv(out / "metrics_basket.csv", index=False)

    sl, hs = pd.read_parquet(out / "sleeves.parquet"), pd.read_parquet(out / "hold_sleeves.parquet")
    held = pd.read_parquet(out / "held.parquet")
    t95, t0 = trades[trades["variant"] == "95"], trades[trades["variant"] == "95|비용 0"]
    rows = []
    for t in sl.columns:
        tt = t95[t95["ticker"] == t]
        live = hs[t].ne(0).cumsum().gt(0)          # from the first day the hold sleeve moves
        s, h = sl[t][live], hs[t][live]
        rows.append({"ticker": t, "name": cfg["tickers"][t], **trade_stats(tt), "cagr": C.cagr(s), "mdd": C.mdd(s),
                     "exposure": held[t][live].mean(), "hold_cagr": C.cagr(h), "hold_mdd": C.mdd(h),
                     "diff": compare(s, h)["diff"],
                     "mean_cost0": t0[t0["ticker"] == t]["ret"].mean(), "breakeven": breakeven(tt)})
    mt = pd.DataFrame(rows)
    mt.to_csv(out / "metrics_ticker.csv", index=False)
    extra = pd.DataFrame([{"exposure_basket": held.mean().mean(), "breakeven_pooled": breakeven(t95),
                           "kind_ma": (t95["kind"] == "ma").mean(), "kind_max": (t95["kind"] == "max").mean(),
                           "mean_ma": t95[t95["kind"] == "ma"]["ret"].mean(),
                           "mean_max": t95[t95["kind"] == "max"]["ret"].mean(),
                           "trades_per_year": len(t95) / ((sl.index[-1] - sl.index[0]).days / 365.25)}])
    extra.to_csv(out / "metrics_extra.csv", index=False)
    pd.set_option("display.width", 250)
    print(mb.round(4).to_string(index=False))
    print(mt.round(4).to_string(index=False))
    print(extra.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
