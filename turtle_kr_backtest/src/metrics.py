"""SPEC 4.5: metrics per variant and stretch, with KOSPI total return and the drawdown-matched KOSPI (w x KOSPI +
cash, rebalanced daily, w by bisection per stretch). Trades are counted in the stretch where they were sold.
Run after portfolio.py.
Usage: python -m src.metrics   -> results/metrics.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C


def daily(eq: pd.Series, capital: float) -> pd.Series:
    return eq.pct_change().fillna(eq.iloc[0] / capital - 1)


def trade_stats(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0}
    win = t["pnl"] > 0
    pos_sum = t.loc[win, "pnl"].sum()
    return {"trades": len(t), "win_rate": float(win.mean()), "avg_ret": float(t["ret"].mean()),
            "profit_factor": float(pos_sum / -t.loc[~win, "pnl"].sum()) if (~win).any() else np.nan,
            "top10_share": float(t["pnl"].nlargest(10).clip(lower=0).sum() / pos_sum) if pos_sum > 0 else np.nan,
            "avg_units": float(t["units"].mean()), "same_day_exit": float(t["reason"].str.startswith("same_day").mean())}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    eq = pd.read_parquet(out / "equity.parquet")
    inv = pd.read_parquet(out / "invested.parquet")
    un = pd.read_parquet(out / "units.parquet")
    tr = pd.read_parquet(out / "trades.parquet")
    cap = cfg["turtle"]["capital"]
    k_all = C.kospi_tr(cfg, C.calendar(cfg)).reindex(eq.index)
    rows = []
    for k in eq.columns:
        d_all = daily(eq[k], cap)
        t_all = tr[tr["variant"] == k]
        for s in C.STRETCHES:
            m = C.stretch_mask(cfg, d_all.index, s)
            d, kk = d_all[m], k_all[m]
            first, last = d.index[0], d.index[-1]
            years = (last - first).days / 365.25
            t = t_all[(t_all["exit_day"] >= first) & (t_all["exit_day"] <= last)]
            w = C.risk_weight(kk, C.mdd(d))
            rows.append({"variant": k, "stretch": s, "first_day": first.date(), "last_day": last.date(),
                         "cagr": C.cagr(d), "mdd": C.mdd(d), "sharpe": C.sharpe(d),
                         "calmar": C.cagr(d) / abs(C.mdd(d)) if C.mdd(d) < 0 else np.nan,
                         "avg_invested": float(inv[k][m].mean()), "avg_units_held": float(un[k][m].mean()),
                         "underwater_days": C.longest_underwater(d), **trade_stats(t),
                         "trades_per_year": len(t) / years,
                         "kospi_cagr": C.cagr(kk), "kospi_mdd": C.mdd(kk), "risk_w": w, "risk_cagr": C.cagr(w * kk)})
    df = pd.DataFrame(rows)
    df["diff_risk"] = df["cagr"] - df["risk_cagr"]
    df.to_csv(out / "metrics.csv", index=False)
    print(df[["variant", "stretch", "cagr", "mdd", "sharpe", "avg_invested", "trades", "win_rate", "avg_ret",
              "profit_factor", "kospi_cagr", "risk_w", "risk_cagr", "diff_risk"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
