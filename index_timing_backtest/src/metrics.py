"""SPEC 4.2 / 4.3: metrics per run and stretch, with the three baselines built from the same market's hold run.
Run after run.py.
Usage: python -m src.metrics   -> results/metrics.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def parse(k: str) -> dict:
    m, r, p, dv = k.split("|")
    return {"market": m, "rule": r, "param": p, "dividend": float(dv) if dv else None}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily = pd.read_parquet(out / "daily.parquet")
    weights = pd.read_parquet(out / "weights.parquet")
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    rows = []
    for k in daily.columns:
        info = parse(k)
        hold_k = f"{info['market']}|HOLD||{'' if info['dividend'] is None else info['dividend']}"
        d_all, h_all, w_all = daily[k].dropna(), daily[hold_k].dropna(), weights[k].dropna()
        t_all = tr.loc[tr["run"] == k, "date"]
        for s in C.STRETCHES:
            msk = C.stretch_mask(cfg, d_all.index, s)
            d, h, w = d_all[msk], h_all[msk], w_all[msk]
            years = (d.index[-1] - d.index[0]).days / 365.25
            n_tr = int(((t_all >= d.index[0]) & (t_all <= d.index[-1])).sum())
            wr = C.risk_weight(h, C.mdd(d))
            wa = float(w.mean())
            rows.append({**info, "run": k, "stretch": s, "first_day": d.index[0].date(), "last_day": d.index[-1].date(),
                         "cagr": C.cagr(d), "mdd": C.mdd(d), "sharpe": C.sharpe(d),
                         "calmar": C.cagr(d) / abs(C.mdd(d)) if C.mdd(d) < 0 else float("nan"),
                         "avg_weight": wa, "trades": n_tr, "trades_per_year": n_tr / years,
                         "underwater_days": C.longest_underwater(d),
                         "hold_cagr": C.cagr(h), "hold_mdd": C.mdd(h), "risk_w": wr, "risk_cagr": C.cagr(wr * h),
                         "avgw_cagr": C.cagr(wa * h), "avgw_mdd": C.mdd(wa * h)})
    df = pd.DataFrame(rows)
    df["diff_hold"] = df["cagr"] - df["hold_cagr"]
    df["diff_risk"] = df["cagr"] - df["risk_cagr"]
    df["diff_avgw"] = df["cagr"] - df["avgw_cagr"]
    df["mdd_ratio"] = df["mdd"] / df["hold_mdd"]
    df.to_csv(out / "metrics.csv", index=False)
    show = df[(df["stretch"] == "full") & df["dividend"].isin([None, cfg["markets"]["KR"]["dividend"]]) | df["dividend"].isna() & (df["stretch"] == "full")]
    print(show[["market", "rule", "param", "cagr", "mdd", "hold_cagr", "hold_mdd", "mdd_ratio", "risk_cagr", "diff_risk",
                "trades_per_year"]].drop_duplicates().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
