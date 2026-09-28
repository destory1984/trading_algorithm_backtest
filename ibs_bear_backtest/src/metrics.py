"""SPEC 4.2: us_shortterm SPEC 8 metrics and SPEC 9 baselines per ticker x version. Run after run.py.
Usage: python -m src.metrics   -> results/metrics.csv, daily.parquet (strategy and hold daily returns)
"""
from __future__ import annotations

import pandas as pd

from src import benchmark, equity
from src import common as C


def main() -> None:
    cfg = C.load_config()
    _, frames, _ = C.load_frames(cfg)
    out = C.results_dir()
    judged = set(cfg["tickers"]["judged"])
    rows, daily = [], {}
    for t, f in frames.items():
        dates, close = C.curve_inputs(f, cfg)
        hold = benchmark.hold_returns(dates, close)
        daily[f"{t}|hold"] = hold
        for v in C.VERSIONS:
            tr = pd.read_parquet(out / f"trades_{t}_{v}.parquet")
            m, d = equity.metrics(tr, dates, close, False, cfg["costs"])
            b = benchmark.compare(hold, m["cagr"], m["mdd"], m["exposure"])
            rows.append({"ticker": t, "group": "판정" if t in judged else "참고", "version": v,
                         "first_day": dates[0].date(), "last_day": dates[-1].date(), **m, **b})
            daily[f"{t}|{v}"] = d
    met = pd.DataFrame(rows)
    assert met["curve_check"].all()
    met.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")
    print(met[["ticker", "version", "trades", "cagr", "risk_cagr", "diff_risk", "exposure", "breakeven_bp"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
