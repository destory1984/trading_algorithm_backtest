"""SPEC 4.7: the main rules at each slippage in `slippages` (one way); CAGR against the drawdown-matched KOSPI over
the full curve.
Usage: python -m src.costs   -> results/costs.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src import signals as SG
from src.metrics import daily
from src.portfolio import Panel, simulate


def main() -> None:
    cfg = C.load_config()
    p = SG.load(mmap=False)
    P = Panel(p)
    kospi = C.kospi_tr(cfg, P.cal)
    rows = []
    for r, s in cfg["rules"].items():
        k = C.vkey(r, s["entry"])
        for sl in cfg["slippages"]:
            res = simulate(cfg, P, s["exit"], p["cands"][k], slippage=sl)
            d = daily(res["equity"], cfg["turtle"]["capital"])
            kk = kospi.reindex(d.index)
            w = C.risk_weight(kk, C.mdd(d))
            rows.append({"rule": r, "slippage": sl, "cagr": C.cagr(d), "mdd": C.mdd(d), "risk_w": w,
                         "risk_cagr": C.cagr(w * kk), "diff_risk": C.cagr(d) - C.cagr(w * kk),
                         "trades": len(res["trades"])})
    df = pd.DataFrame(rows)
    df.to_csv(C.results_dir() / "costs.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
