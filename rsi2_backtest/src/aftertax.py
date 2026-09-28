"""Stage 6 (part 2): after-tax results for a Korean resident (tax.py rules) at tax.capitals_krw.

  wf         each walk-forward variant's taken trades (walkforward_trades.parquet) as one account from
             walkforward.first_test_year; its benchmark is the stitched same-ticker hold (walkforward_equity.csv,
             scaled to the capital), sold and taxed once on the last day (simplification: switching tickers between
             years is treated as costless and untaxed)
  candidate  each variant's candidate combo over the grid window (data.start -> data.end); benchmark = holding its
             ticker with us costs, taxed once on the last day
Year Y's tax is taken on the first trading day of May of Y+1, the final year's on the last day. No interest on idle
cash. Pre-tax results do not depend on the capital; the fixed KRW deduction is why the capital size matters.

Usage:  python -m src.aftertax      (after src.walkforward)
Writes  results/aftertax.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .account import account
from .benchmarks import hold_curve
from .common import daily_returns, load_config, results_dir, stats, with_costs
from .strategy import calendar, frames
from .tax import hold_after_tax
from .walkforward import WF_VARIANTS


def row(kind: str, var: str, lab: str, krw: float, cap: float, pre: pd.Series, post: pd.Series, info: dict,
        hold: pd.Series, tc: dict) -> dict:
    s = stats(daily_returns(post, cap))
    return {"kind": kind, "variant": var, "label": lab, "capital_krw": krw,
            "pre_cagr": stats(daily_returns(pre, cap))["cagr"], "post_cagr": s["cagr"], "post_sharpe": s["sharpe"],
            "post_mdd": s["mdd"], "tax_total_krw": sum(info["taxes"].values()) * tc["fx"],
            "years_taxed": sum(v > 0 for v in info["taxes"].values()), "min_cash": info["min_cash"],
            "min_cash_date": info["min_cash_date"].date(), "min_cash_share": info["min_cash"] / info["min_cash_equity"],
            "hold_pre_cagr": stats(daily_returns(hold, cap))["cagr"],
            "hold_post_cagr": stats(daily_returns(hold_after_tax(hold, cap, tc), cap))["cagr"]}


def main() -> None:
    cfg = load_config()
    tc = cfg["tax"]
    rd = results_dir()
    base = cfg["portfolio"]["initial_capital"]
    wft = pd.read_parquet(rd / "walkforward_trades.parquet")
    weq = pd.read_csv(rd / "walkforward_equity.csv", index_col=0, parse_dates=True)
    cands = pd.read_csv(rd / "candidates.csv")
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"]
    fr = frames(cfg, cfg["grid"]["tickers"])
    closes = pd.DataFrame({t: f["close"] for t, f in fr.items()})
    bc, sk = cost_factors(with_costs(cfg, "us"))
    rows = []
    for var in WF_VARIANTS:
        taken = wft[wft["variant"] == var].reset_index(drop=True)
        cal = weq.index
        c = cands[cands["variant"] == var].iloc[0]
        x = tr[tr["combo"] == c["combo"]].assign(ticker=c["ticker"]).reset_index(drop=True)
        f = fr[c["ticker"]]
        cal2 = calendar(cfg, f)
        for krw in tc["capitals_krw"]:
            cap = krw / tc["fx"]
            one_w, one_f = np.ones(len(taken)), np.ones(len(cal))
            pre, _ = account(cal, taken, closes, cap, one_w, one_f)
            post, info = account(cal, taken, closes, cap, one_w, one_f, tax=tc)
            rows.append(row("wf", var, "걸어가며 검증", krw, cap, pre, post, info, weq[f"hold_{var}"] * cap / base, tc))
            w2, f2 = np.ones(len(x)), np.ones(len(cal2))
            pre2, info2pre = account(cal2, x, closes, cap, w2, f2)
            assert info2pre["trades_taken"] == info2pre["trades_available"]
            post2, info2 = account(cal2, x, closes, cap, w2, f2, tax=tc)
            assert info2["trades_taken"] == info2["trades_available"]
            rows.append(row("candidate", var, c["label"], krw, cap, pre2, post2, info2,
                            hold_curve(f["close"], cal2, cap, bc, sk), tc))
    out = pd.DataFrame(rows)
    out.to_csv(rd / "aftertax.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(out[["kind", "variant", "capital_krw", "pre_cagr", "post_cagr", "tax_total_krw", "hold_pre_cagr",
               "hold_post_cagr", "min_cash"]].round(4).to_string())


if __name__ == "__main__":
    main()
