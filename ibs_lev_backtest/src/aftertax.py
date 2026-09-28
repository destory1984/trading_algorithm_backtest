"""Stage 3: after-tax results for a Korean resident (tax.py rules) at tax.capitals_krw.

Strategies: the stage-2 account; year Y's tax is taken on the first trading day of May of Y+1,
the final year's (2026, no May after it inside the sample) on the last day, 2026-07-31.
Interest on idle cash is not taxed. Cash may go negative when the tax falls due while fully invested.
Holds (the ETF, the 1x ETF, the risk-matched 1x): sold on the last day, taxed once that day.
The k1x curve rebalances daily in reality; here it is taxed once like a hold (a simplification).
The fixed KRW deduction is why the capital size matters; pre-tax results do not change with it.

Usage:  python -m src.aftertax      (after src.backtest)
Writes  results/aftertax.csv
"""
from __future__ import annotations

import pandas as pd

from .backtest import benchmarks, prepared
from .common import daily_returns, load_config, results_dir, stats
from .sizing import account
from .tax import hold_after_tax


def main() -> None:
    cfg = load_config()
    tc = cfg["tax"]
    rd = results_dir()
    g = pd.read_csv(rd / "lev.csv", index_col="combo")
    rows = []
    for cid, c, ctx in prepared(cfg):
        for krw in tc["capitals_krw"]:
            cap = krw / tc["fx"]
            eq, info = account(ctx["cal"], ctx["tr"], ctx["f"]["close"], cap, ctx["w"], ctx["fac"], tax=tc)
            s = stats(daily_returns(eq, cap))
            b = benchmarks(cfg, c, ctx, cap, g.at[cid, "mdd"])
            row = {"combo": cid, **c.to_dict(), "capital_krw": krw, "pre_cagr": g.at[cid, "cagr"],
                   "post_cagr": s["cagr"], "post_sharpe": s["sharpe"], "post_mdd": s["mdd"],
                   "tax_total_krw": sum(info["taxes"].values()) * tc["fx"],
                   "years_taxed": sum(v > 0 for v in info["taxes"].values()), "min_cash": info["min_cash"],
                   "min_cash_date": info["min_cash_date"].date() if info["min_cash_date"] is not None else None,
                   "min_cash_equity": info["min_cash_equity"]}
            for n in ("hold", "u1x", "k1x"):
                row[f"{n}_post_cagr"] = stats(daily_returns(hold_after_tax(b[n], cap, tc), cap))["cagr"]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(rd / "aftertax.csv", index=False, encoding="utf-8-sig")
    base = out[(out["capital_krw"] == tc["base_capital_krw"]) & (out["costs"] == "with") & (out["cash"] == "irx")]
    print(base[["ticker", "exec", "sizing", "pre_cagr", "post_cagr", "tax_total_krw", "u1x_post_cagr", "k1x_post_cagr"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
