"""Stage 5: the RSI(2) candidate (walk-forward pick for the last year, open-execution variant, because the IBS rule is
open-execution) next to work order 1's original IBS rule (IBS < 0.20 in, IBS > 0.80 out, ibs.exec execution; work order 1
did not pass its criteria), on each grid ticker, from data.start, us costs, one slot. The candidate's own ticker is
flagged; the spec compares on that ticker, the other three are context.

  overlap  entry days of both / entry days of RSI(2), and / entry days of IBS
  corr     correlation of the two daily return series (each strategy alone, full capital)
  or       enter when either signals; each trade exits by the rule that opened it; both on the same day -> the one
           that exits earlier (tie: RSI(2))
  and      enter only when both signal on the same day; exit at the earlier of the two rules' exits
  half     half the capital in each strategy, run separately, added up
OR and AND come from each rule's independent trades (engine independent=True: one trade per signal day) run through
run_portfolio(slots=1), which ignores signals while a position is open.
Exposure here = share of days ending with a position open (run_portfolio's invested_share); half = the mean of both.

Usage:  python -m src.mix      (after src.walkforward)
Writes  results/mix.csv, results/mix_daily.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import daily_returns, load_config, results_dir, stats, with_costs
from .strategy import arrays, equity, frames, ibs_rule, rule, run_rule

NAMES = ("rsi2", "ibs", "or", "and", "half")


def or_trades(ind_r: pd.DataFrame, ind_i: pd.DataFrame) -> pd.DataFrame:
    x = pd.concat([ind_r.assign(src="rsi2"), ind_i.assign(src="ibs")], ignore_index=True)
    x["rank"] = x["exit_date"].astype("int64") * 2 + (x["src"] == "ibs").astype("int64")
    return x


def and_trades(ind_r: pd.DataFrame, ind_i: pd.DataFrame) -> pd.DataFrame:
    r, i = ind_r.set_index("signal_date"), ind_i.set_index("signal_date")
    common = r.index.intersection(i.index)
    rr, ii = r.loc[common].copy(), i.loc[common]
    use_i = (ii["exit_date"] < rr["exit_date"]).to_numpy()
    rr.loc[use_i] = ii.loc[use_i]
    rr["src"] = np.where(use_i, "ibs", "rsi2")
    return rr.reset_index().sort_values("entry_date", ignore_index=True)


def main() -> None:
    cfg = load_config()
    cap = cfg["portfolio"]["initial_capital"]
    ibs_exec = cfg["ibs"]["exec"]
    rd = results_dir()
    ucfg = with_costs(cfg, "us")
    cands = pd.read_csv(rd / "candidates.csv")
    sel = cands[cands["variant"] == "open"]
    if len(sel) != 1:
        raise RuntimeError("no open-variant candidate in candidates.csv; run python -m src.walkforward")
    c = sel.iloc[0]
    rows, daily = [], {}
    for t, f in frames(cfg, cfg["grid"]["tickers"]).items():
        a = arrays(f)
        sig = {"rsi2": rule(a, cfg, c["rsi_thr"], c["trend"], c["exit"]), "ibs": ibs_rule(a, cfg)}
        exec_mode = {"rsi2": c["exec"], "ibs": ibs_exec}
        seq = {k: run_rule(a, ucfg, *v, exec_mode[k]) for k, v in sig.items()}
        ind = {k: run_rule(a, ucfg, *v, exec_mode[k], independent=True) for k, v in sig.items()}
        runs = {k: equity(cfg, f, seq[k], t) for k in seq}
        runs["or"] = equity(cfg, f, or_trades(ind["rsi2"], ind["ibs"]), t, rank_by="rank")
        runs["and"] = equity(cfg, f, and_trades(ind["rsi2"], ind["ibs"]), t)
        hr, hi = equity(cfg, f, seq["rsi2"], t, cap=cap / 2), equity(cfg, f, seq["ibs"], t, cap=cap / 2)
        eqs = {k: v[0] for k, v in runs.items()} | {"half": hr[0] + hi[0]}
        d = {k: daily_returns(e, cap) for k, e in eqs.items()}
        expo = {k: v[1]["invested_share"] for k, v in runs.items()} | {"half": (hr[1]["invested_share"] + hi[1]["invested_share"]) / 2}
        ntr = {k: v[1]["trades_taken"] for k, v in runs.items()} | {"half": hr[1]["trades_taken"] + hi[1]["trades_taken"]}
        er, ei = set(seq["rsi2"]["entry_date"]), set(seq["ibs"]["entry_date"])
        both = len(er & ei)
        corr = float(d["rsi2"].corr(d["ibs"]))
        for k in NAMES:
            rows.append({"ticker": t, "candidate": t == c["ticker"], "name": k, **stats(d[k], expo[k]), "trades": ntr[k],
                         "corr": corr, "entries_rsi2": len(er), "entries_ibs": len(ei), "entries_both": both,
                         "overlap_rsi2": both / len(er) if er else np.nan, "overlap_ibs": both / len(ei) if ei else np.nan})
            daily[f"{t}_{k}"] = d[k]
    mx = pd.DataFrame(rows)
    mx.to_csv(rd / "mix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily).to_parquet(rd / "mix_daily.parquet")
    pd.set_option("display.width", 200)
    print(f"RSI(2) candidate (open variant): {c['label']}")
    print(mx[["ticker", "candidate", "name", "cagr", "mdd", "sharpe", "exposure", "trades", "corr", "overlap_rsi2",
              "overlap_ibs"]].round(3).to_string())


if __name__ == "__main__":
    main()
