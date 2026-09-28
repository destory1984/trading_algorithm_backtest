"""Stage 1: the engine against the plain loop, on the same prices, signals from data.start (2008-01-01).

  csv   the original CSV as an engine frame (strategy.csv_frame) vs replicate.simulate on the same arrays
  us    us_data QQQ (engine frame) vs replicate.simulate on that frame's arrays
Original rule (RSI(2) < 15, close > SMA200, green2, open), original costs (5bp one way), both variants of the exit
quirk (original: the signal day counts; no_signal_day: it does not). Also the number of trades (same entry and exit
date) the two price files share, because us_data prices differ a little from the CSV.

Usage:  python -m src.port
Writes  results/engine_match.csv, results/quirk.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import cost_set, daily_returns, in_position, load_config, results_dir, stats, trade_stats, with_costs
from .replicate import VARIANTS, compare_trades, load_csv, loop_trades, sample, with_indicators
from .strategy import arrays, csv_frame, equity, frames, rule, run_rule


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    cap = cfg["portfolio"]["initial_capital"]
    o = cfg["original"]
    smp = sample(with_indicators(cfg, load_csv(cfg)))
    frs = {"csv": csv_frame(cfg, smp), "us": frames(cfg, ["QQQ"])["QQQ"]}
    ocfg, costs = with_costs(cfg, "orig"), cost_set(cfg, "orig")
    match, quirk, keep = [], [], {}
    for src, f in frs.items():
        a = arrays(f)
        cand, target = rule(a, cfg, o["rsi_thr"], "above", "green2")
        for var, skip in VARIANTS.items():
            eng = run_rule(a, ocfg, cand, target, "open", skip)
            ref = loop_trades(f.index, a["open"], a["close"], cand, target, "open", costs, skip)
            bad, diff = compare_trades(ref, eng)
            match.append({"source": src, "variant": var, "loop_trades": len(ref), "engine_trades": len(eng),
                          "mismatched": bad, "max_ret_diff": diff,
                          "first_signal": eng["signal_date"].iloc[0].date() if len(eng) else None})
            eq, _ = equity(cfg, f, eng, "QQQ")
            d = daily_returns(eq, cap)
            quirk.append({"source": src, "variant": var, "first": eq.index[0].date(), "last": eq.index[-1].date(),
                          **trade_stats(eng["ret"].to_numpy()), **stats(d, float(in_position(eq.index, eng).mean())),
                          "signal_day_exits": int(((eng["hold_days"] == 2) & (eng["reason"] == "target")).sum()),
                          "open_at_end": int((eng["reason"] == "forced").sum())})
            keep[(src, var)] = eng
    last = frs["csv"].index[-1]
    cs, us = keep[("csv", "original")], keep[("us", "original")]
    us = us[us["exit_date"] <= last]
    shared = len(set(zip(cs["entry_date"], cs["exit_date"])) & set(zip(us["entry_date"], us["exit_date"])))
    # constant ratio between the two (dividend-adjusted) price files on their overlapping dates
    common = frs["csv"].index.intersection(frs["us"].index)
    csv_us_ratio = float((frs["us"].loc[common, "close"] / frs["csv"].loc[common, "close"]).median())
    for r in quirk:
        r["shared_csv_us"] = shared if r["variant"] == "original" else np.nan
        r["csv_us_ratio"] = csv_us_ratio if r["variant"] == "original" else np.nan
    em, qk = pd.DataFrame(match), pd.DataFrame(quirk)
    em.to_csv(rd / "engine_match.csv", index=False, encoding="utf-8-sig")
    qk.to_csv(rd / "quirk.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(em.to_string())
    print(qk[["source", "variant", "trades", "win_rate", "profit_factor", "cagr", "mdd", "sharpe", "signal_day_exits",
              "shared_csv_us"]].round(4).to_string())


if __name__ == "__main__":
    main()
