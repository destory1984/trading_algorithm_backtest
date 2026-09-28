"""Run the 18 settings (strategy x execution version) on the 4 ETFs = 72 runs, once, with the rules as written.

Execution versions (SPEC 7.1):
  close  entry_on_signal_close = true,  target_exit_on_close = true   (target is masked on signal days)
  open   entry_on_signal_close = false, target_exit_on_close = false
  co     entry_on_signal_close = true,  target_exit_on_close = false  (S9: buy the close, sell the next open)
  bar    bar_sim.py (S10, S11, S12)

Writes results/screen/trades_<strategy>_<version>_<ticker>.parquet, metrics.csv, daily.parquet.
Usage: python -m src.screen.run
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine

from src import benchmark, common, equity
from src.screen import bar_sim
from src.screen.strategies import ALL

RULES = {"close": (True, True), "open": (False, False), "co": (True, False)}


def label(mod) -> str:
    return mod.__name__.rsplit(".", 1)[-1]


def settings() -> list[tuple]:
    return [(mod, v) for mod in ALL for v in mod.VERSIONS]


def entry_on_close(version: str) -> bool:
    return version in ("close", "co")


def trades_for(mod, version: str, f: pd.DataFrame, a: dict, cal: pd.DatetimeIndex, cfg: dict,
               mask_close: bool = True) -> pd.DataFrame:
    """One run. mask_close=False skips the close version's `target & ~cand` (only checks.py uses that)."""
    if mod.KIND == "bar":
        enter, px = mod.entries(f)
        return bar_sim.run(f, a, cfg, enter, px, mod.EXIT)
    cand, target = mod.signals(f, cal)
    cand = np.asarray(cand, np.bool_) & a["eligible"]
    target = np.asarray(target, np.bool_)
    if version == "close" and mask_close:
        target = target & np.logical_not(cand)
    c = common.with_rules(cfg, *RULES[version])
    return engine.run(a, c, cand, target, None, int(cfg["screen"]["hold_max"]))


def curve_inputs(f: pd.DataFrame, cfg: dict) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The screening dates (data.start to the seal) and the dividend-adjusted closes on them."""
    g = f[f.index >= pd.Timestamp(cfg["data"]["start"])]
    assert g["valid"].all(), "gap in an ETF's prices"
    return g.index, g["close"].to_numpy(np.float64)


def run_all(cfg: dict, frames: dict, cal: pd.DatetimeIndex) -> dict:
    """{(label, version, ticker): trades}."""
    out = {}
    for t, f in frames.items():
        a = engine.arrays(f)
        for mod, v in settings():
            out[(label(mod), v, t)] = trades_for(mod, v, f, a, cal, cfg)
    return out


def main() -> None:
    cfg = common.load_config()
    cal, frames = common.load_frames(cfg)
    out = common.results_dir()
    all_trades = run_all(cfg, frames, cal)
    mods = {label(m): m for m in ALL}
    rows, daily = [], {}
    for (name, v, t), tr in all_trades.items():
        tr.to_parquet(out / f"trades_{name}_{v}_{t}.parquet")
        dates, close = curve_inputs(frames[t], cfg)
        m, d = equity.metrics(tr, dates, close, entry_on_close(v), cfg["costs"])
        b = benchmark.compare(benchmark.hold_returns(dates, close), m["cagr"], m["mdd"], m["exposure"])
        mod = mods[name]
        rows.append({"strategy": mod.ID, "name": mod.NAME, "label": name, "version": v, "ticker": t,
                     "first_day": dates[0].date(), "last_day": dates[-1].date(), **m, **b})
        daily[f"{name}|{v}|{t}"] = d
    met = pd.DataFrame(rows)
    assert len(met) == 72, len(met)
    assert met["curve_check"].all()
    met.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")
    print(f"runs: {len(met)} (settings {met.groupby(['label', 'version']).ngroups} x tickers {met['ticker'].nunique()})")
    print(met[["strategy", "version", "ticker", "trades", "cagr", "risk_cagr", "breakeven_bp"]].to_string(index=False))


if __name__ == "__main__":
    main()
