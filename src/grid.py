"""Stage 4: run every parameter combination on every ticker and summarise.

Simulation runs once per (market filter, threshold, stop, hold) = 288 combos
per ticker. The market axis (KOSPI / KOSDAQ / ALL) is a filter on the result,
so trades.parquet holds each trade once.

Usage:
  python -m src.grid                 # simulate + summarise
  python -m src.grid --limit 50      # first 50 tickers (smoke test)
  python -m src.grid --summary-only  # re-summarise existing trades.parquet
"""
from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .common import load_calendar, load_config, load_universe, price_dir, results_dir
from .indicators import index_features, ticker_frame
from .simulate import arrays, candidates, simulate


def combos(cfg: dict) -> pd.DataFrame:
    g = cfg["grid"]
    rows = list(itertools.product(g["market_filter"], g["thresholds"], g["stops"], g["holds"]))
    df = pd.DataFrame(rows, columns=["market_filter", "threshold", "stop", "hold"])
    df.index.name = "combo"
    return df


def universe_tickers(cfg: dict) -> pd.DataFrame:
    """One row per usable ticker with its latest market and delisting flag."""
    uni = load_universe(cfg)
    uni = uni[uni["excluded"].isna()].sort_values("snapshot")
    last = uni.groupby("ticker").tail(1).set_index("ticker")
    last["delisted"] = last["snapshot"] < uni["snapshot"].max()
    have = {p.stem for p in price_dir(cfg).glob("*.parquet")}
    return last[last.index.isin(have)][["market", "name", "delisted"]]


_W: dict = {}


def _init(cfg):
    _W["cfg"] = cfg
    _W["cal"] = load_calendar(cfg)
    _W["idx"] = index_features(cfg)
    _W["combos"] = combos(cfg)


def _run_ticker(args):
    ticker, market, delisted = args
    cfg, cmb = _W["cfg"], _W["combos"]
    f = ticker_frame(cfg, ticker, market, _W["cal"], _W["idx"], delisted)
    if f is None:
        return None, None
    a = arrays(f)
    out, counts = [], []
    for cid, c in cmb.iterrows():
        stop = None if pd.isna(c["stop"]) else float(c["stop"])
        tr = simulate(f, cfg, c["threshold"], stop, int(c["hold"]), c["market_filter"], a)
        if len(tr):
            tr.insert(0, "combo", np.int16(cid))
            out.append(tr)
    for mf in cfg["grid"]["market_filter"]:
        for thr in cfg["grid"]["thresholds"]:
            counts.append((mf, thr, int(candidates(a, thr, mf).sum())))
    trades = pd.concat(out, ignore_index=True) if out else None
    if trades is not None:
        trades.insert(1, "ticker", ticker)
        trades.insert(2, "market", market)
    cnt = pd.DataFrame(counts, columns=["market_filter", "threshold", "signals"])
    cnt["ticker"], cnt["market"] = ticker, market
    return trades, cnt


def run_all(cfg: dict, limit: int | None = None) -> pd.DataFrame:
    tk = universe_tickers(cfg)
    if limit:
        tk = tk.head(limit)
    jobs = list(zip(tk.index, tk["market"], tk["delisted"]))
    print(f"simulating {len(jobs)} tickers x {len(combos(cfg))} combos")
    t0 = time.time()
    trades, counts = [], []
    with ProcessPoolExecutor(cfg["grid"]["workers"], initializer=_init, initargs=(cfg,)) as ex:
        for i, (tr, cnt) in enumerate(ex.map(_run_ticker, jobs, chunksize=8), 1):
            if tr is not None:
                trades.append(tr)
            if cnt is not None:
                counts.append(cnt)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)} tickers, {time.time() - t0:.0f}s")
    df = pd.concat(trades, ignore_index=True)
    brk = df[df["data_break"]]
    brk.groupby("combo").size().rename("dropped").to_csv(results_dir() / "data_break_trades.csv")
    print(f"dropped {len(brk):,} trades held across a price-data break "
          f"({brk.drop_duplicates(['ticker', 'entry_date'])['ticker'].nunique()} tickers)")
    df = df[~df["data_break"]].drop(columns="data_break").reset_index(drop=True)
    df["ticker"] = df["ticker"].astype("category")
    df["market"] = df["market"].astype("category")
    df["reason"] = df["reason"].astype("category")
    for col in ("ret", "disp", "idx_disp", "entry_px", "exit_px"):
        df[col] = df[col].astype("float32")
    df["delisted"] = df["ticker"].map(tk["delisted"]).astype(bool)
    rd = results_dir()
    df.to_parquet(rd / "trades.parquet", index=False)
    pd.concat(counts, ignore_index=True).to_parquet(rd / "signal_counts.parquet", index=False)
    combos(cfg).to_csv(rd / "combos.csv")
    print(f"trades: {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


# ---------------------------------------------------------------- summary
def _metrics(g: pd.DataFrame) -> pd.DataFrame:
    r = g["ret"]
    out = pd.DataFrame({
        "trades": r.count(),
        "rebound_rate": g["rebound"].mean(),
        "win_rate": g["win"].mean(),
        "mean_ret": r.mean(),
        "median_ret": r.median(),
        "avg_win": g["pos"].mean(),
        "avg_loss": g["neg"].mean(),
        "avg_hold": g["hold_days"].mean(),
        "worst": r.min(),
    })
    out["payoff"] = out["avg_win"] / out["avg_loss"].abs()
    out["expectancy"] = out["win_rate"] * out["avg_win"] + (1 - out["win_rate"]) * out["avg_loss"].fillna(0)
    return out


def with_all_market(df: pd.DataFrame) -> pd.DataFrame:
    """Add a copy of every row labelled market=ALL (cheap view for groupby)."""
    return pd.concat([df.assign(market=df["market"].astype(str)),
                      df.assign(market="ALL")], ignore_index=True)


def summarize(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    rd = results_dir()
    if df is None:
        df = pd.read_parquet(rd / "trades.parquet")
    cmb = combos(cfg)
    df = df[["combo", "market", "ret", "rebound", "hold_days", "entry_date"]].copy()
    df["win"] = df["ret"] > 0
    df["pos"] = df["ret"].where(df["win"])
    df["neg"] = df["ret"].where(~df["win"])
    df["year"] = df["entry_date"].dt.year
    df = with_all_market(df)
    keys = ["combo", "market"]
    s = _metrics(df.groupby(keys))

    years = (load_calendar(cfg)[-1] - pd.Timestamp(cfg["data"]["start"])).days / 365.25
    cnt = pd.read_parquet(rd / "signal_counts.parquet")
    cnt = with_all_market(cnt).groupby(["market_filter", "threshold", "market"])["signals"].sum()

    ev_year = df.groupby(keys + ["year"])["ret"].mean().unstack("year")
    ev_year.columns = [f"ev_{y}" for y in ev_year.columns]
    crash = {}
    for label, (a, b) in cfg["stage2"]["crash_periods"].items():
        m = df["entry_date"].between(pd.Timestamp(a), pd.Timestamp(b))
        g = df[m].groupby(keys)["ret"]
        crash[f"ev_{label}"] = g.mean()
        crash[f"n_{label}"] = g.count()
    s = s.join(ev_year).join(pd.DataFrame(crash))

    s = s.reset_index().join(cmb, on="combo")
    s["signals"] = [cnt.get((rf, th, m), 0) for rf, th, m in zip(s["market_filter"], s["threshold"], s["market"])]
    s["signals_per_year"] = s["signals"] / years
    s["trades_per_year"] = s["trades"] / years
    front = ["combo", "market", "market_filter", "threshold", "stop", "hold", "signals", "signals_per_year",
             "trades", "trades_per_year", "rebound_rate", "win_rate", "mean_ret", "median_ret", "avg_win",
             "avg_loss", "payoff", "expectancy", "avg_hold", "worst"]
    s = s[front + [c for c in s.columns if c not in front]]
    s = s.sort_values("expectancy", ascending=False)
    s.to_csv(rd / "grid_summary.csv", index=False, encoding="utf-8-sig")
    print(f"grid_summary.csv: {len(s)} rows")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    df = None if a.summary_only else run_all(cfg, a.limit)
    s = summarize(cfg, df)
    cols = ["market", "market_filter", "threshold", "stop", "hold", "trades", "rebound_rate", "win_rate",
            "expectancy", "median_ret", "avg_hold", "worst"]
    pd.set_option("display.width", 200)
    print(s[s["trades"] >= cfg["stage2"]["min_trades"]].head(15)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
