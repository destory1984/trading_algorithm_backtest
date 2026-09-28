"""Stage 0: reproduce the original repo's numbers before touching the engine.

Original (toniker10/SPY-IBS-Mean-Reversion-Strategy):
    position  = raw_position.shift(1)
    strat_ret = position * Close.pct_change()
The day-t signal switches the position on at t+1, and t+1 earns close(t) ->
close(t+1): it is the same as buying at the signal-day close and selling at
the exit-signal-day close ("close" here). "open" buys and sells at the next
day's open instead.

This file is a plain loop, independent of krxbt, so stage 1 can compare the
engine against it.

Usage:  python -m src.replicate
Writes  results/replicate.csv, results/replicate_trades.csv, results/spy_1993.parquet (cache)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ibs, load_config, results_dir, stats, trade_stats


def download(cfg: dict) -> pd.DataFrame:
    """SPY daily bars from 1993, auto_adjust=True (dividend-adjusted OHLC), cached."""
    rp = cfg["replicate"]
    path = results_dir() / f"{rp['ticker'].lower()}_{rp['start'][:4]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf
    raw = yf.download(rp["ticker"], start=rp["start"], auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().rename(columns=str.lower)
    df.index = pd.DatetimeIndex(df.index.tz_localize(None) if df.index.tz else df.index, name="date")
    df.to_parquet(path)
    return df


def signal_pairs(ib: np.ndarray, entry: float, exit: float) -> list[tuple[int, int]]:
    """The original state machine: (entry-signal bar, exit-signal bar or -1 if still open)."""
    out, s = [], -1
    for i, v in enumerate(ib):
        if s < 0:
            if v < entry:
                s = i
        elif v > exit:
            out.append((s, i))
            s = -1
    if s >= 0:
        out.append((s, -1))
    return out


def simulate(df: pd.DataFrame, pairs: list[tuple[int, int]], mode: str,
             buy_cost: float = 1.0, sell_keep: float = 1.0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Trades, daily strategy returns, in-position flags.

    close: in at close[s], out at close[t]; position earns days s+1..t.
    open:  in at open[s+1], out at open[t+1]; day s+1 earns open->close,
           day t+1 earns close[t]->open[t+1].
    Costs: the entry day's return is divided by buy_cost, the exit day's is
    multiplied by sell_keep. A position still open at the end is marked at the
    last close with no selling cost (as the original does).
    """
    o, c = df["open"].to_numpy(np.float64), df["close"].to_numpy(np.float64)
    n = len(c)
    daily = np.zeros(n)
    inpos = np.zeros(n, bool)
    rows = []
    for s, t in pairs:
        if mode == "close":
            e, epx = s, c[s]
            if s + 1 >= n:  # signal on the last bar: never held (original skips it too)
                continue
            forced = t < 0
            x = n - 1 if forced else t
            xpx = c[x]
            for d in range(s + 1, x + 1):
                daily[d] = c[d] / c[d - 1] - 1
            first = s + 1
        elif mode == "open":
            e = s + 1
            if e >= n:
                continue
            epx = o[e]
            forced = t < 0 or t + 1 >= n
            x = n - 1 if forced else t + 1
            xpx = c[x] if forced else o[x]
            daily[e] = c[e] / o[e] - 1
            for d in range(e + 1, x + 1):
                daily[d] = c[d] / c[d - 1] - 1
            if not forced:
                daily[x] = o[x] / c[x - 1] - 1
            first = e
        else:
            raise ValueError(mode)
        inpos[first:x + 1] = True
        daily[first] = (1 + daily[first]) / buy_cost - 1
        keep = 1.0 if forced else sell_keep
        if not forced:
            daily[x] = (1 + daily[x]) * sell_keep - 1
        rows.append({"signal_date": df.index[s], "entry_date": df.index[e], "exit_date": df.index[x],
                     "entry_px": epx, "exit_px": xpx, "ret": xpx * keep / (epx * buy_cost) - 1, "forced": forced})
    idx = df.index
    return pd.DataFrame(rows), pd.Series(daily, idx), pd.Series(inpos, idx)


def run(df: pd.DataFrame, cfg: dict, mode: str, costs: bool) -> tuple[dict, pd.DataFrame]:
    cs = cfg["costs"]
    bc = (1 + cs["slippage"]) * (1 + cs["buy_fee"]) if costs else 1.0
    sk = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - cs["sell_tax"]) if costs else 1.0
    ib = ibs(df["high"], df["low"], df["close"], cfg["ibs"]["flat_value"])
    tr, daily, inpos = simulate(df, signal_pairs(ib, cfg["ibs"]["entry"], cfg["ibs"]["exit"]), mode, bc, sk)
    return {**stats(daily, inpos), **trade_stats(tr["ret"].to_numpy())}, tr


def main() -> None:
    cfg = load_config()
    full = download(cfg)
    rp = cfg["replicate"]
    rows, trades = [], []
    for span, df in (("original_end", full[full.index <= pd.Timestamp(rp["end_original"])]), ("latest", full)):
        period = f"{df.index[0].date()} → {df.index[-1].date()}"
        if span == "original_end":
            o = rp["original"]
            rows.append({"span": span, "period": period, "method": "원본 README", "costs": "0", "cagr": o["cagr"],
                         "mdd": o["mdd"], "sharpe": o["sharpe"], "trades": o["trades"], "win_rate": o["win_rate"]})
            rows.append({"span": span, "period": period, "method": "원본 README SPY 보유", "costs": "0",
                         "cagr": o["bh_cagr"], "mdd": o["bh_mdd"]})
        for mode in ("close", "open"):
            for costs in (False, True):
                m, tr = run(df, cfg, mode, costs)
                rows.append({"span": span, "period": period, "method": f"재현 {mode}", "costs": "설정" if costs else "0", **m})
                if span == "latest":
                    trades.append(tr.assign(mode=mode, costs=costs))
        bh = df["close"].pct_change().fillna(0)
        rows.append({"span": span, "period": period, "method": "재현 SPY 보유", "costs": "0", **stats(bh)})
    out = pd.DataFrame(rows)
    flat = int((full["high"] == full["low"]).sum())
    out.attrs["flat_days"] = flat
    rd = results_dir()
    out.to_csv(rd / "replicate.csv", index=False, encoding="utf-8-sig")
    pd.concat(trades, ignore_index=True).to_csv(rd / "replicate_trades.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["period"]).round(4).to_string())
    print(f"high == low days in {full.index[0].date()} → {full.index[-1].date()}: {flat}")


if __name__ == "__main__":
    main()
