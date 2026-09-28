"""US daily prices from Yahoo (yfinance) into a separate data folder.

Universe = current Nasdaq-100 + Dow 30 members (Wikipedia tables) + the
tickers listed one per line in <data dir>/watchlist.txt (kept out of git).
Only today's members are known, so stocks that left the indices or were
delisted are missing: results on this data are survivorship biased.

Files (same layout as the Korean data):
  prices/<TICKER>.parquet   open high low close volume adj_close
                            (OHLC split-adjusted, adj_close also dividend-adjusted)
  index_NDX.parquet, index_DJI.parquet, index_SPX.parquet
  universe.parquet          ticker, name, groups, snapshot

Usage:
  python -m krxbt.fetch_us --dir ../us_data            # everything
  python -m krxbt.fetch_us --dir ../us_data --tickers AAPL MSFT
Env US_DATA_DIR replaces --dir.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

START = "2007-01-01"
INDICES = {"NDX": "^NDX", "DJI": "^DJI", "SPX": "^GSPC"}
WIKI = {
    "ndx": ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "Ticker"),
    "dow": ("https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies", "Symbol"),
}
UA = {"User-Agent": "krx-backtest-core (research script)"}


def _members(url: str, col: str) -> pd.DataFrame:
    html = requests.get(url, headers=UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        if col in t.columns and len(t) >= 25:
            name = "Company" if "Company" in t.columns else t.columns[0]
            return pd.DataFrame({"ticker": t[col].astype(str).str.strip(), "name": t[name].astype(str)})
    raise RuntimeError(f"no member table with column {col!r} at {url}")


def _quote_type(symbol: str) -> str:
    """EQUITY / ETF / ... from Yahoo, "" when unknown."""
    import yfinance as yf
    try:
        return str(yf.Ticker(symbol).info.get("quoteType") or "")
    except Exception:
        return ""


def build_universe(ddir: Path) -> pd.DataFrame:
    rows = []
    for group, (url, col) in WIKI.items():
        m = _members(url, col)
        m["group"] = group
        rows.append(m)
        print(f"universe {group}: {len(m)} tickers")
    wl = ddir / "watchlist.txt"
    if wl.exists():
        tick = [x.split("#")[0].strip().upper() for x in wl.read_text(encoding="utf-8").splitlines()]
        tick = [x for x in tick if x]
        rows.append(pd.DataFrame({"ticker": tick, "name": "", "group": "watch"}))
        print(f"universe watch: {len(tick)} tickers")
    df = pd.concat(rows, ignore_index=True)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo)
    uni = df.groupby("ticker").agg(name=("name", lambda s: max(s, key=len)),
                                   groups=("group", lambda s: ",".join(sorted(set(s))))).reset_index()
    uni["snapshot"] = pd.Timestamp.today().normalize()
    uni["type"] = "EQUITY"  # index members are stocks; ask Yahoo only for the watchlist
    for i in uni.index[uni["groups"] == "watch"]:
        uni.loc[i, "type"] = _quote_type(uni.loc[i, "ticker"])
        time.sleep(1.0)
    uni.to_parquet(ddir / "universe.parquet", index=False)
    print(f"universe: {len(uni)} tickers")
    return uni


def _download(symbol: str, start=START) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=False)
    if h.empty:
        return h
    h.index = pd.DatetimeIndex(h.index.tz_localize(None).normalize(), name="date")
    h = h.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    return h[["open", "high", "low", "close", "volume", "adj_close"]].astype("float64")


def _retry(fn, tries=3, delay=5.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def fetch(ddir: Path, tickers: list[str], delay: float) -> None:
    pdir = ddir / "prices"
    pdir.mkdir(parents=True, exist_ok=True)
    meta_path = pdir / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ok = empty = fail = 0
    for i, t in enumerate(tickers, 1):
        try:
            df = _retry(lambda: _download(t))
            if df.empty:
                empty += 1
                print(f"  {t}: no data")
            else:
                df.to_parquet(pdir / f"{t}.parquet")
                meta[t] = {"first": str(df.index[0].date()), "last": str(df.index[-1].date()), "rows": len(df)}
                ok += 1
        except Exception as e:  # one bad ticker should not stop the run
            fail += 1
            print(f"  {t} failed: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(delay)
        if i % 25 == 0:
            meta_path.write_text(json.dumps(meta))
            print(f"  prices {i}/{len(tickers)} ok={ok} empty={empty} fail={fail}")
    meta_path.write_text(json.dumps(meta))
    print(f"prices done: ok={ok} empty={empty} fail={fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../us_data")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between Yahoo requests")
    a = ap.parse_args()
    ddir = Path(os.environ.get("US_DATA_DIR") or a.dir).resolve()
    ddir.mkdir(parents=True, exist_ok=True)
    for name, sym in INDICES.items():
        df = _retry(lambda: _download(sym))
        df.drop(columns="adj_close").to_parquet(ddir / f"index_{name}.parquet")
        print(f"index {name}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}")
        time.sleep(a.delay)
    tickers = a.tickers or sorted(build_universe(ddir)["ticker"])
    fetch(ddir, tickers, a.delay)


if __name__ == "__main__":
    main()
