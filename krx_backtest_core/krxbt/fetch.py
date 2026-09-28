"""Stage 1: collect the universe, index and per-ticker daily prices.

Universe
  With KRX login (env KRX_ID / KRX_PW), pykrx gives the listed tickers on the
  first trading day of every month since the start date, so delisted tickers
  are included (no survivorship bias).
  Without login, KRX refuses pykrx requests. We then use the current listing
  (FinanceDataReader) plus the delisted-company list from KRX KIND
  (kind.krx.co.kr, no login). The KIND list only has a 5-char company code;
  the common-share ticker is that code + "0".

Prices and indices
  Naver chart API (api.finance.naver.com/siseJson.naver) daily OHLCV. It takes
  a date range, returns split/bonus-issue adjusted prices and still serves
  delisted tickers. (FinanceDataReader uses a different Naver endpoint that
  stops at the latest 3000 rows, about 12 years.) Trading value is estimated as
  close * volume because this source has no value column.

Usage (from an algorithm repo; reads the `data` and `universe` parts of its config.yaml):
  python -m krxbt.fetch                 # everything
  python -m krxbt.fetch --limit 30      # first 30 tickers only (smoke test)
  python -m krxbt.fetch --tickers 005930 085370
  python -m krxbt.fetch --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import FinanceDataReader as fdr
import pandas as pd
import requests

from .config import data_dir, end_date, load_config, price_dir

COLS = ["open", "high", "low", "close", "volume"]
NAVER_URL = "https://api.finance.naver.com/siseJson.naver"
ROW_RE = re.compile(r'\["(\d{8})",\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)')


def has_krx_login() -> bool:
    return bool(os.environ.get("KRX_ID") and os.environ.get("KRX_PW"))


def _retry(fn, tries=3, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


# ---------------------------------------------------------------- index
def fetch_indices(cfg: dict) -> None:
    start, end = cfg["data"]["warmup_start"], end_date(cfg)
    for market in ("KOSPI", "KOSDAQ"):
        df = _retry(lambda: _download(market, start, end))[["open", "high", "low", "close"]]
        df.to_parquet(data_dir(cfg) / f"index_{market}.parquet")
        print(f"index {market}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}")
        time.sleep(cfg["data"]["request_delay"])


# ---------------------------------------------------------------- universe
def _excluded(ticker: str, name: str, cfg: dict) -> str | None:
    u = cfg["universe"]
    if u.get("exclude_preferred") and not ticker.endswith("0"):
        return "preferred"
    for pat in u.get("exclude_name_patterns", []):
        if name and re.search(pat, name):
            return pat
    return None


def _universe_from_krx(cfg: dict, cal: pd.DatetimeIndex, old: pd.DataFrame | None) -> pd.DataFrame:
    from pykrx import stock

    start = pd.Timestamp(cfg["data"]["start"])
    days = pd.Series(cal[cal >= start])
    snaps = days.groupby(days.dt.to_period("M")).first().tolist()
    done = set(old["snapshot"]) if old is not None and "source" in old and (old["source"] == "krx").all() else set()
    rows = [] if not done else [old]
    for d in snaps:
        if d in done and d != snaps[-1]:
            continue
        for m in cfg["data"]["markets"]:
            tick = _retry(lambda: stock.get_market_ticker_list(d.strftime("%Y%m%d"), market=m))
            if not tick:
                raise RuntimeError(f"empty ticker list for {m} {d.date()} (KRX login failed?)")
            rows.append(pd.DataFrame({"snapshot": d, "ticker": tick, "market": m}))
            time.sleep(cfg["data"]["request_delay"])
        print(f"  snapshot {d.date()} ok")
    uni = pd.concat(rows, ignore_index=True).drop_duplicates(["snapshot", "ticker"], keep="last")
    names = {} if old is None or "name" not in old else dict(zip(old["ticker"], old["name"]))
    for t in uni["ticker"].unique():
        if not names.get(t):
            try:
                names[t] = stock.get_market_ticker_name(t)
            except Exception:
                names[t] = ""
    uni["name"] = uni["ticker"].map(names)
    uni["source"] = "krx"
    return uni


def _universe_from_fdr(cfg: dict, cal: pd.DatetimeIndex) -> pd.DataFrame:
    lst = _retry(lambda: fdr.StockListing("KRX"))
    lst = lst[lst["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])].copy()
    lst["market"] = lst["Market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})
    lst = lst[lst["market"].isin(cfg["data"]["markets"])]
    return pd.DataFrame({
        "snapshot": cal[-1], "ticker": lst["Code"].values, "market": lst["market"].values,
        "name": lst["Name"].values, "source": "fdr_current",
    })


KIND_URL = "https://kind.krx.co.kr/investwarn/delcompany.do"
KIND_ROW = re.compile(
    r"icn_t_(\w+)\.gif.*?companysummary_open\('(\w+)'\).*?title='([^']*)'"
    r".*?<td class=\"txc\">([\d-]+)</td>\s*<td>(.*?)</td>", re.S)
KIND_MARKET = {"yu": "KOSPI", "ko": "KOSDAQ"}


def _delisted_from_kind(cfg: dict) -> pd.DataFrame:
    r = requests.post(KIND_URL, data={
        "method": "searchDelCompanySub", "currentPageSize": "5000", "pageIndex": "1",
        "orderMode": "2", "orderStat": "D", "marketType": "",
        "fromDate": cfg["data"]["start"], "toDate": end_date(cfg).strftime("%Y-%m-%d"),
    }, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    rows = []
    for tr in re.findall(r"<tr.*?</tr>", r.text, re.S):
        m = KIND_ROW.search(tr)
        if not m:
            continue
        icon, code, name, date, reason = m.groups()
        market = KIND_MARKET.get(icon)
        if market in cfg["data"]["markets"] and re.fullmatch(r"\d{4}[0-9A-Z]", code):
            rows.append({"snapshot": pd.Timestamp(date), "ticker": code + "0", "market": market,
                         "name": name.strip(), "delist_reason": re.sub(r"<[^>]+>", "", reason).strip()})
    df = pd.DataFrame(rows)
    df["source"] = "kind_delisted"
    return df


def build_universe(cfg: dict) -> pd.DataFrame:
    path = data_dir(cfg) / "universe.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    cal = pd.DatetimeIndex(pd.read_parquet(data_dir(cfg) / "index_KOSPI.parquet").index)
    if has_krx_login():
        print("universe: KRX login found, collecting monthly listings (includes delisted)")
        uni = _universe_from_krx(cfg, cal, old)
    else:
        print("universe: no KRX_ID/KRX_PW -> current listing + KIND delisted list")
        cur = _universe_from_fdr(cfg, cal)
        dl = _delisted_from_kind(cfg)
        dl = dl[~dl["ticker"].isin(cur["ticker"])]  # moved KOSDAQ->KOSPI: still listed
        dl = dl.sort_values("snapshot").drop_duplicates("ticker", keep="last")
        print(f"universe: {len(dl)} delisted tickers from KIND")
        uni = pd.concat([cur, dl], ignore_index=True)
    uni["excluded"] = [_excluded(t, n, cfg) for t, n in zip(uni["ticker"], uni["name"])]
    uni.to_parquet(path)
    last = uni.sort_values("snapshot").groupby("ticker").tail(1)
    print(f"universe: {last['ticker'].nunique()} tickers, "
          f"{last['excluded'].notna().sum()} excluded, by market:\n{last[last['excluded'].isna()]['market'].value_counts().to_string()}")
    return uni


# ---------------------------------------------------------------- prices
def _download(symbol: str, start, end) -> pd.DataFrame:
    """Daily adjusted OHLCV from Naver. symbol is a 6-char code or KOSPI/KOSDAQ."""
    r = requests.get(NAVER_URL, params={
        "symbol": symbol, "requestType": 1, "timeframe": "day",
        "startTime": pd.Timestamp(start).strftime("%Y%m%d"),
        "endTime": pd.Timestamp(end).strftime("%Y%m%d"),
    }, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    rows = ROW_RE.findall(r.text)
    df = pd.DataFrame(rows, columns=["date"] + COLS)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype("float64")


def fetch_prices(cfg: dict, tickers: list[str], delisted: set[str]) -> None:
    pdir = price_dir(cfg)
    meta_path = pdir / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    start, end = pd.Timestamp(cfg["data"]["warmup_start"]), end_date(cfg)
    cal_last = pd.DatetimeIndex(pd.read_parquet(data_dir(cfg) / "index_KOSPI.parquet").index)[-1]
    delay = max(0.3, cfg["data"]["request_delay"])
    n_new = n_upd = n_skip = n_fail = 0
    for i, t in enumerate(tickers, 1):
        path = pdir / f"{t}.parquet"
        info = meta.get(t, {})
        fetched_until = pd.Timestamp(info["fetched_until"]) if "fetched_until" in info else None
        try:
            if info.get("from") != str(start.date()):
                fetched_until = None  # start date moved: cached history is too short
                if path.exists():
                    path.unlink()
            if path.exists() and fetched_until is not None and (
                fetched_until >= cal_last or (t in delisted and info.get("delisted"))
            ):
                n_skip += 1
                continue
            if path.exists():
                old = pd.read_parquet(path)
                new = _download(t, old.index[-1] - pd.Timedelta(days=14), end)
                common = old.index.intersection(new.index)
                o, n = old.loc[common], new.loc[common]
                live = (o["volume"] > 0) & (n["volume"] > 0)
                # Adjusted prices shift backwards after a split: refetch everything then.
                if live.any() and ((o["close"][live] - n["close"][live]).abs() / o["close"][live]).max() > 0.005:
                    time.sleep(delay)
                    df = _download(t, start, end)
                else:
                    df = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
                n_upd += 1
            else:
                df = _download(t, start, end)
                n_new += 1
            df.to_parquet(path)
            meta[t] = {"fetched_until": str(cal_last.date()), "from": str(start.date()), "delisted": t in delisted,
                       "rows": int(len(df))}
        except Exception as e:  # keep going; one bad ticker should not stop the run
            n_fail += 1
            print(f"  {t} failed: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(delay)
        if i % 100 == 0:
            meta_path.write_text(json.dumps(meta))
            print(f"  prices {i}/{len(tickers)} new={n_new} upd={n_upd} skip={n_skip} fail={n_fail}")
    meta_path.write_text(json.dumps(meta))
    print(f"prices done: new={n_new} upd={n_upd} skip={n_skip} fail={n_fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--skip-universe", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    fetch_indices(cfg)
    upath = data_dir(cfg) / "universe.parquet"
    uni = pd.read_parquet(upath) if a.skip_universe and upath.exists() else build_universe(cfg)
    ok = uni[uni["excluded"].isna()]
    last_snap = uni["snapshot"].max()
    last_seen = ok.groupby("ticker")["snapshot"].max()
    delisted = set(last_seen[last_seen < last_snap].index)
    tickers = a.tickers or sorted(last_seen.index)
    if a.limit:
        tickers = tickers[: a.limit]
    print(f"fetching prices for {len(tickers)} tickers ({len(delisted)} delisted in universe)")
    fetch_prices(cfg, tickers, delisted)


if __name__ == "__main__":
    main()
