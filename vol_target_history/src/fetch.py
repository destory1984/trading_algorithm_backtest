"""SPEC 2: download ^GSPC daily bars once, before 2007-01-01 only, into hist_data (outside the repo).
Refuses to run again if the file exists (the verdict uses the first download).
Usage: python -m src.fetch   -> ../../hist_data/GSPC.parquet, results/fetch.md
"""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    path = C.hist_dir(cfg) / cfg["data"]["file"]
    if path.exists():
        sys.exit(f"{path.name} already exists; SPEC 2 says download once. Delete it by hand to re-download.")
    import yfinance as yf
    h = yf.Ticker(cfg["data"]["symbol"]).history(start=cfg["data"]["fetch_start"], end=cfg["data"]["fetch_end"],
                                                auto_adjust=False, actions=False)
    if h.empty:
        sys.exit("empty download")
    h.index = pd.DatetimeIndex(h.index.tz_localize(None).normalize(), name="date")
    h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype("float64")
    assert h.index.max() < pd.Timestamp(cfg["data"]["fetch_end"]), "got data on or after fetch_end"
    path.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(path)
    note = [f"# 받은 기록", "",
            f"- 받은 날(한국 시간): {dt.datetime.now():%Y-%m-%d %H:%M}",
            f"- 출처: 야후 {cfg['data']['symbol']}, yfinance {yf.__version__}, "
            f"history(start={cfg['data']['fetch_start']}, end={cfg['data']['fetch_end']}, auto_adjust=False)",
            f"- 행 수 {len(h)}, 첫날 {h.index.min().date()}, 마지막 날 {h.index.max().date()}",
            f"- 파일 크기 {path.stat().st_size / 1e6:.2f}MB, 위치: 저장소 밖 hist_data/{path.name}"]
    (C.results_dir() / "fetch.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))


if __name__ == "__main__":
    main()
