"""SPEC 2: download the 19 country ETFs once into hist_data/country (outside the repo).
Refuses to run again if any file exists (the verdict uses the first download).
Usage: python -m src.fetch   -> ../../hist_data/country/<ticker>.parquet, results/fetch.md
"""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    d = C.data_dir(cfg)
    have = [t for t in cfg["tickers"] if (d / f"{t}.parquet").exists()]
    if have:
        sys.exit(f"already downloaded: {have}; SPEC 2 says download once. Delete by hand to re-download.")
    import yfinance as yf
    d.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(cfg["data"]["fetch_end"])
    rows, size = [], 0
    for t, name in cfg["tickers"].items():
        h = yf.Ticker(t).history(start=cfg["data"]["fetch_start"], end=cfg["data"]["fetch_end"],
                                 auto_adjust=False, actions=False)
        if h.empty:
            rows.append({"종목": t, "나라": name, "행 수": 0, "첫날": "없음", "마지막 날": ""})
            continue
        h.index = pd.DatetimeIndex(h.index.tz_localize(None).normalize(), name="date")
        h = h.rename(columns=lambda c: c.lower().replace(" ", "_"))
        h = h[["open", "high", "low", "close", "adj_close", "volume"]].astype("float64")
        assert h.index.max() < end, "got data on or after fetch_end"
        h.to_parquet(d / f"{t}.parquet")
        size += (d / f"{t}.parquet").stat().st_size
        rows.append({"종목": t, "나라": name, "행 수": len(h), "첫날": h.index.min().date(),
                     "마지막 날": h.index.max().date()})
    note = ["# 받은 기록", "",
            f"- 받은 날(한국 시간): {dt.datetime.now():%Y-%m-%d %H:%M}",
            f"- 출처: 야후, yfinance {yf.__version__}, history(start={cfg['data']['fetch_start']}, "
            f"end={cfg['data']['fetch_end']}, auto_adjust=False)",
            f"- 파일 {sum(r['행 수'] > 0 for r in rows)}개, 합계 {size / 1e6:.2f}MB, 위치: 저장소 밖 hist_data/{cfg['data']['sub']}/",
            "", C.md_table(pd.DataFrame(rows))]
    (C.results_dir() / "fetch.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))


if __name__ == "__main__":
    main()
