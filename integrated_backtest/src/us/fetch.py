"""SPEC 2: download SCZ, AGG, BIL (listing -> 2026-09-25) and QQQ (1999-03-10 -> 2026-09-25) once from Yahoo into
hist_data (outside the repo). Skips a file that already exists (the verdict uses the first download).
Usage: python -m src.us.fetch   -> ../../hist_data/{SCZ,AGG,BIL,QQQ_1999}.parquet, results/us/fetch.md
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src import common as C


def download(symbol: str, start: str | None, end: str) -> pd.DataFrame:
    import yfinance as yf
    t = yf.Ticker(symbol)
    h = t.history(start=start, end=end, auto_adjust=False, actions=False) if start else \
        t.history(period="max", end=end, auto_adjust=False, actions=False)
    h.index = pd.DatetimeIndex(h.index.tz_localize(None).normalize(), name="date")
    h = h.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    return h[["open", "high", "low", "close", "volume", "adj_close"]].astype("float64")


def main() -> None:
    cfg = C.load_config()
    u = cfg["us"]
    d = C.env_dir(u["hist_dir"], "HIST_DATA_DIR")
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(t, t, None) for t in u["fetch"]] + [("QQQ", "QQQ_1999", u["qqq_from"])]
    notes = ["# 받은 기록", ""]
    for sym, name, start in jobs:
        p = d / f"{name}.parquet"
        if p.exists():
            notes.append(f"- {name}: 이미 있어 받지 않았다")
            continue
        h = download(sym, start, u["fetch_end"])
        assert len(h) and h.index.max() <= pd.Timestamp(u["end"]), f"{sym}: data past the seal"
        h.to_parquet(p)
        notes.append(f"- {name}: 야후 {sym}, {dt.datetime.now():%Y-%m-%d %H:%M} 받음, {len(h)}행, "
                     f"{h.index.min().date()} → {h.index.max().date()}, {p.stat().st_size / 1e6:.2f}MB")
    (C.results_dir("us") / "fetch.md").write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
