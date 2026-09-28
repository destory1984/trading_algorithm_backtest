"""SPEC 4.3: T3 (main + neighbours) and the hold on ^GSPC 1928 -> 2006, under the main assumption and each
sensitivity; plus the SPY 2008 -> 2026 runs used by check 2 and the execution-price note (SPEC 2.3).
Usage: python -m src.run   -> results/daily.parquet, weights.parquet, trades.csv, spy_daily.parquet
Columns are "<rule>|<target>|<assumption>" (assumption = "본" or a sens name) and "SPY|<target>|<open|close>".
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src import common as C
from src import rules as R
from src import sim as S


def assumptions(cfg: dict) -> dict[str, dict]:
    return {"본": {"dividend": cfg["dividend"], "cash": cfg["cash"]}, **cfg["sens"]}


def runs(cfg: dict) -> list[tuple]:
    rc = cfg["rule"]
    return [("HOLD", None)] + [("T3", t) for t in [rc["main"], *rc["neighbors"]]]


def key(r, t, a) -> str:
    return f"{r}|{'' if t is None else t}|{a}"


def one(cfg: dict, r: str, t, a: dict, table: pd.DataFrame | None = None):
    raw = C.raw(cfg) if table is None else table
    close = C.total_return(raw["close"], a["dividend"])
    dec = R.decisions(close, r, t, cfg)
    px = pd.DataFrame({"close": close})
    return S.simulate(px, dec, pd.Timestamp(cfg["data"]["start"]), cfg["cost"], "close", a["cash"])


def spy_prices(cfg: dict) -> pd.DataFrame:
    rp = cfg["replicate"]
    e = os.environ.get("US_DATA_DIR")
    d = Path(e) if e else (C.ROOT / rp["us_dir"]).resolve()
    r = pd.read_parquet(d / "prices" / "SPY.parquet")
    r = C.seal(r[~r.index.duplicated(keep="last")], pd.Timestamp(rp["end"]))
    k = r["adj_close"] / r["close"]
    return pd.DataFrame({"open": r["open"] * k, "close": r["adj_close"]})


def spy_runs(cfg: dict) -> tuple[dict, dict]:
    px = spy_prices(cfg)
    daily, ntr = {}, {}
    for r, t in runs(cfg):
        dec = R.decisions(px["close"], r, t, cfg)
        for ex in ("open", "close"):
            d, _, tr = S.simulate(px, dec, pd.Timestamp(cfg["replicate"]["start"]), cfg["cost"], ex, 0.0)
            k = f"SPY|{'' if t is None else t}|{ex}" if r == "T3" else f"SPY|HOLD|{ex}"
            daily[k], ntr[k] = d, len(tr)
    return daily, ntr


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    daily, weights, trades = {}, {}, []
    for name, a in assumptions(cfg).items():
        for r, t in runs(cfg):
            d, w, tr = one(cfg, r, t, a)
            k = key(r, t, name)
            daily[k], weights[k] = d, w
            trades += [{"run": k, "date": x.date(), "weight": w[x]} for x in tr]
    pd.DataFrame(daily).to_parquet(out / "daily.parquet")
    pd.DataFrame(weights).to_parquet(out / "weights.parquet")
    pd.DataFrame(trades).to_csv(out / "trades.csv", index=False)
    sd, sn = spy_runs(cfg)
    pd.DataFrame(sd).to_parquet(out / "spy_daily.parquet")
    pd.Series(sn, name="trades").to_csv(out / "spy_trades.csv")
    print(len(daily), "runs;", len(sd), "SPY runs")


if __name__ == "__main__":
    main()
