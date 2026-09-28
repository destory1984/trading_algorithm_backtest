"""SPEC2 4.5: yearly results and the market regime at the signal, full stretch 2008 -> deep_end.

Yearly: the rule's return, the hold's, and the risk-matched hold's with the full stretch's weight w (one w for
all years, so the rows line up with the full-stretch comparison).
Regime: SPX close above / below its 200-day SMA on the signal day; mean trade return (with costs), count, t.
Reported only; never used as a filter.
Usage: python -m src.deep.periods   -> results/deep/years.csv, regime.csv
"""
from __future__ import annotations

import krxbt.us
import numpy as np
import pandas as pd

from src import common
from src.deep import base
from src.indicators import sma


def main() -> None:
    cfg = common.load_config()
    _, st = base.all_stretches(cfg)
    out = common.results_dir("deep")
    with common.sealed_loaders(cfg, end=base.deep_end(cfg)):
        spx = krxbt.us.load_index(cfg, "SPX")["close"]
    up = spx > sma(spx, int(cfg["deep"]["regime_ma"]))
    yrs, reg = [], []
    for t in cfg["screen"]["tickers"]:
        s = st[("full", t)]
        w = s["m"]["risk_w"]
        g = pd.DataFrame({"s": s["daily"], "h": s["hold"]})
        for y, x in g.groupby(g.index.year):
            rs, rh = float((1 + x["s"]).prod() - 1), float((1 + x["h"]).prod() - 1)
            rw = float((1 + w * x["h"]).prod() - 1)
            yrs.append({"ticker": t, "year": y, "strategy": rs, "hold": rh, "risk_hold": rw, "diff_risk": rs - rw,
                        "risk_w": w})
        tr = s["trades"]
        u = up.reindex(pd.DatetimeIndex(tr["signal_date"])).to_numpy(bool)
        for name, k in (("SPX > 200일선", u), ("SPX < 200일선", ~u)):
            r = tr["ret"].to_numpy()[k]
            sd = r.std(ddof=1) if len(r) > 1 else np.nan
            reg.append({"ticker": t, "regime": name, "trades": len(r), "mean_ret": r.mean() if len(r) else np.nan,
                        "t": r.mean() / (sd / np.sqrt(len(r))) if len(r) > 1 and sd > 0 else np.nan,
                        "win_rate": (r > 0).mean() if len(r) else np.nan})
    pd.DataFrame(yrs).to_csv(out / "years.csv", index=False)
    pd.DataFrame(reg).to_csv(out / "regime.csv", index=False)
    print(pd.DataFrame(reg).to_string(index=False))


if __name__ == "__main__":
    main()
