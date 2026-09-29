"""SPEC 7.2 (US): runs, the drawdown-matched hold, the five criteria.
Judged rows: U1 TQQQ (10%), U1 SOXL (12%), U2 TQQQ, U2 SOXL, U3-G, U3-A. Halves split each run's own period at its
middle date. The drawdown-matched hold is w x (hold of the same ETF, or SPY for U3) + cash, rebalanced daily.
Usage: python -m src.us.evaluate   -> results/us/{metrics,verdicts,u1_grid,u1_stats}.csv, equity.parquet
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import common as C
from src.us import data as D
from src.us import strategies as S

OUT = "us"


def compare(cfg, eq: pd.Series, hold_eq: pd.Series) -> dict:
    cap = cfg["us"]["capital"]
    d = C.from_equity(eq, cap)
    h = C.from_equity(hold_eq.reindex(eq.index), cap) if hold_eq.index[0] == eq.index[0] else \
        hold_eq.reindex(eq.index).pct_change().fillna(0.0)
    out = {}
    mid = eq.index[0] + (eq.index[-1] - eq.index[0]) / 2
    for s, m in {"full": np.ones(len(d), bool), "first": np.asarray(d.index <= mid), "second": np.asarray(d.index > mid)}.items():
        x, y = d[m], h[m]
        w = C.risk_weight(y, C.mdd(x))
        out[s] = {"first_day": x.index[0].date(), "last_day": x.index[-1].date(), "cagr": C.cagr(x), "mdd": C.mdd(x),
                  "sharpe": C.sharpe(x), "hold_cagr": C.cagr(y), "hold_mdd": C.mdd(y), "risk_w": w,
                  "risk_cagr": C.cagr(w * y), "diff_risk": C.cagr(x) - C.cagr(w * y)}
    return out


def main() -> None:
    cfg = C.load_config()
    u = cfg["us"]
    out = C.results_dir(OUT)
    names = ["TQQQ", "SOXL", "SPY", "EFA", "AGG", "BIL", "SCZ", "TLT"]
    px = {t: D.prices(cfg, t) for t in names}
    synth = D.synthetic(cfg)
    rows, eqs, verdict_inputs = [], {}, {}

    def record(key, eq, hold_eq, **info):
        eqs[key] = eq
        cmp = compare(cfg, eq, hold_eq)
        for s, r in cmp.items():
            rows.append({"run": key, "stretch": s, **info, **r})
        return cmp

    # U1 grid on both tickers (+ doubled costs for the judged cells), synthetic for TQQQ
    g1 = u["U1"]["grid"]
    u1_stats = []
    for tk, tgt in u["U1"]["tickers"].items():
        p = px[tk]
        h = S.hold(cfg, p)
        for n in g1["splits"]:
            for tg in g1["target"]:
                r = S.infinite(cfg, p, n, tg)
                cmp = record(f"U1|{tk}|{n}|{tg}", r["equity"], h, strategy="U1", ticker=tk, splits=n, target=tg)
                rec = [e[1] - e[0] if e[1] is not None else None for e in r["exhausts"]]
                u1_stats.append({"ticker": tk, "splits": n, "target": tg, "cycles": len(r["cycles"]),
                                 "exhausts": len(r["exhausts"]), "open_exhaust": sum(x is None for x in rec),
                                 "median_recovery_days": float(np.median([x for x in rec if x is not None])) if any(x is not None for x in rec) else np.nan,
                                 "max_recovery_days": max([x for x in rec if x is not None], default=np.nan)})
        r2 = S.infinite(cfg, p, u["U1"]["splits"], tgt, mult=2.0)
        record(f"U1|{tk}|main_cost2", r2["equity"], S.hold(cfg, p, 2.0), strategy="U1", ticker=tk)
    sp = synth[synth.index < px["TQQQ"].index[0]]
    rs = S.infinite(cfg, sp, u["U1"]["splits"], u["U1"]["tickers"]["TQQQ"])
    record("U1|SYNTH|main", rs["equity"], S.hold(cfg, sp), strategy="U1", ticker="SYNTH")
    ov = synth.loc[px["TQQQ"].index[0]:]
    gap = C.cagr(ov["close"].pct_change().dropna()) - C.cagr(px["TQQQ"]["close"].pct_change().dropna())
    pd.DataFrame(u1_stats).to_csv(out / "u1_stats.csv", index=False)

    # U2
    for tk in u["U2"]["tickers"]:
        p = px[tk]
        h = S.hold(cfg, p)
        for band in [u["U2"]["band"], *u["U2"]["neighbors_band"]]:
            record(f"U2|{tk}|{band}", S.value_rebalance(cfg, p, band)["equity"], h, strategy="U2", ticker=tk, band=band)
        record(f"U2|{tk}|main_cost2", S.value_rebalance(cfg, p, u["U2"]["band"], 2.0)["equity"], S.hold(cfg, p, 2.0),
               strategy="U2", ticker=tk)

    # U3
    spy_hold = lambda eq, mult=1.0: S.hold(cfg, px["SPY"].loc[eq.index[0]:], mult)
    logs = {}
    for lb in [u["U3"]["G"]["lookback"], *u["U3"]["G"]["neighbors"]]:
        r = S.switcher(cfg, px, S.momentum_picks(cfg, px, "G", lb))
        record(f"U3G|{lb}", r["equity"], spy_hold(r["equity"]), strategy="U3G", lookback=lb)
        logs[f"U3G|{lb}"] = r["log"]
    r = S.switcher(cfg, px, S.momentum_picks(cfg, px, "G", u["U3"]["G"]["lookback"]), 2.0)
    record("U3G|main_cost2", r["equity"], spy_hold(r["equity"], 2.0), strategy="U3G")
    for safe in [u["U3"]["A"]["safe"], u["U3"]["A"]["neighbor_safe"]]:
        r = S.switcher(cfg, px, S.momentum_picks(cfg, px, "A", safe=safe))
        record(f"U3A|{safe}", r["equity"], spy_hold(r["equity"]), strategy="U3A", safe=safe)
        logs[f"U3A|{safe}"] = r["log"]
    r = S.switcher(cfg, px, S.momentum_picks(cfg, px, "A", safe=u["U3"]["A"]["safe"]), 2.0)
    record("U3A|main_cost2", r["equity"], spy_hold(r["equity"], 2.0), strategy="U3A")
    pd.concat({k: v for k, v in logs.items()}, names=["run"]).reset_index(level=0).to_csv(out / "u3_switches.csv", index=False)

    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(eqs).to_parquet(out / "equity.parquet")
    get = lambda run, s="full": met[(met["run"] == run) & (met["stretch"] == s)].iloc[0]
    ok1 = lambda run, s="full": get(run, s)["diff_risk"] > 0

    vrows = []
    for tk, tgt in u["U1"]["tickers"].items():
        n0 = u["U1"]["splits"]
        main = f"U1|{tk}|{n0}|{tgt}"
        ts, ns = g1["target"], g1["splits"]
        nb = [f"U1|{tk}|{n}|{tgt}" for n in (ns[ns.index(n0) - 1], ns[ns.index(n0) + 1])] + \
             [f"U1|{tk}|{n0}|{t}" for t in (ts[ts.index(tgt) - 1], ts[ts.index(tgt) + 1])]
        nb_ok = sum(ok1(x) for x in nb)
        c = {1: ok1(main), 2: ok1(main, "first") and ok1(main, "second"), 3: nb_ok >= math.ceil(len(nb) * 2 / 3),
             4: ok1(f"U1|{tk}|main_cost2")}
        if tk == "TQQQ":
            c[5] = ok1("U1|SYNTH|main")
        vrows.append({"row": f"U1 {tk}", **{f"c{i}": bool(v) for i, v in c.items()}, "passed": all(c.values()),
                      "failed": ", ".join(str(i) for i, v in c.items() if not v), "neighbors_ok": f"{nb_ok}/{len(nb)}"})
    for tk in u["U2"]["tickers"]:
        main = f"U2|{tk}|{u['U2']['band']}"
        nb = [f"U2|{tk}|{b}" for b in u["U2"]["neighbors_band"]]
        c = {1: ok1(main), 2: ok1(main, "first") and ok1(main, "second"), 3: all(ok1(x) for x in nb),
             4: ok1(f"U2|{tk}|main_cost2")}
        vrows.append({"row": f"U2 {tk}", **{f"c{i}": bool(v) for i, v in c.items()}, "passed": all(c.values()),
                      "failed": ", ".join(str(i) for i, v in c.items() if not v), "neighbors_ok": f"{sum(ok1(x) for x in nb)}/{len(nb)}"})
    g0 = u["U3"]["G"]["lookback"]
    c = {1: ok1(f"U3G|{g0}"), 2: ok1(f"U3G|{g0}", "first") and ok1(f"U3G|{g0}", "second"),
         3: all(ok1(f"U3G|{x}") for x in u["U3"]["G"]["neighbors"]), 4: ok1("U3G|main_cost2")}
    vrows.append({"row": "U3-G", **{f"c{i}": bool(v) for i, v in c.items()}, "passed": all(c.values()),
                  "failed": ", ".join(str(i) for i, v in c.items() if not v)})
    sa, sn = u["U3"]["A"]["safe"], u["U3"]["A"]["neighbor_safe"]
    c = {1: ok1(f"U3A|{sa}"), 2: ok1(f"U3A|{sa}", "first") and ok1(f"U3A|{sa}", "second"), 3: ok1(f"U3A|{sn}"),
         4: ok1("U3A|main_cost2")}
    vrows.append({"row": "U3-A", **{f"c{i}": bool(v) for i, v in c.items()}, "passed": all(c.values()),
                  "failed": ", ".join(str(i) for i, v in c.items() if not v)})
    v = pd.DataFrame(vrows)
    v["synth_gap"] = gap
    v.to_csv(out / "verdicts.csv", index=False)
    print(v.to_string(index=False))
    print(f"synthetic - TQQQ CAGR gap over the overlap: {gap:+.2%}")
    show = met[(met["stretch"] == "full") & ~met["run"].str.contains(r"\|20\||\|30\||\|60\|", regex=True)]
    print(show[["run", "cagr", "mdd", "hold_cagr", "hold_mdd", "risk_w", "risk_cagr", "diff_risk"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
