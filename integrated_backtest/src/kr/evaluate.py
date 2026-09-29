"""SPEC 6-7.1 (Korea): grid summary, portfolios, random entry, Deflated Sharpe and the six criteria.
Run after src.kr.build.
Usage: python -m src.kr.evaluate   -> results/kr/{grid_summary,metrics,random,dsr,verdicts}.csv, equity.parquet,
                                      trades.parquet
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist

import numpy as np
import pandas as pd
from krxbt.data import load_index

from src import common as C
from src.kr import build as B
from src.kr.kernel import BRK
from src.kr.portfolio import Panel, c_orders, simulate

OUT = "kr"
STRATS = ("A2", "BO", "BM", "C")
EULER = 0.5772156649015329
ND = NormalDist()


def slip(cfg, strat) -> float:
    return cfg["kr"]["C"]["slippage"] if strat == "C" else cfg["costs"]["slippage"]


def net_ret(cfg, df: pd.DataFrame, cal: pd.DatetimeIndex, slippage: float) -> np.ndarray:
    cs = cfg["costs"]
    tax = C.tax_table(cfg, cal)[df["exit_i"].to_numpy()]
    keep = (1 - slippage) * (1 - cs["sell_fee"] - tax)
    return df["exit_px"].to_numpy() * keep / (df["entry_px"].to_numpy() * (1 + slippage) * (1 + cs["buy_fee"])) - 1


def grid_trades(cfg, cal, market: pd.Series) -> pd.DataFrame:
    g = pd.read_parquet(C.results_dir(OUT) / "grid_trades.parquet")
    g["strategy"] = g["strategy"].astype(str)
    g = g[g["reason"] != BRK].copy()
    g["ret"] = 0.0
    for s in ("A2", "B", "C"):
        m = g["strategy"] == s
        g.loc[m, "ret"] = net_ret(cfg, g[m], cal, slip(cfg, "C" if s == "C" else "A2"))
    g["market"] = g["tid"].map(market).map({0: "KOSPI", 1: "KOSDAQ"})
    return g


def grid_summary(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, c), x in g.groupby(["strategy", "combo"]):
        for mk, y in [("ALL", x), *list(x.groupby("market"))]:
            r = y["ret"]
            day = y.groupby("entry_i")["ret"].mean()
            pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
            top = r.nlargest(max(1, len(r) // 10))
            rows.append({"strategy": s, "combo": c, "market": mk, "trades": len(r), "win": float((r > 0).mean()),
                         "mean": float(r.mean()), "median": float(r.median()),
                         "pf": float(pos / neg) if neg > 0 else np.nan, "worst": float(r.min()),
                         "hold": float((y["exit_i"] - y["entry_i"] + 1).mean()),
                         "top10_share": float(top[top > 0].sum() / pos) if pos > 0 else np.nan,
                         "days": len(day), "day_sr": float(day.mean() / day.std(ddof=1)) if len(day) > 2 else np.nan})
    return pd.DataFrame(rows)


def dsr(x: np.ndarray, n: int, v: float) -> dict:
    m, s0 = x.mean(), x.std(ddof=0)
    sr = float(m / x.std(ddof=1))
    sk = float(np.mean((x - m) ** 3) / s0 ** 3)
    ku = float(np.mean((x - m) ** 4) / s0 ** 4)
    z_n = (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(v) * z_n
    z = (sr - sr0) * math.sqrt(len(x) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return {"sr": sr, "T": len(x), "skew": sk, "kurt": ku, "var_sr": v, "sr0": sr0, "dsr": ND.cdf(z)}


def kospi_tr(cfg, cal) -> pd.Series:
    c = load_index(cfg, "KOSPI")["close"].reindex(cal)
    return c.pct_change().fillna(0.0) + (1 + cfg["kr"]["kospi_dividend"]) ** (1 / 252) - 1


def stretches(cfg, idx) -> dict[str, np.ndarray]:
    te = pd.Timestamp(cfg["data"]["train_end"])
    return {"full": np.ones(len(idx), bool), "train": np.asarray(idx <= te), "test": np.asarray(idx > te)}


def curve_metrics(cfg, d: pd.Series, kospi: pd.Series) -> list[dict]:
    rows = []
    for s, m in stretches(cfg, d.index).items():
        x, k = d[m], kospi.reindex(d.index)[m]
        w = C.risk_weight(k, C.mdd(x))
        rows.append({"stretch": s, "cagr": C.cagr(x), "mdd": C.mdd(x), "sharpe": C.sharpe(x),
                     "kospi_cagr": C.cagr(k), "kospi_mdd": C.mdd(k), "risk_w": w, "risk_cagr": C.cagr(w * k),
                     "diff_risk": C.cagr(x) - C.cagr(w * k)})
    return rows


def candidates(cfg, strat: str) -> pd.DataFrame:
    c = pd.read_parquet(C.results_dir(OUT) / f"cand_{strat}.parquet")
    if strat == "C":
        c = pd.concat([c_orders(g, cfg["kr"]["C"]["orders"]) for _, g in c.groupby("label")], ignore_index=True)
    return c


_W = {}


def _init():
    _W["cfg"] = C.load_config()
    _W["P"] = Panel(in_ram=False)
    _W["rand"] = {}


def _random(job):
    strat, table, ks, seed = job
    cfg = _W["cfg"]
    if table not in _W["rand"]:
        _W["rand"][table] = pd.read_parquet(C.results_dir(OUT) / f"rand_{table}.parquet")
    res = simulate(cfg, _W["P"], None, slip(cfg, strat), random_k=ks, rand_table=_W["rand"][table], seed=seed)
    d = C.from_equity(res["equity"], cfg["kr"]["capital"])
    return {"strategy": strat, "seed": seed, "cagr": C.cagr(d), "mdd": C.mdd(d), "trades": len(res["trades"])}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir(OUT)
    P = Panel()
    cal = P.cal
    kospi = kospi_tr(cfg, cal)
    cap = cfg["kr"]["capital"]

    # grid
    g = grid_trades(cfg, cal, P.market)
    gs = grid_summary(g)
    gs.to_csv(out / "grid_summary.csv", index=False)
    print("grid summary rows", len(gs))

    # portfolios
    eqs, trs, rows, entries = {}, [], [], {}
    for strat in STRATS:
        cand = candidates(cfg, strat)
        for label, cd in cand.groupby("label", sort=False):
            runs = [(label, "equal", slip(cfg, strat))]
            if label == "main":
                runs += [("main_slip2", "equal", slip(cfg, strat) * cfg["dsr"]["slippage_mult"]),
                         ("main_turtle", "turtle", slip(cfg, strat))]
            for name, sizing, sl in runs:
                res = simulate(cfg, P, cd, sl, sizing)
                key = f"{strat}|{name}"
                eqs[key] = res["equity"]
                t = res["trades"].assign(run=key)
                trs.append(t)
                if name == "main":
                    entries[strat] = res["entries"]
                d = C.from_equity(res["equity"], cap)
                for r in curve_metrics(cfg, d, kospi):
                    rows.append({"strategy": strat, "run": name, **r, "trades": len(t),
                                 "win": float((t["ret"] > 0).mean()) if len(t) else np.nan,
                                 "avg_ret": float(t["ret"].mean()) if len(t) else np.nan,
                                 "invested": float(res["invested"].mean())})
                print(key, f"CAGR {C.cagr(d):+.2%} MDD {C.mdd(d):+.1%} trades {len(t)}")
    pd.DataFrame(eqs).to_parquet(out / "equity.parquet")
    pd.concat(trs, ignore_index=True).to_parquet(out / "trades.parquet")
    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(entries, index=cal).to_parquet(out / "entries.parquet")

    # random entry
    table = {"A2": "A2", "BO": "B", "BM": "B", "C": "C"}
    jobs = [(s, table[s], entries[s], cfg["kr"]["seed"] + i) for s in STRATS for i in range(cfg["kr"]["random_runs"])]
    with ProcessPoolExecutor(cfg["kr"]["workers"], initializer=_init) as ex:
        rnd = pd.DataFrame(list(ex.map(_random, jobs, chunksize=4)))
    rnd.to_csv(out / "random.csv", index=False)

    # Deflated Sharpe and criterion 6 on the seq grid trades of the judged combos (market ALL)
    main_combo = {"A2": ("A2", B.combo_id(B.a2_grid(cfg), B.judged(cfg)["A2"][0][1])),
                  "BO": ("B", B.combo_id(B.b_grid(cfg), B.judged(cfg)["BO"][0][1])),
                  "BM": ("B", B.combo_id(B.b_grid(cfg), B.judged(cfg)["BM"][0][1])),
                  "C": ("C", B.combo_id(B.c_grid(cfg), B.judged(cfg)["C"][0][1]))}
    drows = []
    for strat, (gname, cid) in main_combo.items():
        x = g[(g["strategy"] == gname) & (g["combo"] == cid)]
        day = x.groupby("entry_i")["ret"].mean()
        v = float(gs.loc[gs["strategy"] == gname, "day_sr"].var(ddof=1))
        top = day.nlargest(cfg["dsr"]["top_days"]).index
        rest = float(x.loc[~x["entry_i"].isin(top), "ret"].mean())
        drows.append({"strategy": strat, "grid": gname, "combo": cid, "trades": len(x), **dsr(day.to_numpy(), cfg["dsr"]["trials_kr"], v),
                      "best_days_removed_mean": rest})
    ds = pd.DataFrame(drows)
    ds.to_csv(out / "dsr.csv", index=False)

    # verdicts
    nbs = {s: [lbl for lbl, _ in B.judged(cfg)[s][1:]] for s in STRATS}
    vrows = []
    for strat in STRATS:
        m = met[met["strategy"] == strat]
        get = lambda run, st: m[(m["run"] == run) & (m["stretch"] == st)].iloc[0]
        f = get("main", "full")
        r = rnd[rnd["strategy"] == strat]["cagr"]
        pctile = float((r < f["cagr"]).mean())
        nb_ok = [get(lbl, "full")["diff_risk"] > 0 for lbl in nbs[strat] if len(m[m["run"] == lbl])]
        d = ds[ds["strategy"] == strat].iloc[0]
        ok = {1: d["dsr"] >= cfg["dsr"]["dsr_min"], 2: pctile >= cfg["dsr"]["random_pct_min"],
              3: all(get("main", st)["diff_risk"] > 0 for st in ("full", "train", "test")),
              4: sum(nb_ok) >= math.ceil(len(nbs[strat]) * 2 / 3),
              5: get("main_slip2", "full")["diff_risk"] > 0, 6: d["best_days_removed_mean"] > 0}
        vrows.append({"strategy": strat, **{f"c{i}": bool(v) for i, v in ok.items()}, "passed": all(ok.values()),
                      "failed": ", ".join(str(i) for i, v in ok.items() if not v), "random_pctile": pctile,
                      "random_median": float(r.median()), "random_p95": float(r.quantile(0.95)),
                      "neighbors_ok": f"{sum(nb_ok)}/{len(nbs[strat])}"})
    v = pd.DataFrame(vrows)
    v.to_csv(out / "verdicts.csv", index=False)
    print(v.to_string(index=False))
    print(ds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
