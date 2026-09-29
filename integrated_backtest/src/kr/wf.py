"""SPEC3: walk-forward selection over the 540-cell B grid (market ALL).

For each test year Y (wf.first_year .. wf.last_year) the cell with the highest mean net trade return among the seq
grid trades that were already sold by Dec 31 of Y-1 (entries from wf.train_start; cells with fewer than
wf.min_trades such trades are left out; ties go to the lower cell number) is used for new entries in year Y. One
5-slot account runs 2017-01-02 -> end; a position keeps the exit rules of the cell it was bought under.
Sensitivities (not judged): rolling 3-year window, day-Sharpe yardstick, slippage 0.2% (that one is criterion 5).
The per-ticker pass below repeats the B part of src.kr.build._pass2 (signals, exits, all-mode trades) for the picked
cells only, and builds the random-entry tables for their exit rules.
Usage: python -m src.kr.wf   -> results/kr/wf/{picks.csv, equity.parquet, trades.parquet, random.csv, metrics.csv,
                                 years.csv, verdict.csv, report.md, checks.md}
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import common as C
from src.kr import build as B
from src.kr import kernel as K
from src.kr.evaluate import curve_metrics, dsr, grid_trades, kospi_tr, net_ret
from src.kr.portfolio import Panel, simulate

OUT = "kr/wf"


# ---------------------------------------------------------------- selection
def picks(cfg, g: pd.DataFrame, cal: pd.DatetimeIndex, mode: str = "main") -> pd.DataFrame:
    w = cfg["wf"]
    b = g[g["strategy"] == "B"]
    rows = []
    for y in range(w["first_year"], w["last_year"] + 1):
        end_i = int(cal.searchsorted(pd.Timestamp(f"{y - 1}-12-31"), side="right")) - 1
        start = pd.Timestamp(w["train_start"]) if mode != "rolling" else pd.Timestamp(f"{y - w['rolling_years']}-01-01")
        s_i = int(cal.searchsorted(start))
        x = b[(b["entry_i"] >= s_i) & (b["exit_i"] <= end_i)]
        if mode == "sharpe":
            day = x.groupby(["combo", "entry_i"])["ret"].mean()
            stat = day.groupby("combo").agg(lambda s: s.mean() / s.std(ddof=1) if len(s) > 2 else np.nan)
        else:
            stat = x.groupby("combo")["ret"].mean()
        n = x.groupby("combo").size()
        ok = n[n >= w["min_trades"]].index
        st = stat.reindex(ok).dropna().sort_index()
        best = int(st.idxmax())
        rows.append({"year": y, "mode": mode, "combo": best, **B.b_grid(cfg)[best], "train_trades": int(n[best]),
                     "train_stat": float(st[best]), "train_end": str(cal[end_i].date()), "leak": int((x["exit_i"] > end_i).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-ticker candidates for picked cells
_J = {}


def _init(combos, exits):
    B._init()
    _J["combos"], _J["exits"] = combos, exits


def _worker(tid: int):
    cfg, cal = B._G["cfg"], B._G["cal"]
    f = B._frame(tid)
    if f is None:
        return None
    off = int(cal.get_loc(f.index[0]))
    start = int(cal.searchsorted(pd.Timestamp(cfg["data"]["start"])))
    ind = B.indicators(f)
    n = len(f)
    o, h, l, c = (f[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    valid = f["valid"].to_numpy(bool)
    locked = valid & (h == l)
    brk = f["data_break"].to_numpy(bool)
    elig = f["eligible"].to_numpy(bool) & valid
    regime = f["idx_regime"].to_numpy(bool)
    ends_early = off + n - 1 < len(cal) - 1
    rs = np.asarray(B._G["rs"][off: off + n, tid], np.float64)
    st = max(start - off, 0)
    g = lambda a, v: np.nan_to_num(a, nan=v)
    hh = {20: g(ind["hh20"], np.inf), 60: g(ind["hh60"], np.inf)}
    mult = g(ind["mult"], 0)
    ma = {k: g(ind[k], np.nan) for k in ("ma50", "ma150", "ma200", "ma200_20")}
    tmpl = (c > ma["ma50"]) & (ma["ma50"] > ma["ma150"]) & (ma["ma150"] > ma["ma200"]) & (ma["ma200"] > ma["ma200_20"]) \
        & (c >= g(ind["lo52"], np.inf) * 1.3) & (c >= g(ind["hi52"], np.inf) * 0.75) & (g(rs, 0) >= 70)
    filt = {"none": np.ones(n, bool), "rs80": g(rs, 0) >= 80, "template": tmpl}
    exits = {"ll10": c < g(ind["ll10"], -np.inf), "ll20": c < g(ind["ll20"], -np.inf), "ma20": c < g(ind["ma20"], -np.inf)}
    atr = g(ind["atr20"], np.inf)
    max_hold = cfg["kr"]["B"]["max_hold"]

    def run(sig, stop, ex):
        sm, sp = (2, 0.0) if stop == "atr2" else (1, float(stop))
        return K.run_trades(o, h, l, c, valid, locked, brk, sig, sm, sp, atr, exits[ex], max_hold, st, True, ends_early)

    cand, rand = [], []
    for cid in _J["combos"]:
        cb = B.b_grid(cfg)[cid]
        sig = elig & (c > hh[cb["n"]]) & (mult >= cb["mult"]) & filt[cb["filter"]]
        if cb["regime"]:
            sig &= regime
        df = B._trades_df(run(sig, cb["stop"], cb["exit"]), tid, {"prio": mult, "atr": g(ind["atr20"], np.nan)})
        df.insert(0, "combo", cid)
        cand.append(df)
    for stop, ex in _J["exits"]:
        df = B._trades_df(run(elig.copy(), stop, ex), tid)
        df.insert(0, "exit_cfg", f"{stop}|{ex}")
        rand.append(df)
    for df in cand + rand:
        for col in ("sig_i", "entry_i", "exit_i"):
            df[col] = df[col] + off
    return pd.concat(cand, ignore_index=True), pd.concat(rand, ignore_index=True)


def build_cands(cfg, combos: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = B.b_grid(cfg)
    exits = sorted({(grid[c]["stop"], grid[c]["exit"]) for c in combos}, key=str)
    uni_n = len(B.universe_tickers(cfg))
    cand, rand = [], []
    with ProcessPoolExecutor(cfg["kr"]["workers"], initializer=_init, initargs=(combos, exits)) as ex:
        for res in ex.map(_worker, range(uni_n), chunksize=8):
            if res is not None:
                cand.append(res[0])
                rand.append(res[1])
    return pd.concat(cand, ignore_index=True), pd.concat(rand, ignore_index=True)


# ---------------------------------------------------------------- walk-forward account
def year_rows(cal: pd.DatetimeIndex, df: pd.DataFrame, pk: pd.DataFrame, key: str, cfg) -> pd.DataFrame:
    """Rows of `df` whose entry falls in a test year, taken from that year's picked cell (key = combo or exit_cfg)."""
    grid = B.b_grid(cfg)
    yr = cal.year[df["entry_i"].to_numpy()]
    want = {int(r.year): (r.combo if key == "combo" else f"{grid[int(r.combo)]['stop']}|{grid[int(r.combo)]['exit']}")
            for r in pk.itertuples()}
    m = np.array([want.get(int(y)) == v for y, v in zip(yr, df[key].to_numpy())])
    return df[m]


_R = {}


def _rinit(table_path):
    _R["cfg"] = C.load_config()
    _R["P"] = Panel(in_ram=False)
    _R["table"] = pd.read_parquet(table_path)


def _rrun(job):
    ks, seed = job
    cfg = _R["cfg"]
    res = simulate(cfg, _R["P"], None, cfg["costs"]["slippage"], random_k=ks, rand_table=_R["table"], seed=seed)
    d = C.from_equity(res["equity"], cfg["kr"]["capital"])
    return {"seed": seed, "cagr": C.cagr(d), "mdd": C.mdd(d)}


def main() -> None:
    cfg = C.load_config()
    w = cfg["wf"]
    out = C.results_dir(OUT)
    P = Panel()
    cal = P.cal
    g = grid_trades(cfg, cal, P.market)
    pk = {m: picks(cfg, g, cal, m) for m in ("main", "rolling", "sharpe")}
    allp = pd.concat(pk.values(), ignore_index=True)
    allp.to_csv(out / "picks.csv", index=False)
    print(pk["main"][["year", "combo", "n", "mult", "stop", "exit", "filter", "regime", "train_trades", "train_stat"]].to_string(index=False))
    combos = sorted(set(allp["combo"].astype(int)))
    cand, rand = build_cands(cfg, combos)
    t0 = int(cal.searchsorted(pd.Timestamp(f"{w['first_year']}-01-01")))
    cap = cfg["kr"]["capital"]
    kospi = kospi_tr(cfg, cal)
    runs, eqs, trs, mrows = {}, {}, [], []
    cfg_wf = {**cfg, "data": {**cfg["data"], "start": f"{w['first_year']}-01-01"}}
    for mode in ("main", "rolling", "sharpe"):
        cd = year_rows(cal, cand[cand["entry_i"] >= t0], pk[mode], "combo", cfg)
        slips = [("main", cfg["costs"]["slippage"])] + ([("slip2", w["slippage_check"])] if mode == "main" else [])
        for tag, sl in slips:
            res = simulate(cfg_wf, P, cd, sl)
            key = f"{mode}|{tag}"
            runs[key] = res
            eqs[key] = res["equity"]
            trs.append(res["trades"].assign(run=key))
            d = C.from_equity(res["equity"], cap)
            for r in curve_metrics(cfg_wf, d, kospi):
                mrows.append({"run": key, **r})
            for i, (a, b) in enumerate(w["halves"]):
                m = (d.index >= a) & (d.index <= b)
                x, k = d[m], kospi.reindex(d.index)[m]
                wr = C.risk_weight(k, C.mdd(x))
                mrows.append({"run": key, "stretch": f"half{i + 1}", "cagr": C.cagr(x), "mdd": C.mdd(x), "sharpe": C.sharpe(x),
                              "kospi_cagr": C.cagr(k), "kospi_mdd": C.mdd(k), "risk_w": wr, "risk_cagr": C.cagr(wr * k),
                              "diff_risk": C.cagr(x) - C.cagr(wr * k)})
            print(key, f"CAGR {C.cagr(d):+.2%} MDD {C.mdd(d):+.1%} trades {len(res['trades'])}")
    # fixed-cell references cut to the same period
    fixed = pd.read_parquet(C.results_dir("kr") / "equity.parquet")
    for k in ("BO|main", "BM|main"):
        e = fixed[k][fixed.index >= cal[t0]]
        d = e.pct_change().fillna(0.0)
        for r in curve_metrics(cfg_wf, d, kospi):
            mrows.append({"run": f"ref {k}", **r})
    met = pd.DataFrame(mrows)
    met.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(eqs).to_parquet(out / "equity.parquet")
    trades = pd.concat(trs, ignore_index=True)
    trades.to_parquet(out / "trades.parquet")

    # random entry for the main walk-forward account
    rt = year_rows(cal, rand[rand["entry_i"] >= t0], pk["main"], "exit_cfg", cfg)
    rpath = out / "rand_table.parquet"
    rt.to_parquet(rpath)
    ks = runs["main|main"]["entries"]
    with ProcessPoolExecutor(cfg["kr"]["workers"], initializer=_rinit, initargs=(rpath,)) as ex:
        rnd = pd.DataFrame(list(ex.map(_rrun, [(ks, cfg["kr"]["seed"] + i) for i in range(cfg["kr"]["random_runs"])], chunksize=4)))
    rnd.to_csv(out / "random.csv", index=False)

    # year-by-year rank of the picked cell among the 540 cells (test-year entries, seq grid trades)
    b = g[g["strategy"] == "B"]
    yrows = []
    for r in pk["main"].itertuples():
        y = int(r.year)
        m = (cal.year[b["entry_i"].to_numpy()] == y)
        mean = b[m].groupby("combo")["ret"].mean()
        pick = mean.get(int(r.combo), np.nan)
        yrows.append({"year": y, "combo": int(r.combo), "pick_mean": pick, "grid_median": float(mean.median()),
                      "pctile": float((mean < pick).mean()), "above_median": bool(pick > mean.median()), "cells": len(mean)})
    yr = pd.DataFrame(yrows)
    yr.to_csv(out / "years.csv", index=False)

    # verdict
    gs = pd.read_csv(C.results_dir("kr") / "grid_summary.csv")
    v_sr = float(gs.loc[gs["strategy"] == "B", "day_sr"].var(ddof=1))
    tm = trades[trades["run"] == "main|main"]
    day = tm.groupby("entry_i")["ret"].mean().to_numpy()
    ds = dsr(day, w["trials"], v_sr)
    get = lambda run, st: met[(met["run"] == run) & (met["stretch"] == st)].iloc[0]
    f = get("main|main", "full")
    pct = float((rnd["cagr"] < f["cagr"]).mean())
    ok = {1: f["diff_risk"] > 0, 2: get("main|main", "half1")["diff_risk"] > 0 and get("main|main", "half2")["diff_risk"] > 0,
          3: int(yr["above_median"].sum()) >= w["years_above_median"], 4: pct >= cfg["dsr"]["random_pct_min"],
          5: get("main|slip2", "full")["diff_risk"] > 0, 6: ds["dsr"] >= cfg["dsr"]["dsr_min"]}
    vd = pd.DataFrame([{**{f"c{i}": bool(x) for i, x in ok.items()}, "passed": all(ok.values()),
                        "failed": ", ".join(str(i) for i, x in ok.items() if not x), "random_pctile": pct,
                        "random_median": float(rnd["cagr"].median()), "random_p95": float(rnd["cagr"].quantile(0.95)),
                        "years_above": int(yr["above_median"].sum()), **{f"dsr_{k}": v for k, v in ds.items()}}])
    vd.to_csv(out / "verdict.csv", index=False)
    report(cfg, pk, met, yr, rnd, vd, trades)
    print(vd.T.to_string())


def report(cfg, pk, met, yr, rnd, vd, trades) -> None:
    out = C.results_dir(OUT)
    v = vd.iloc[0]
    tm = trades[trades["run"] == "main|main"]
    top10 = tm["pnl"].nlargest(10).sum() / tm["pnl"].sum()
    get = lambda run, st: met[(met["run"] == run) & (met["stretch"] == st)].iloc[0]
    f = get("main|main", "full")
    names = {1: "공짜 아님", 2: "두 기간", 3: "고르는 힘", 4: "무작위 95%", 5: "슬리피지 0.2%", 6: "DSR ≥ 0.95"}
    det = {1: f"연 {C.pct(f['cagr'])}, 최대 낙폭 {C.pct(f['mdd'])}; 코스피 {C.pct(f['kospi_cagr'])}; 낙폭 맞춘 코스피(w {f['risk_w']:.2f}) {C.pct(f['risk_cagr'])} → {C.pp(f['diff_risk'])}",
           2: " / ".join(f"{'2017 → 2021' if i == 1 else '2022 → 끝'} {C.pp(get('main|main', f'half{i}')['diff_risk'])}" for i in (1, 2)),
           3: f"10개 해 중 {int(v['years_above'])}개가 540칸 중앙값보다 높음(기준 8)",
           4: f"무작위 200번 중앙값 {C.pct(v['random_median'])}, 95 백분위 {C.pct(v['random_p95'])} → {v['random_pctile']:.0%} 위치",
           5: f"연 {C.pct(get('main|slip2', 'full')['cagr'])}, 대비 {C.pp(get('main|slip2', 'full')['diff_risk'])}",
           6: f"DSR {v['dsr_dsr']:.3f}(진입일 샤프 {v['dsr_sr']:+.3f}, 기대 최대 {v['dsr_sr0']:.3f}, 진입일 {int(v['dsr_T'])}개)"}
    L = ["# 걸어가며 고르기 결과 (SPEC3)", "",
         f"B 그리드 540칸(합산 시장)에서 해마다 칸을 골라 2017-01 → {cfg['data']['end']} 한 계좌를 굴렸다.", "",
         f"**판정: {'통과' if v['passed'] else '탈락'}**" + ("" if v["passed"] else f" (못 넘은 기준 {v['failed']})"), "",
         C.md_table(pd.DataFrame([{"기준": f"{i} {names[i]}", "결과": "O" if v[f"c{i}"] else "X", "내용": det[i]} for i in names])), "",
         "## 해마다 고른 칸", "",
         C.md_table(pd.DataFrame([{"해": int(r.year), "칸": int(r.combo), "신고가": r.n, "배수": r.mult, "손절": r.stop, "이탈": r.exit,
                                   "필터": r.filter, "국면": r.regime, "학습 거래": r.train_trades, "학습 거래당": C.pct(r.train_stat, 2),
                                   "그 해 거래당": C.pct(y.pick_mean, 2), "그 해 540칸 중앙값": C.pct(y.grid_median, 2), "백분위": f"{y.pctile:.0%}"}
                                  for r, y in zip(pk["main"].itertuples(), yr.itertuples())])), "",
         "## 계좌 표", "",
         C.md_table(pd.DataFrame([{"판본": r.run, "구간": {"full": "전체", "train": "2017 → 2020", "test": "2021 → 끝",
                                                          "half1": "2017 → 2021", "half2": "2022 → 끝"}.get(r.stretch, r.stretch), "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd), "샤프": C.num(r.sharpe),
                                   "코스피": C.pct(r.kospi_cagr), "낙폭 맞춘 코스피(w)": f"{C.pct(r.risk_cagr)} ({r.risk_w:.2f})", "그 대비": C.pp(r.diff_risk)}
                                  for r in met.itertuples()])), "",
         f"가장 큰 거래 10건의 손익이 판정 계좌 전체 손익의 {top10:.0%} 다(나머지 거래를 합치면 손실).", "",
         "main|main 이 판정 대상이다. main|slip2 는 기준 5, rolling·sharpe 는 민감도(판정 제외), ref 는 SPEC 의 고정 칸 계좌를 같은 기간으로 자른 참고다.", "",
         "민감도의 고른 칸:", "",
         C.md_table(pd.concat([pk["rolling"], pk["sharpe"]])[["mode", "year", "combo", "n", "mult", "stop", "exit", "filter", "regime"]]), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
