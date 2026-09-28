"""Walk-forward check: pick the combo with only the past, trade it the next year.

For every test year Y (config walkforward.first_test_year .. last year):
  train = all grid trades with entry before Y (expanding window)
  pick  = the (combo, market) with the highest mean return per trade among
          those with at least min_trades train trades
  test  = that pick's trades entered in year Y
The yearly picks are then run as one portfolio (1/5 slots as in
src.portfolio): in year Y only the pick of Y may open positions, positions
already open keep running. Compared with the index and with the combo that is
best over the whole period (hindsight).

Also per year: rank correlation of train vs test expectancy over all combos,
and where the pick landed among all combos in the test year.

Usage:  python -m src.walkforward          # Korea (results/trades.parquet)
        python -m src.walkforward --us     # US (results/us/trades.parquet)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from krxbt import portfolio as kp
from krxbt.portfolio import stats

from .common import load_calendar, load_config, results_dir
from .grid import combos
from .portfolio import filter_label
from .report import md_table, pct, stop_label


def yearly(tr: pd.DataFrame, key: str) -> pd.DataFrame:
    """sum and count of returns per (combo, key, year), with key=ALL added."""
    d = pd.DataFrame({"combo": tr["combo"].to_numpy(), key: tr[key].astype(str).to_numpy(),
                      "year": tr["entry_date"].dt.year.to_numpy(), "ret": tr["ret"].astype("float64").to_numpy()})
    g = d.groupby(["combo", key, "year"])["ret"].agg(["sum", "count"])
    a = d.groupby(["combo", "year"])["ret"].agg(["sum", "count"])
    a[key] = "ALL"
    a = a.reset_index().set_index(["combo", key, "year"])
    return pd.concat([g, a]).sort_index()


def walk(y: pd.DataFrame, key: str, years: list[int], min_trades: int, min_test: int) -> pd.DataFrame:
    rows = []
    by_year = y.reset_index()
    for Y in years:
        tr = by_year[by_year["year"] < Y].groupby(["combo", key])[["sum", "count"]].sum()
        te = by_year[by_year["year"] == Y].set_index(["combo", key])[["sum", "count"]]
        tr["ev"] = tr["sum"] / tr["count"]
        te["ev"] = te["sum"] / te["count"]
        ok = tr[tr["count"] >= min_trades].sort_values("ev", ascending=False)
        if ok.empty:
            continue
        pick = ok.index[0]
        top10 = ok.index[:10]
        t_ok = te[te["count"] >= min_test]
        pick_ev = te["ev"].get(pick, np.nan)
        pick_n = int(te["count"].get(pick, 0))
        both = ok.join(t_ok, lsuffix="_tr", rsuffix="_te", how="inner")
        rho = both["ev_tr"].rank().corr(both["ev_te"].rank()) if len(both) > 5 else np.nan
        pctile = (t_ok["ev"] < pick_ev).mean() if pick in t_ok.index else np.nan
        top10_te = te.reindex(top10)
        rows.append({"year": Y, "combo": int(pick[0]), key: pick[1], "train_ev": ok["ev"].iloc[0],
                     "train_n": int(ok["count"].iloc[0]), "test_ev": pick_ev, "test_n": pick_n,
                     "test_pctile": pctile, "median_test_ev": t_ok["ev"].median(),
                     "top10_test_ev": top10_te["sum"].sum() / top10_te["count"].sum() if top10_te["count"].sum() else np.nan,
                     "rho": rho, "n_combos": len(both)})
    return pd.DataFrame(rows)


def label(c: pd.Series, key_val: str) -> str:
    return (f"{key_val} ≤{c['threshold']} {stop_label(c['stop'])} {int(c['hold'])}일 "
            f"{filter_label(c['market_filter'])}")


def stitched(picks: pd.DataFrame, key: str, cand: pd.DataFrame) -> pd.DataFrame:
    """Candidate trades of each year's pick, entered in that year."""
    parts = []
    for _, p in picks.iterrows():
        t = cand[(cand["combo"] == p["combo"]) & (cand["entry_date"].dt.year == p["year"])]
        if p[key] != "ALL":
            t = t[t[key].astype(str) == p[key]]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def kr_candidates(cfg: dict, pairs: list[tuple[int, str]]) -> pd.DataFrame:
    from .portfolio import candidate_trades
    return candidate_trades(cfg, sorted({c for c, _ in pairs}))


def us_candidates(cfg: dict, pairs: list[tuple[int, str]]) -> pd.DataFrame:
    from .simulate import simulate
    from .us import frames
    cmb = combos(cfg)
    fr = list(frames(cfg))
    out = []
    for cid in sorted({c for c, _ in pairs}):
        c = cmb.loc[cid]
        stop = None if pd.isna(c["stop"]) else float(c["stop"])
        for t, kind, f in fr:
            x = simulate(f, cfg, c["threshold"], stop, int(c["hold"]), c["market_filter"], independent=True)
            if len(x):
                x["ticker"], x["kind"], x["combo"] = t, kind, cid
                out.append(x)
    df = pd.concat(out, ignore_index=True)
    return df, pd.DataFrame({t: f["close"] for t, _, f in fr})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true")
    a = ap.parse_args()
    if a.us:
        from .us import us_config
        from krxbt import us as kus
        cfg = us_config()
        rd = results_dir() / "us"
        key, min_trades, cap = "kind", cfg["us"]["min_trades"], cfg["us"]["initial_capital"]
        cal_all = kus.load_calendar(cfg)
        idx_name = cfg["us"]["market_index"]
        index_close = kus.load_index(cfg, idx_name)["close"]
    else:
        cfg = load_config()
        rd = results_dir()
        key, min_trades, cap = "market", cfg["stage2"]["min_trades"], cfg["stage2"]["initial_capital"]
        cal_all = load_calendar(cfg)
        idx_name = "KOSPI"
        from .common import load_index
        index_close = load_index(cfg, "KOSPI")["close"]
    wf = cfg["walkforward"]
    tr = pd.read_parquet(rd / "trades.parquet", columns=["combo", key, "entry_date", "ret"])
    y = yearly(tr, key)
    del tr
    last = int(y.index.get_level_values("year").max())
    years = list(range(wf["first_test_year"], last + 1))
    picks = walk(y, key, years, min_trades, wf["min_test_trades"])
    cmb = combos(cfg)
    picks = picks.join(cmb, on="combo")
    picks.to_csv(rd / "walkforward.csv", index=False, encoding="utf-8-sig")

    # hindsight best over the whole period, same rule
    tot = y.groupby(["combo", key])[["sum", "count"]].sum()
    tot = tot[tot["count"] >= min_trades]
    best = (tot["sum"] / tot["count"]).idxmax()
    print("hindsight best:", best, label(cmb.loc[best[0]], best[1]))

    pairs = list(zip(picks["combo"], picks[key])) + [best]
    if a.us:
        cand, closes = us_candidates(cfg, pairs)
    else:
        cand = kr_candidates(cfg, pairs)
        closes = None
    start = pd.Timestamp(f"{wf['first_test_year']}-01-01")
    cal = cal_all[cal_all >= start]
    if closes is None:
        closes = kp.close_panel(cfg, cand["ticker"].astype(str).unique(), cal)
    else:
        closes = closes.reindex(cal).ffill()
    slots = cfg["stage2"]["max_positions"]
    eq_wf, info_wf = kp.run_portfolio(cal, stitched(picks, key, cand), closes, cap, slots, "disp")
    hb = cand[(cand["combo"] == best[0]) & (cand["entry_date"] >= start)]
    if best[1] != "ALL":
        hb = hb[hb[key].astype(str) == best[1]]
    eq_hb, info_hb = kp.run_portfolio(cal, hb, closes, cap, slots, "disp")
    ix = index_close.reindex(cal).ffill()
    ix = ix / ix.iloc[0] * cap

    out = [f"# 걸어가며 검증 ({'미국' if a.us else '한국'})", "",
           f"해마다 그 해 전까지의 거래만 보고 기대값 1등 조합(거래 {min_trades}건 이상)을 고른 뒤, 그 조합을 다음 한 해 동안 쓴다. "
           f"{wf['first_test_year']}년부터 한 해씩 옮긴다. 학습 기간은 {cfg['data']['start'][:4]}년부터 늘어난다.", "",
           "- 검증 해 기대값: 고른 조합의 그 해 거래당 평균",
           f"- 그 해 전체 중앙값: 그 해 거래 {wf['min_test_trades']}건 이상인 모든 조합의 중앙값",
           "- 백분위: 그 해 모든 조합 가운데 고른 조합의 위치 (100% = 1등)",
           "- 순위 상관: 학습 기대값 순위와 그 해 기대값 순위의 스피어만 상관", ""]
    out += [md_table(pd.DataFrame({
        "검증 해": picks["year"], "고른 조합": [label(r, r[key]) for _, r in picks.iterrows()],
        "학습 기대값": picks["train_ev"].map(pct), "검증 해 기대값": picks["test_ev"].map(pct),
        "검증 해 거래": picks["test_n"], "그 해 전체 중앙값": picks["median_test_ev"].map(pct),
        "학습 상위10 평균": picks["top10_test_ev"].map(pct),
        "백분위": picks["test_pctile"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}"),
        "순위 상관": picks["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")})), ""]
    w = picks["test_n"]
    pooled = (picks["test_ev"].fillna(0) * w).sum() / w.sum() if w.sum() else np.nan
    out += [f"고른 조합의 검증 거래 전체 평균: {pct(pooled)} ({int(w.sum()):,}건). "
            f"고른 조합이 그 해 중앙값을 넘은 해: 거래가 있던 {picks['test_ev'].notna().sum()}해 중 "
            f"{(picks['test_ev'] > picks['median_test_ev']).sum()}해 (거래 없는 해 {picks['test_ev'].isna().sum()}). "
            f"순위 상관 평균 {picks['rho'].mean():.2f}.", ""]
    rows = [{"방식": "해마다 고른 조합 (걸어가며)", **stats(eq_wf), "체결": info_wf["trades_taken"],
             "보유 중인 날": info_wf["invested_share"]},
            {"방식": f"전체 기간 1등 조합 (뒤늦게 앎): {label(cmb.loc[best[0]], best[1])}", **stats(eq_hb),
             "체결": info_hb["trades_taken"], "보유 중인 날": info_hb["invested_share"]},
            {"방식": f"{idx_name} 지수 보유", **stats(ix)}]
    pr = pd.DataFrame(rows)
    out += [f"## 포트폴리오 ({wf['first_test_year']}-01 → 최근, 최대 {slots}종목)", "",
            "걸어가며: 그 해에는 그 해 고른 조합의 신호만 산다. 이미 들고 있던 종목은 원래 규칙대로 판다.", "",
            md_table(pd.DataFrame({
                "방식": pr["방식"], "연환산": pr["cagr"].map(lambda v: f"{v:+.1%}"), "MDD": pr["mdd"].map(lambda v: f"{v:.0%}"),
                "체결 거래": pr.get("체결", pd.Series(dtype=float)).map(lambda v: "" if pd.isna(v) else f"{int(v):,}"),
                "보유 중인 날": pr.get("보유 중인 날", pd.Series(dtype=float)).map(lambda v: "" if pd.isna(v) else f"{v:.0%}")})), ""]
    pd.DataFrame({"walkforward": eq_wf, "hindsight": eq_hb, "index": ix}).to_csv(rd / "walkforward_equity.csv")
    (rd / "walkforward.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
