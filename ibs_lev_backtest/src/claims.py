"""Stage 1: the original's five claims, re-measured on us_data with our own code.

Unit of measure for every claim: the days in the bottom 20% of a ticker's IBS
(IBS <= its own 20th percentile over the sample) and the NEXT day's open ->
close return, with its t value (mean / (std / sqrt(n))).
Sample: data.start (2008-01-01) to data.end (2026-07-31, inclusive); leveraged ETFs from
listing. Deciles and the 20% cut are taken on the whole sample (descriptive,
as the original's decile_response).
Gap days: signal days whose next open equals that day's close (the original
warns Yahoo index data before ~2010 has open = previous close, which inflates
C1). They are counted per ticker; every claim is also measured without them,
and the verdicts use the version without them.

Usage:  python -m src.claims [--only C1 C2 ...]
Writes  results/claims.csv, results/claims_deciles.csv, results/claims_verdicts.csv,
        results/claims.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from krxbt import us as kus

from .common import capital_usd, daily_returns, fmt_table, load_config, md_table, pct, results_dir, share, stats, tstat
from .data import frames, index_close
from .sizing import account
from .strategy import calendar, trades


def forward_frame(f: pd.DataFrame, start: str, tol: float) -> pd.DataFrame:
    """Per traded day t: IBS_t, oc = close_{t+1} / open_{t+1} - 1, on = open_{t+1} / close_t - 1.

    gap uses RAW (unadjusted) open/close: the adjustment factor adj_close / close drifts by
    ~1e-7 per day for dividend payers, so testing the tiny gap_tol against adjusted prices misses
    real "open == previous close" days for every dividend-paying ticker."""
    g = f[f["valid"]]
    no, nc = g["open"].shift(-1), g["close"].shift(-1)
    rno, rc = g["raw_open"].shift(-1), g["raw_close"]
    gap = (rno / rc - 1).abs() <= tol
    out = pd.DataFrame({"ibs": g["ibs"], "oc": nc / no - 1, "on": no / g["close"] - 1, "gap": gap}, index=g.index)
    out = out[out.index >= pd.Timestamp(start)].dropna(subset=["ibs", "oc"])
    return out


def mark(fw: pd.DataFrame, bottom_share: float, buckets: int) -> pd.DataFrame:
    fw = fw.copy()
    fw["decile"] = pd.qcut(fw["ibs"], buckets, labels=False, duplicates="drop")
    fw["bottom"] = fw["ibs"] <= fw["ibs"].quantile(bottom_share)
    fw["top"] = fw["ibs"] >= fw["ibs"].quantile(1 - bottom_share)
    return fw


def sample(cfg: dict, tickers: list[str]) -> dict[str, pd.DataFrame]:
    cl = cfg["claims"]
    return {t: mark(forward_frame(f, cfg["data"]["start"], cl["gap_tol"]), cl["bottom_share"], cl["buckets"])
            for t, f in frames(cfg, tickers).items()}


def variants(fw: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [("all", fw), ("no_gap", fw[~fw["gap"]])]


def decile_table(fw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, x in fw.groupby("decile"):
        rows.append({"decile": int(k) + 1, "ibs_low": float(x["ibs"].min()), "ibs_high": float(x["ibs"].max()),
                     **tstat(x["oc"])})
    return pd.DataFrame(rows)


def rank_corr(means) -> float:
    m = np.asarray(means, np.float64)
    return float(np.corrcoef(np.arange(len(m)), m)[0, 1])


def split_half(fw: pd.DataFrame) -> float:
    """Correlation of the decile means of the first and second half of the days."""
    h = len(fw) // 2
    a = fw.iloc[:h].groupby("decile")["oc"].mean()
    b = fw.iloc[h:].groupby("decile")["oc"].mean().reindex(a.index)
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def gap_table(cfg: dict, sm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of days with open == previous close, per ticker and for the us_data index files.

    `days` is the count the share is taken over: for tickers, the forward-frame rows (each has a
    next day); for the index rows, the days with a previous close (len(ix) - 1), the same unit."""
    rows = [{"name": t, "days": len(fw), "gap_share": float(fw["gap"].mean())} for t, fw in sm.items()]
    for name in ("SPX", "NDX"):
        ix = kus.load_index(cfg, name)
        ix = ix[ix.index >= pd.Timestamp(cfg["data"]["start"])]
        ix = ix[ix.index <= pd.Timestamp(cfg["data"]["end"])]
        g = (ix["open"] / ix["close"].shift(1) - 1).abs() <= cfg["claims"]["gap_tol"]
        rows.append({"name": f"지수 {name}", "days": len(ix) - 1, "gap_share": float(g.iloc[1:].mean())})
    return pd.DataFrame(rows)


def verdict_c1(tab: pd.DataFrame, cl: dict) -> str:
    v = tab[(tab["variant"] == "no_gap") & tab["ticker"].isin(cl["c1_tickers"])]
    if len(v) != len(cl["c1_tickers"]):
        return "재현 안 됨"  # missing verdict-ticker rows must not read as a pass (empty .all() == True)
    strong = ((v["bottom_mean"] > 0) & (v["bottom_t"] >= cl["t_strong"]) & (v["rank_corr"] < 0) & (v["split_half"] > 0)).all()
    if strong:
        return "재현됨"
    if (v["bottom_mean"] > 0).all() and (v["bottom_t"] >= cl["t_weak"]).any():
        return "약함"
    return "재현 안 됨"


def c1(cfg: dict, sm: dict[str, pd.DataFrame]) -> dict:
    cl = cfg["claims"]
    rows, dec = [], []
    for t in cl["c1_tickers"] + cl["leveraged"]:
        for var, fw in variants(sm[t]):
            d = decile_table(fw)
            dec.append(d.assign(ticker=t, variant=var))
            b, tp = tstat(fw.loc[fw["bottom"], "oc"]), tstat(fw.loc[fw["top"], "oc"])
            rows.append({"ticker": t, "variant": var, "days": len(fw), "gap_share": float(sm[t]["gap"].mean()),
                         "bottom_mean": b["mean"], "bottom_t": b["t"], "bottom_n": b["n"], "top_mean": tp["mean"],
                         "rank_corr": rank_corr(d["mean"]), "split_half": split_half(fw)})
    tab = pd.DataFrame(rows)
    notes = [f"- 판정은 {', '.join(cl['c1_tickers'])} 의 no_gap 행으로 한다. 재현됨 = 둘 다 하위 20% 평균 > 0, t ≥ {cl['t_strong']}, "
             f"10분위 상관 < 0, 앞뒤 절반 일치도 > 0. 약함 = 둘 다 평균 > 0 이고 하나라도 t ≥ {cl['t_weak']}.",
             "- 레버리지 ETF 행은 참고용이다(상장 뒤 구간만)."]
    return {"table": tab, "verdict": verdict_c1(tab, cl), "notes": notes, "deciles": pd.concat(dec, ignore_index=True)}


def verdict_c2(tab: pd.DataFrame, cl: dict) -> str:
    v = tab[tab["variant"] == "no_gap"].set_index("ticker")
    required = cl["equity"] + cl["non_equity"]
    if not set(required).issubset(v.index):
        return "재현 안 됨"  # missing verdict-ticker rows must not read as a pass (empty .all() == True)
    eq_t = v.loc[cl["equity"], "bottom_t"]
    non_t = v.loc[cl["non_equity"], "bottom_t"]
    if (eq_t > cl["t_strong"]).sum() >= cl["c2_min_equity"] and (non_t < cl["t_strong"]).all():
        return "재현됨"
    if eq_t.median() > non_t.max():
        return "약함"
    return "재현 안 됨"


def c2(cfg: dict, sm: dict[str, pd.DataFrame]) -> dict:
    cl = cfg["claims"]
    kind = {**{t: "주식" for t in cl["equity"]}, **{t: "채권·금" for t in cl["non_equity"]},
            **{t: "레버리지" for t in cl["leveraged"]}}
    rows = []
    for t in cl["tickers"] + cl["leveraged"]:
        for var, fw in variants(sm[t]):
            b, tp = tstat(fw.loc[fw["bottom"], "oc"]), tstat(fw.loc[fw["top"], "oc"])
            rows.append({"ticker": t, "class": kind[t], "variant": var, "bottom_mean": b["mean"], "bottom_t": b["t"],
                         "bottom_n": b["n"], "top_mean": tp["mean"], "spread_mean": b["mean"] - tp["mean"]})
    tab = pd.DataFrame(rows)
    notes = [f"- 재현됨 = 주식 ETF {len(cl['equity'])}개 중 {cl['c2_min_equity']}개 이상 t > {cl['t_strong']} 이고 TLT, GLD 둘 다 t < {cl['t_strong']}. "
             "약함 = 주식 t 의 중앙값이 TLT, GLD 의 t 보다 크다.",
             "- 원본 표와 같이 비레버리지 13개(주식 11개 + TLT, GLD)다. FXI 는 지시서 목록에 없었으나 사용자 결정으로 넣었다. 레버리지 ETF 는 판정에 넣지 않는다."]
    return {"table": tab, "verdict": verdict_c2(tab, cl), "notes": notes}


def verdict_c3(tab: pd.DataFrame, cl: dict) -> str:
    v = tab[(tab["variant"] == "no_gap") & tab["ticker"].isin(cl["c1_tickers"])]
    if len(v) != len(cl["c1_tickers"]):
        return "재현 안 됨"  # missing verdict-ticker rows must not read as a pass (empty .all() == True)
    ordered = (v["below_mean"] > v["above_mean"]).all()
    if ordered and (v["below_t"] >= cl["t_strong"]).all() and (v["above_t"] < cl["t_strong"]).all():
        return "재현됨"
    return "약함" if ordered else "재현 안 됨"


def c3(cfg: dict, sm: dict[str, pd.DataFrame]) -> dict:
    cl = cfg["claims"]
    w = cl["index_ma"]
    above = {}
    for name in ("SPX", "NDX"):
        c = index_close(cfg, name)
        ma = c.rolling(w, min_periods=w).mean()
        above[name] = (c > ma).where(ma.notna())  # NaN until the MA is ready
    rows = []
    for t in cl["c3_tickers"]:
        name = cl["index_of"].get(t, "SPX")
        for var, fw in variants(sm[t]):
            flag = above[name].reindex(fw.index, method="ffill")  # the signal day's close (known before the trade)
            x = fw[fw["bottom"] & flag.notna()]
            up = x[flag[x.index].astype(bool)]
            dn = x[~flag[x.index].astype(bool)]
            a, b = tstat(up["oc"]), tstat(dn["oc"])
            rows.append({"ticker": t, "index": name, "variant": var, "above_mean": a["mean"], "above_t": a["t"],
                         "above_n": a["n"], "below_mean": b["mean"], "below_t": b["t"], "below_n": b["n"],
                         "below_share": b["n"] / max(1, a["n"] + b["n"])})
    tab = pd.DataFrame(rows)
    notes = [f"- 지수 {w}일선 위·아래는 신호일 종가 기준(다음 날 시가에 사기 전에 안다). 나스닥 계열(QQQ, SOXX, TQQQ, SOXL)은 NDX, 나머지는 SPX.",
             f"- 재현됨 = {', '.join(cl['c1_tickers'])} 둘 다 아래 평균 > 위 평균, 아래 t ≥ {cl['t_strong']}, 위 t < {cl['t_strong']}. 약함 = 방향만 같다."]
    return {"table": tab, "verdict": verdict_c3(tab, cl), "notes": notes}


def verdict_c4(tab: pd.DataFrame, cl: dict) -> str:
    v = tab[(tab["variant"] == "no_gap") & tab["ticker"].isin(cl["c1_tickers"])]
    if len(v) != len(cl["c1_tickers"]):
        return "재현 안 됨"  # missing verdict-ticker rows must not read as a pass (empty .all() == True)
    if (v["overnight_t"] >= cl["t_strong"]).all():
        return "재현됨"
    if (v["overnight_mean"] > 0).all() and (v["overnight_t"] >= cl["t_weak"]).any():
        return "약함"
    return "재현 안 됨"


def c4(cfg: dict, sm: dict[str, pd.DataFrame]) -> dict:
    cl = cfg["claims"]
    rows = []
    for t in cl["tickers"] + cl["leveraged"]:
        for var, fw in variants(sm[t]):
            o, d = tstat(fw.loc[fw["bottom"], "on"]), tstat(fw.loc[fw["bottom"], "oc"])
            rows.append({"ticker": t, "variant": var, "overnight_mean": o["mean"], "overnight_t": o["t"],
                         "overnight_n": o["n"], "next_day_mean": d["mean"]})
    tab = pd.DataFrame(rows)
    notes = ["- 밤사이 = 하위 20% 날 종가 → 다음 날 시가. 시가 = 전날 종가인 날은 밤사이 수익이 0 이라 all 행을 낮춘다.",
             f"- 재현됨 = {', '.join(cl['c1_tickers'])} 둘 다 no_gap 밤사이 t ≥ {cl['t_strong']}."]
    return {"table": tab, "verdict": verdict_c4(tab, cl), "notes": notes}


def verdict_c5(single: pd.DataFrame, basket: dict, corr: float, cl: dict) -> str:
    if not set(cl["basket"]).issubset(single.index):
        return "재현 안 됨"  # missing verdict-ticker rows must not read as a pass (empty .all() == True)
    better_sharpe = basket["sharpe"] > single["sharpe"].mean()
    if better_sharpe and basket["mdd"] > single["mdd"].mean() and corr < cl["c5_max_corr"]:
        return "재현됨"
    return "약함" if better_sharpe else "재현 안 됨"


def c5(cfg: dict, sm: dict[str, pd.DataFrame]) -> dict:
    """Same rule on each sleeve (open fills, no costs, no interest, full weight), equal weight
    rebalanced daily = the mean of the six daily returns."""
    cl = cfg["claims"]
    cap = capital_usd(cfg)
    d = {}
    for t, f in frames(cfg, cl["basket"]).items():
        cal = calendar(cfg, f)
        tr = trades(cfg, f, t, "open", with_costs=False)
        eq, _ = account(cal, tr, f["close"], cap, np.ones(len(tr)), np.ones(len(cal)))
        d[t] = daily_returns(eq, cap)
    d = pd.DataFrame(d).dropna()
    single = pd.DataFrame({t: stats(d[t]) for t in d}).T
    basket = stats(d.mean(axis=1))
    cm = d.corr().to_numpy()
    corr = float(cm[np.triu_indices_from(cm, 1)].mean())
    tab = single[["cagr", "sharpe", "mdd"]].rename_axis("sleeve").reset_index()
    tab = pd.concat([tab, pd.DataFrame([
        {"sleeve": "단일 평균", "cagr": single["cagr"].mean(), "sharpe": single["sharpe"].mean(), "mdd": single["mdd"].mean()},
        {"sleeve": "동일 비중 바구니", "cagr": basket["cagr"], "sharpe": basket["sharpe"], "mdd": basket["mdd"]}])],
        ignore_index=True)
    tab["mean_corr"] = corr
    notes = [f"- 기간 {d.index[0].date()} → {d.index[-1].date()}, open 체결, 비용 0, 현금 이자 0, 전액. 바구니는 날마다 같은 비중으로 맞춘다.",
             f"- 상관 = 여섯 전략 일별 수익률의 쌍별 상관 15개 평균 {corr:.2f}.",
             f"- 재현됨 = 바구니 샤프 > 단일 평균, 바구니 낙폭이 단일 평균보다 얕다, 상관 < {cl['c5_max_corr']}. 약함 = 샤프만 낫다."]
    return {"table": tab, "verdict": verdict_c5(single, basket, corr, cl), "notes": notes}


CLAIMS = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


def write(cfg: dict, sm: dict[str, pd.DataFrame], res: dict[str, dict]) -> None:
    rd = results_dir()
    orig = cfg["claims"]["original"]
    pd.concat([r["table"].assign(claim=n) for n, r in res.items()], ignore_index=True).to_csv(
        rd / "claims.csv", index=False, encoding="utf-8-sig")
    dec = [r["deciles"].assign(claim=n) for n, r in res.items() if "deciles" in r]
    if dec:
        pd.concat(dec, ignore_index=True).to_csv(rd / "claims_deciles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"claim": n, "title": orig[n]["claim"], "verdict": r["verdict"], "original": orig[n]["value"]}
                  for n, r in res.items()]).to_csv(rd / "claims_verdicts.csv", index=False, encoding="utf-8-sig")
    gt = gap_table(cfg, sm)
    gt["gap_share"] = gt["gap_share"].map(share)
    out = ["# 단계 1: 원본의 다섯 가지 주장", "",
           f"표본 {cfg['data']['start']} → {cfg['data']['end']} (이 날 포함, 레버리지 ETF 는 상장일부터). "
           "측정 단위: IBS 하위 20% 날의 다음 날 시가 → 종가 수익률과 t값. 판정은 시가 = 전날 종가인 날을 뺀 결과(no_gap)로 한다.", "",
           "## 시가 = 전날 종가인 날의 비율", "", md_table(gt), "",
           "- 이 판정은 조정 전 원본 시가/종가로 한다(배당 조정 비율이 하루 ~1e-7 씩 움직여, 조정가로는 배당을 주는 "
           "종목에서 실제 갭 날을 놓친다). no_gap 변형은 십분위 절단점(20%, 80% 분위수, 10분위 경계)을 전체 표본(all)으로 "
           "구한 뒤 시가 = 전날 종가인 날만 뺀 것이다 — 절단점 자체를 다시 구하지 않는다.", ""]
    for n, r in res.items():
        out += [f"## {n} {orig[n]['claim']}: **{r['verdict']}**", "", f"원본: {orig[n]['value']}", "",
                md_table(fmt_table(r["table"])), "", *r["notes"], ""]
        if n == "C1" and "deciles" in r:
            cl = cfg["claims"]
            dec = r["deciles"]
            dv = dec[(dec["variant"] == "no_gap") & dec["ticker"].isin(cl["c1_tickers"])]
            piv = dv.pivot(index="decile", columns="ticker", values="mean").reindex(columns=cl["c1_tickers"])
            dt = pd.DataFrame({"10분위": piv.index, **{t: piv[t].map(lambda v: pct(v, 3)) for t in cl["c1_tickers"]}})
            out += [f"### {', '.join(cl['c1_tickers'])} no_gap 10분위별 평균 (다음 날 시가 → 종가 수익률)", "",
                    md_table(dt), ""]
    (rd / "claims.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = load_config()
    cl = cfg["claims"]
    sm = sample(cfg, cl["tickers"] + cl["leveraged"])
    names = a.only or list(CLAIMS)
    res = {n: CLAIMS[n](cfg, sm) for n in names}
    write(cfg, sm, res)
    for n, r in res.items():
        print(f"{n}: {r['verdict']}")
        print(r["table"].round(5).to_string())


if __name__ == "__main__":
    main()
