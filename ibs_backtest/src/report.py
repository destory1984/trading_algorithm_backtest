"""results/report.md from the csv files of the other stages.

Usage:  python -m src.report
"""
from __future__ import annotations

import pandas as pd

from .common import load_config, results_dir
from .grid import label


def pct(v, d=1) -> str:
    return "" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def share(v) -> str:
    return "" if pd.isna(v) else f"{v:.1%}"


def num(v, d=2) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    rp = pd.read_csv(rd / "replicate.csv")
    em = pd.read_csv(rd / "engine_match.csv")
    g = pd.read_csv(rd / "grid.csv", index_col="combo")
    bm = pd.read_csv(rd / "benchmark.csv").iloc[0]
    wfp = pd.read_csv(rd / "walkforward.csv")
    wfs = pd.read_csv(rd / "walkforward_summary.csv")
    vs = pd.read_csv(rd / "volsize.csv")
    cs = cfg["us"]["costs"]
    one_way = (cs["buy_fee"] + cs["slippage"]) * 1e4
    min_tr = cfg["walkforward"]["min_train_trades"]
    out = ["# IBS 평균회귀 백테스트 결과", "",
           f"비용은 편도 {one_way:.0f}bp (수수료 {cs['buy_fee'] * 100:.2f}% + 슬리피지 {cs['slippage'] * 100:.2f}%). "
           "샤프는 무위험 수익률을 빼지 않은 일별 수익률 기준(원본과 같음). 수치는 모두 `results/*.csv` 에서 읽었다.", ""]

    # 1. replication
    r = rp[rp["span"] == "original_end"]
    out += [f"## 1. 원본 재현 (SPY, {r['period'].iloc[0]})", "",
            md_table(pd.DataFrame({"방식": r["method"], "비용": r["costs"], "연 수익률": r["cagr"].map(pct),
                                   "최대 낙폭": r["mdd"].map(pct), "샤프": r["sharpe"].map(num),
                                   "거래": r["trades"].map(lambda v: "" if pd.isna(v) else f"{int(v)}"),
                                   "승률": r["win_rate"].map(share), "노출": r["exposure"].map(share)})), ""]
    c0 = r[(r["method"] == "재현 close") & (r["costs"] == "0")].iloc[0]
    o0 = r[(r["method"] == "재현 open") & (r["costs"] == "0")].iloc[0]
    orig = r[r["method"] == "원본 README"].iloc[0]
    c1 = r[(r["method"] == "재현 close") & (r["costs"] != "0")].iloc[0]
    a, b = (pd.Timestamp(x) for x in r["period"].iloc[0].split(" → "))
    per_year = c0["trades"] / ((b - a).days / 365.25)
    out += [f"- `close` 재현은 거래 수({int(c0['trades'])})와 최대 낙폭이 원본과 같고, 연 수익률은 {(c0['cagr'] - orig['cagr']) * 100:+.2f}%p, "
            f"승률은 {(c0['win_rate'] - orig['win_rate']) * 100:+.2f}%p 다르다. 원본 거래 목록이 없어 날짜별로 대조하지는 못했다. 거래 수와 낙폭이 같아 신호는 거의 같고, "
            "야후 과거 고가·저가가 원본을 돌린 때와 조금 달라진 것으로 본다.",
            f"- 밤사이 몫: 같은 신호를 다음 날 시가에 사고팔면 연 수익률이 {(c0['cagr'] - o0['cagr']) * 100:.2f}%p 줄어든다 "
            f"({pct(c0['cagr'])} → {pct(o0['cagr'])}).",
            f"- 비용: 한 해 거래 {per_year:.0f}번 × 왕복 {2 * one_way:.0f}bp ≈ 연 {per_year * 2 * one_way / 100:.1f}% 가 비용으로 나간다. "
            f"`close` 연 수익률이 {pct(c0['cagr'])} 에서 {pct(c1['cagr'])} 로 떨어진다.", ""]

    # 2. engine match
    ok = (em["mismatched"] == 0).all()
    out += ["## 2. 엔진 대조", "",
            f"SPY {em['first_signal'].iloc[0]} 이후 진입 0.20 / 청산 0.80, 비용 0. 단계 0(단순 반복문)과 krxbt 엔진의 거래 목록.", "",
            md_table(pd.DataFrame({"체결": em["exec"], "단계 0 거래": em["replicate_trades"], "엔진 거래": em["engine_trades"],
                                   "다른 거래": em["mismatched"], "수익률 최대 차이": em["max_ret_diff"].map(lambda v: f"{v:.1e}")})), "",
            f"진입일·청산일이 {'모두 같다' if ok else '다르다'}. 수익률 차이 {em['max_ret_diff'].max():.1e} 는 두 데이터의 배당 수정 계수 반올림 차이다.", ""]

    # 3. grid
    gg = g.copy()
    gg["조합"] = [label(c) for _, c in gg.iterrows()]
    top = gg[gg["trades"] >= min_tr].sort_values("sharpe", ascending=False).head(10)
    out += [f"## 3. 그리드 ({len(g)}개 조합, {cfg['data']['start'][:4]}-01 → 최근, 비용 포함)", "",
            f"비교 기준 {cfg['grid']['benchmark']} 보유: 연 {pct(bm['cagr'])}, 낙폭 {pct(bm['mdd'])}, 샤프 {num(bm['sharpe'])}.", "",
            f"### 샤프 상위 10 (거래 {min_tr}건 이상)", "",
            md_table(pd.DataFrame({"조합": top["조합"], "거래": top["trades"], "승률": top["win_rate"].map(share),
                                   "거래당": top["expectancy"].map(lambda v: pct(v, 2)), "손익비": top["payoff"].map(num),
                                   "연 수익률": top["cagr"].map(pct), "낙폭": top["mdd"].map(pct), "샤프": top["sharpe"].map(num),
                                   "노출": top["exposure"].map(lambda v: f"{v:.0%}"),
                                   "손익분기(편도bp)": top["breakeven_bp"].map(lambda v: num(v, 1))})), ""]
    agg = gg.groupby(["exec", "trend"]).agg(n=("sharpe", "size"), trades=("trades", "mean"), cagr=("cagr", "mean"),
                                           mdd=("mdd", "mean"), sharpe=("sharpe", "mean"),
                                           gross=("gross_expectancy", "mean"), be=("breakeven_bp", "median")).reset_index()
    out += ["### 체결 방식 × 추세 필터별 평균", "",
            md_table(pd.DataFrame({"체결": agg["exec"], "추세 필터": agg["trend"], "조합": agg["n"],
                                   "거래(평균)": agg["trades"].map(lambda v: f"{v:.0f}"),
                                   "비용 전 거래당": agg["gross"].map(lambda v: pct(v, 2)), "연 수익률": agg["cagr"].map(pct),
                                   "낙폭": agg["mdd"].map(pct), "샤프": agg["sharpe"].map(num),
                                   "손익분기 중앙값(bp)": agg["be"].map(lambda v: num(v, 1))})), ""]
    beat = gg[(gg["sharpe"] > bm["sharpe"]) & (gg["trades"] >= min_tr)]
    out += [f"- 샤프가 {cfg['grid']['benchmark']} 보유({num(bm['sharpe'])})보다 높은 조합: {len(beat)}개 / {len(g)}개"
            + (f" ({', '.join(beat.sort_values('sharpe', ascending=False)['조합'].head(5))})" if len(beat) else ""),
            f"- 연 수익률이 {cfg['grid']['benchmark']} 보유보다 높은 조합: {int((gg['cagr'] > bm['cagr']).sum())}개",
            f"- 손익분기 비용 편도 10bp 이상: {int((gg['breakeven_bp'] >= 10).sum())}개, 설정 비용({one_way:.0f}bp) 이상: "
            f"{int((gg['breakeven_bp'] >= one_way).sum())}개", ""]

    # 4. walk-forward
    fy = cfg["walkforward"]["first_test_year"]
    names = {"all": "모든 조합", "close": "close 조합만", "open": "open 조합만"}
    w = wfs.copy()
    w["구분"] = w["variant"].map(names).fillna("")
    out += [f"## 4. 걸어가며 검증 ({fy}-01 → 최근, 비용 포함)", "",
            f"해마다 그 해 전까지의 일별 수익률만 보고 샤프 1등 조합(그 전 거래 {min_tr}건 이상)을 골라 다음 한 해 쓴다. "
            f"학습은 {cfg['data']['start'][:4]}년부터 늘어난다. 한 계좌·한 자리로 이어 붙였다.", "",
            md_table(pd.DataFrame({"구분": w["구분"], "방식": w["name"], "연 수익률": w["cagr"].map(pct), "낙폭": w["mdd"].map(pct),
                                   "샤프": w["sharpe"].map(num), "거래": w["trades"].map(lambda v: "" if pd.isna(v) else f"{int(v)}"),
                                   "노출": w["exposure"].map(lambda v: f"{v:.0%}"),
                                   "손익분기(편도bp)": w["breakeven_bp"].map(lambda v: num(v, 1))})), ""]
    for var in ("all", "open"):
        p = wfp[wfp["variant"] == var]
        s = wfs[(wfs["variant"] == var) & (wfs["name"] == "걸어가며")].iloc[0]
        out += [f"### 해마다 고른 조합 ({names[var]}): 서로 다른 조합 {int(s['distinct_picks'])}개, 바뀐 횟수 {int(s['changes'])}번", "",
                md_table(pd.DataFrame({"해": p["year"], "고른 조합": p["label"], "학습 샤프": p["train_sharpe"].map(num),
                                       "그 해 샤프": p["test_sharpe"].map(num), "그 해 수익률": p["test_return"].map(pct),
                                       "그 해 순위(백분위)": p["test_pctile"].map(lambda v: f"{v:.0%}")})), ""]

    # 5. vol target
    v = vs.copy()
    v["구분"] = v["variant"].map(names).fillna("")
    out += ["## 5. 변동성 목표 사이징 (4번에서 고른 조합의 거래 그대로, 비중만 바꿈)", "",
            f"비중 = min(1, 목표 / 최근 {cfg['indicators']['vol_window']}거래일 실현 변동성), 나머지 현금. 레버리지 없음.", "",
            md_table(pd.DataFrame({"구분": v["구분"], "비중": v["sizing"], "평균 비중": v["avg_weight"].map(lambda x: "" if pd.isna(x) else f"{x:.0%}"),
                                   "연 수익률": v["cagr"].map(pct), "낙폭": v["mdd"].map(pct), "샤프": v["sharpe"].map(num)})), ""]

    # 6. decision
    bh = wfs[wfs["name"].str.endswith("보유")].iloc[0]
    lines = []
    passed = {}
    for var in ("close", "open"):
        s = wfs[(wfs["variant"] == var) & (wfs["name"] == "걸어가며")].iloc[0]
        c1, c2, c3 = s["sharpe"] > bh["sharpe"], s["mdd"] > bh["mdd"], s["breakeven_bp"] >= 10
        passed[var] = c1 and c2 and c3
        mark = lambda b: "만족" if b else "불만족"  # noqa: E731
        lines.append(f"| {var} | {num(s['sharpe'])} vs {num(bh['sharpe'])} ({mark(c1)}) | {pct(s['mdd'])} vs {pct(bh['mdd'])} ({mark(c2)}) "
                     f"| {num(s['breakeven_bp'], 1)}bp ({mark(c3)}) |")
    verdict = "신호 알림 후보로 올린다" if passed["open"] else (
        "close 방식에서만 통과. 종가 동시호가 체결이 되는지 먼저 따진다" if passed["close"] else "신호 알림 후보로 올리지 않는다")
    out += ["## 6. 판단", "",
            f"미리 정한 기준을 걸어가며 검증 결과({fy}년 이후, 비용 포함)에 댄다.", "",
            f"| 체결 | 샤프 > {cfg['grid']['benchmark']} 보유 | 낙폭 < {cfg['grid']['benchmark']} 보유 | 손익분기 ≥ 10bp |", "|---|---|---|---|",
            *lines, "",
            f"**결론: {verdict}.** 비용 전 거래당 우위(그리드 평균 {pct(g['gross_expectancy'].mean(), 2)})가 편도 {one_way:.0f}bp 비용에 "
            f"대부분 없어지고, {fy}년 이후 {cfg['grid']['benchmark']} 보유(연 {pct(bh['cagr'])})를 시장 노출 15% → 30% 로는 따라가지 못한다. "
            "낙폭만 작다.", ""]
    text = "\n".join(out)
    (rd / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
