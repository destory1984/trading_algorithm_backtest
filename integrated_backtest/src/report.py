"""The report: verdicts, Korean and US details, integrated comparison.
Run after src.kr.evaluate, src.us.evaluate and src.compare.
Usage: python -m src.report   -> results/report.md
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.kr import build as B

KR_NAMES = {"A2": "A2 코너스 RSI(2)", "BO": "B-O 오닐 기준", "BM": "B-M 미너비니 템플릿", "C": "C 변동성 돌파"}
KR_CRIT = {1: "DSR ≥ 0.95", 2: "무작위 95%", 3: "낙폭 맞춘 코스피", 4: "이웃 2/3", 5: "슬리피지 2배", 6: "상위 5일 빼도 +"}
US_CRIT = {1: "낙폭 맞춘 보유", 2: "두 기간", 3: "이웃", 4: "비용 2배", 5: "합성 구간"}
STRETCH = {"full": "전체", "train": "학습 2015 → 2020", "test": "검증 2021 →"}


def ox(b) -> str:
    return "" if pd.isna(b) else ("O" if bool(b) else "X")


def main() -> None:
    cfg = C.load_config()
    kd, ud, cd = C.results_dir("kr"), C.results_dir("us"), C.results_dir("compare")
    kv, km = pd.read_csv(kd / "verdicts.csv"), pd.read_csv(kd / "metrics.csv")
    kdsr, krnd = pd.read_csv(kd / "dsr.csv"), pd.read_csv(kd / "random.csv")
    gs = pd.read_csv(kd / "grid_summary.csv")
    uv, um = pd.read_csv(ud / "verdicts.csv"), pd.read_csv(ud / "metrics.csv")
    u1s = pd.read_csv(ud / "u1_stats.csv")
    L = ["# 통합 백테스트 결과", "",
         f"한국 {cfg['data']['start']} → {cfg['data']['end']}(학습 → {cfg['data']['train_end']}, 검증 그 뒤), 미국은 자산마다 첫날 → "
         f"{cfg['us']['end']}. 지시서 SPEC.md(원문 ORIGINAL_SPEC.md).", "", "## 판정 요약", ""]
    rows = []
    for x in kv.itertuples():
        f = km[(km["strategy"] == x.strategy) & (km["run"] == "main") & (km["stretch"] == "full")].iloc[0]
        rows.append({"전략": KR_NAMES[x.strategy], "시장": "한국", "연 수익률": C.pct(f["cagr"]), "최대 낙폭": C.pct(f["mdd"]),
                     "비교 기준 대비": C.pp(f["diff_risk"], 2), "판정": "통과" if x.passed else "탈락",
                     "못 넘은 기준": x.failed if isinstance(x.failed, str) else ""})
    main_run = {"U1 TQQQ": "U1|TQQQ|40|0.1", "U1 SOXL": "U1|SOXL|40|0.12", "U2 TQQQ": "U2|TQQQ|0.15",
                "U2 SOXL": "U2|SOXL|0.15", "U3-G": "U3G|12", "U3-A": "U3A|TLT"}
    for x in uv.itertuples():
        f = um[(um["run"] == main_run[x.row]) & (um["stretch"] == "full")].iloc[0]
        rows.append({"전략": x.row, "시장": "미국", "연 수익률": C.pct(f["cagr"]), "최대 낙폭": C.pct(f["mdd"]),
                     "비교 기준 대비": C.pp(f["diff_risk"], 2), "판정": "통과" if x.passed else "탈락",
                     "못 넘은 기준": x.failed if isinstance(x.failed, str) else ""})
    L += [C.md_table(pd.DataFrame(rows)), "",
          "비교 기준: 한국은 낙폭을 맞춘 코스피(총수익 근사, 코스피 w + 현금), 미국 U1·U2 는 낙폭을 맞춘 같은 ETF 보유, U3 는 낙폭을 맞춘 SPY 보유.", ""]

    # Korea
    L += ["## 한국 전략", ""]
    for x in kv.itertuples():
        s = x.strategy
        m = km[km["strategy"] == s]
        get = lambda run, st: m[(m["run"] == run) & (m["stretch"] == st)].iloc[0]
        d = kdsr[kdsr["strategy"] == s].iloc[0]
        det = {1: f"DSR {d['dsr']:.3f} (진입일 샤프 {d['sr']:+.3f}, 기대 최대 {d['sr0']:.3f}, 진입일 {int(d['T'])}개)",
               2: f"연 {C.pct(get('main', 'full')['cagr'])}, 무작위 200번 중앙값 {C.pct(x.random_median)}, 95 백분위 {C.pct(x.random_p95)} → {x.random_pctile:.0%} 위치",
               3: " / ".join(f"{STRETCH[st]} {C.pp(get('main', st)['diff_risk'])}" for st in STRETCH),
               4: f"{x.neighbors_ok} 칸이 기준 3 전체 만족: " + ", ".join(f"{r.run} {C.pp(r.diff_risk)}" for r in m[(m['stretch'] == 'full') & m['run'].str.startswith('nb_')].itertuples()),
               5: f"슬리피지 2배 연 {C.pct(get('main_slip2', 'full')['cagr'])}, 대비 {C.pp(get('main_slip2', 'full')['diff_risk'])}",
               6: f"그리드 거래(자리 제한 없음) {int(d['trades'])}건, 상위 5일을 뺀 거래당 평균 {C.pct(d['best_days_removed_mean'], 2)}"}
        L += [f"### {KR_NAMES[s]}: {'통과' if x.passed else '탈락'}", "",
              C.md_table(pd.DataFrame([{"기준": f"{i} {KR_CRIT[i]}", "결과": ox(getattr(x, f'c{i}')), "내용": det[i]} for i in KR_CRIT])), ""]
    L += ["### 포트폴리오 표 (1억 원, 5자리)", "",
          C.md_table(pd.DataFrame([{"전략": KR_NAMES[r.strategy], "판본": r.run, "구간": STRETCH[r.stretch], "연 수익률": C.pct(r.cagr),
                                    "최대 낙폭": C.pct(r.mdd), "샤프": C.num(r.sharpe), "거래": r.trades if r.stretch == "full" else "",
                                    "승률": C.pct(r.win, 0) if r.stretch == "full" else "",
                                    "거래당": C.pct(r.avg_ret, 2) if r.stretch == "full" else "",
                                    "투자 비율": f"{r.invested:.0%}" if r.stretch == "full" else "", "코스피": C.pct(r.kospi_cagr),
                                    "낙폭 맞춘 코스피(w)": f"{C.pct(r.risk_cagr)} ({r.risk_w:.2f})", "그 대비": C.pp(r.diff_risk)}
                                   for r in km.itertuples()])), "",
          "main_turtle 은 터틀식 비중(종목당 위험 1%, 20% 상한)으로 돌린 민감도로 판정에 쓰지 않는다.", ""]
    L += ["### 무작위 진입 200번", "",
          C.md_table(krnd.groupby("strategy")["cagr"].describe(percentiles=[0.05, 0.5, 0.95]).reset_index()
                     .rename(columns={"strategy": "전략"})[["전략", "min", "5%", "50%", "95%", "max"]]
                     .assign(**{c: (lambda c: lambda d: d[c].map(C.pct))(c) for c in ["min", "5%", "50%", "95%", "max"]})), ""]
    # grid landscape
    L += ["### 그리드 지형 (거래 단위, 자리 제한 없음)", ""]
    grids = {"A2": B.a2_grid(cfg), "B": B.b_grid(cfg), "C": B.c_grid(cfg)}
    for gname, grid in grids.items():
        x = gs[(gs["strategy"] == gname) & (gs["market"] == "ALL")].copy()
        x["combo_desc"] = x["combo"].map(lambda i: ", ".join(f"{k}={v}" for k, v in grid[int(i)].items()))
        top = x[x["trades"] >= 200].sort_values("mean", ascending=False).head(5)
        L += [f"#### {gname}: 합산 시장, 거래 200건 이상 중 거래당 평균 상위 5 (결과를 보고 고른 것이라 판정하지 않는다)", "",
              C.md_table(pd.DataFrame([{"조합": r.combo_desc, "거래": r.trades, "승률": C.pct(r.win, 0), "평균": C.pct(r.mean, 2),
                                        "중앙값": C.pct(r.median, 2), "손익비": C.num(r.pf), "평균 보유일": C.num(r.hold, 1)}
                                       for r in top.itertuples()])), "",
              f"합산 시장 {len(x)}칸 중 거래당 평균이 0 보다 큰 칸 {int((x['mean'] > 0).sum())}개, 중앙값 칸의 평균 {C.pct(x['mean'].median(), 2)}.", ""]
        if gname == "B":
            x["filter"] = x["combo"].map(lambda i: grid[int(i)]["filter"])
            x["stop"] = x["combo"].map(lambda i: str(grid[int(i)]["stop"]))
            x["regime"] = x["combo"].map(lambda i: grid[int(i)]["regime"])
            L += ["B 칸들을 필터·손절·국면별로 묶은 거래당 평균의 평균:", "",
                  C.md_table(pd.concat([x.groupby(k)["mean"].mean().rename("평균").reset_index().rename(columns={k: "값"}).assign(묶음=k)
                                        for k in ("filter", "stop", "regime")])[["묶음", "값", "평균"]]
                             .assign(평균=lambda d: d["평균"].map(lambda v: C.pct(v, 2)))), ""]
    # US
    L += ["## 미국 전략", ""]
    for x in uv.itertuples():
        run = main_run[x.row]
        m = um[um["run"] == run]
        f, a, b = (m[m["stretch"] == s].iloc[0] for s in ("full", "first", "second"))
        det = {1: f"연 {C.pct(f['cagr'])}, 최대 낙폭 {C.pct(f['mdd'])}; 보유 {C.pct(f['hold_cagr'])} / {C.pct(f['hold_mdd'])}; 낙폭 맞춘 보유(w {f['risk_w']:.2f}) {C.pct(f['risk_cagr'])} → {C.pp(f['diff_risk'])}",
               2: f"앞 {a['first_day']} → {a['last_day']} {C.pp(a['diff_risk'])}, 뒤 → {b['last_day']} {C.pp(b['diff_risk'])}",
               3: f"이웃 만족 {x.neighbors_ok if isinstance(getattr(x, 'neighbors_ok', None), str) else ''}",
               4: "", 5: ""}
        cost_run = {"U1 TQQQ": "U1|TQQQ|main_cost2", "U1 SOXL": "U1|SOXL|main_cost2", "U2 TQQQ": "U2|TQQQ|main_cost2",
                    "U2 SOXL": "U2|SOXL|main_cost2", "U3-G": "U3G|main_cost2", "U3-A": "U3A|main_cost2"}[x.row]
        cm = um[(um["run"] == cost_run) & (um["stretch"] == "full")].iloc[0]
        det[4] = f"비용 2배 연 {C.pct(cm['cagr'])}, 대비 {C.pp(cm['diff_risk'])}"
        if x.row == "U1 TQQQ":
            sm = um[(um["run"] == "U1|SYNTH|main") & (um["stretch"] == "full")].iloc[0]
            det[5] = (f"합성 {sm['first_day']} → {sm['last_day']}: 연 {C.pct(sm['cagr'])}, 낙폭 {C.pct(sm['mdd'])}; 합성 보유 {C.pct(sm['hold_cagr'])} / "
                      f"{C.pct(sm['hold_mdd'])}; 대비 {C.pp(sm['diff_risk'])}. 합성 오차 {C.pp(x.synth_gap)}(한도 2%p) → 합성 오차 큼")
        if x.row.startswith("U3"):
            nbr = m.iloc[0:0]
            runs = [r for r in um["run"].unique() if r.startswith(run.split("|")[0] + "|") and r != run and "cost2" not in r]
            det[3] = ", ".join(f"{r} {C.pp(um[(um['run'] == r) & (um['stretch'] == 'full')].iloc[0]['diff_risk'])}" for r in runs)
        elif x.row.startswith("U2"):
            tk = x.row.split()[1]
            det[3] = ", ".join(f"밴드 {b_:.0%} {C.pp(um[(um['run'] == f'U2|{tk}|{b_}') & (um['stretch'] == 'full')].iloc[0]['diff_risk'])}"
                               for b_ in cfg["us"]["U2"]["neighbors_band"])
        crit = [i for i in US_CRIT if f"c{i}" in uv.columns and not pd.isna(getattr(x, f"c{i}"))]
        L += [f"### {x.row}: {'통과' if x.passed else '탈락'}", "",
              C.md_table(pd.DataFrame([{"기준": f"{i} {US_CRIT[i]}", "결과": ox(getattr(x, f'c{i}')), "내용": det[i]} for i in crit])), ""]
    g = um[(um["strategy"] == "U1") & (um["stretch"] == "full") & um["splits"].notna()]
    for tk in cfg["us"]["U1"]["tickers"]:
        t = g[g["ticker"] == tk].pivot(index="splits", columns="target", values="diff_risk")
        t.index = t.index.astype(int)
        L += [f"#### U1 {tk} 그리드: 낙폭 맞춘 보유 대비 연 수익률 차이", "",
              C.md_table(t.map(C.pp).reset_index().rename(columns={"splits": "분할"}).rename(columns=lambda c: f"목표 {c:.0%}" if isinstance(c, float) else c)), ""]
    L += ["#### U1 사이클과 회차 소진 (판정 대상 칸)", "",
          C.md_table(u1s[((u1s["ticker"] == "TQQQ") & (u1s["target"] == 0.10) | (u1s["ticker"] == "SOXL") & (u1s["target"] == 0.12)) & (u1s["splits"] == 40)]
                     .rename(columns={"ticker": "종목", "cycles": "사이클", "exhausts": "회차 소진", "open_exhaust": "끝날 때 소진 중",
                                      "median_recovery_days": "회복 중앙값(거래일)", "max_recovery_days": "회복 최장(거래일)"})
                     [["종목", "사이클", "회차 소진", "끝날 때 소진 중", "회복 중앙값(거래일)", "회복 최장(거래일)"]]), ""]
    # compare
    rank = pd.read_csv(cd / "ranking.csv")
    L += ["## 통합 비교 (판정 없음)", "", "### 순위표 (2015-01 → 끝, 샤프 순)", "",
          C.md_table(pd.DataFrame([{"전략": r.series, "시장": r.market, "첫날": r.first_day, "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd),
                                    "샤프": C.num(r.sharpe), "학습 연": C.pct(r.cagr_train), "검증 연": C.pct(r.cagr_test)} for r in rank.itertuples()])), ""]
    best = pd.read_csv(cd / "best_expectancy.csv")
    L += ["한국 그리드의 거래당 평균 1위 칸(합산 시장, 거래 200건 이상, 거래 단위): " +
          "; ".join(f"{r.strategy} 조합 {int(r.combo)} {C.pct(r.mean, 2)} ({int(r.trades)}건)" for r in best.itertuples()) +
          ". 결과를 보고 고른 것이라 판정하지 않는다.", "",
          "상관 행렬과 누적 곡선은 `results/compare/correlation.png`, `equity_compare.png` 에 있다.", ""]
    corr = pd.read_csv(cd / "correlation.csv", index_col=0)
    L += ["### 월간 수익률 상관", "", C.md_table(corr.map(lambda v: C.num(v)).reset_index().rename(columns={"index": ""})), ""]
    reg = pd.read_csv(cd / "regimes.csv")
    L += ["### 국면별 월평균 수익 (전 달 말 기준 국면, 한국 전략은 코스피, 미국 전략은 SPY)", "",
          C.md_table(reg.set_index("series").map(lambda v: C.pct(v, 2)).reset_index()), ""]
    mx = pd.read_csv(cd / "mixes.csv")
    L += ["### 상관이 낮은 조합의 균등 혼합", "",
          C.md_table(pd.DataFrame([{"개수": r.k, "조합": r.series, "평균 상관": C.num(r.mean_corr), "혼합 연 수익률": C.pct(r.mix_cagr),
                                    "혼합 최대 낙폭": C.pct(r.mix_mdd), "구성 낙폭 평균": C.pct(r.parts_mdd_mean), "기간": f"{r.first} → {r.last}"}
                                   for r in mx.itertuples()])), ""]
    rb = pd.read_csv(cd / "robustness.csv")
    L += ["### 학습·검증 순위 유지", "",
          f"전략들의 학습 연 수익률 순위와 검증 연 수익률 순위의 상관: {rb['series_cagr_rank_corr'].iloc[0]:.2f}", "",
          C.md_table(pd.DataFrame([{"그리드": r.strategy, "비교한 칸": r.combos, "학습·검증 거래당 평균 순위 상관": C.num(r.rank_corr),
                                    "학습 상위 10칸 중 검증 상위 4분의 1": C.pct(r.top10_train_in_test_top_quartile, 0),
                                    "학습 상위 10칸의 검증 거래당 평균": C.pct(r.test_mean_of_train_top10, 2)} for r in rb.itertuples()])), ""]
    (C.results_dir() / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
