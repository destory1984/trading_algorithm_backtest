"""results/deep/report.md from the CSV files of the other src.deep modules (SPEC2 5 and 7).
Usage: python -m src.deep.report
"""
from __future__ import annotations

import pandas as pd

from src.deep import overfit
from src.deep import shared as S

pct, md = S.pct, S.md_table
NAMES = {"full": "전체", "train": "학습기 2008 → 2016", "test": "검증기 2017 → 2026"}


def num(v, d=2) -> str:
    return "" if pd.isna(v) else f"{v:.{d}f}"


def main() -> None:
    cfg = S.config()
    out = S.results_dir()
    d = cfg["deep"]
    met = pd.read_csv(out / "metrics.csv")
    dsr = pd.read_csv(out / "dsr.csv")
    mk = pd.read_csv(out / "monkey.csv")
    nb = pd.read_csv(out / "neighbors.csv")
    cr = pd.read_csv(out / "crashes.csv")
    top = pd.read_csv(out / "top_days.csv")
    cs = pd.read_csv(out / "costs.csv")
    cap = pd.read_csv(out / "capacity.csv").iloc[0]
    br = pd.read_csv(out / "breaks.csv")
    yrs = pd.read_csv(out / "years.csv")
    m = met.set_index("stretch")

    j = dsr.iloc[0]
    day = dsr[dsr["role"] == "민감도 신호일 단위"].iloc[0]
    same = mk.iloc[0]
    n_win = int((nb["diff_risk"] > 0).sum())
    slip = cs[(cs["slippage"].round(4) == round(d["slippage_pass"], 4)) & (cs["stretch"] == "full")].iloc[0]
    t5 = top[top["top_days"] == 5].iloc[0]
    crit = [
        ("1. 거래 단위 Deflated Sharpe ≥ 0.95", j["dsr"] >= d["dsr_min"], f"{j['dsr']:.3f} (N {int(j['n_trials']):,}, T {int(j['T'])})"),
        ("2. 같은 날 무작위 종목 대비 PF 백분위 ≥ 0.95", same["pf_pct"] >= d["monkey_pct_min"], f"{same['pf_pct']:.3f}"),
        ("3. 낙폭 맞춘 코스피보다 높음 (전체, 학습기, 검증기)", all(m.loc[s, "diff_risk"] > 0 for s in ("full", "train", "test")),
         " / ".join(pct(m.loc[s, "diff_risk"]) for s in ("full", "train", "test"))),
        (f"4. 이웃 18칸 중 {d['neighbors_min']}칸 이상 이김", n_win >= d["neighbors_min"], f"{n_win}칸"),
        (f"5. 슬리피지 편도 {d['slippage_pass']:.1%} 에서 전체 구간 이김", slip["diff_risk"] > 0, pct(slip["diff_risk"])),
        ("6. 가장 좋은 신호일 5일을 뺀 거래당 평균 > 0", t5["mean_without"] > 0, pct(t5["mean_without"])),
    ]
    n_ok = sum(bool(c[1]) for c in crit)
    final = "신호 알림 후보" if n_ok == len(crit) else "후보 아님"
    r = S.rule(cfg)
    L = ["# bnf 동반급락 규칙 심화 보고서", "",
         f"규칙: 코스피 종목, 종가 25일선 이격도 ≤ {r['threshold']}, 신호일 코스피 지수 이격도 ≤ {r['market_filter'][5:]}, 다음 날 시가 진입, "
         f"25일선 회복 다음 날 시가 또는 {r['hold']}일째 종가 청산, 손절 없음. 1억 원 5자리. 기간 {m.loc['full', 'first_day']} → {m.loc['full', 'last_day']}. 지시서는 `SPEC2.md`.", "",
         f"비교 지수는 코스피 가격 지수에 연 {d['kospi_dividend']:.1%} 배당을 더한 총수익 근사다. 전 구간을 이미 보았으므로 학습기·검증기 나눔도 깨끗한 표본 밖이 아니다.", ""]
    L += ["## 1. 판정", "", md(pd.DataFrame([{"기준": a, "값": c, "판정": "만족" if b else "불만족"} for a, b, c in crit])), "",
          f"**최종 판정: {final}** (기준 6개 중 {n_ok}개 만족)", "",
          f"기준 1 주의: 거래 804건을 서로 독립으로 본 값이다. 거래가 {int(day['T'])}개 신호일에 몰려 있어(하루 평균 {int(j['T']) / day['T']:.1f}건) 독립이 아니다. "
          f"신호일을 한 단위로 보면(판정에 쓰지 않는 민감도) DSR 이 {day['dsr']:.2f} 다.", ""]

    rows = []
    for s in ("full", "train", "test"):
        x = m.loc[s]
        rows.append({"구간": NAMES[s], "연 수익률": pct(x["cagr"]), "최대 낙폭": pct(x["mdd"]), "샤프": num(x["sharpe"]),
                     "노출": f"{x['exposure']:.0%}", "코스피": pct(x["kospi_cagr"]), "낙폭 맞춘 코스피": pct(x["risk_cagr"]),
                     "비중 w": num(x["risk_w"]), "차이": pct(x["diff_risk"]), "노출 맞춘 코스피": pct(x["exp_cagr"]),
                     "차이 ": pct(x["diff_exp"]), "거래(순차)": int(x["t_trades"]), "거래당 평균": pct(x["t_mean_ret"]),
                     "승률": f"{x['t_win_rate']:.0%}"})
    f = m.loc["full"]
    L += ["## 2. 기준 결과", "", md(pd.DataFrame(rows)), "",
          f"포트폴리오는 신호 {int(f['p_trades_available']):,}건 중 {int(f['p_trades_taken'])}건만 샀다(자리 5개). 산 거래의 평균은 {pct(f['p_taken_mean_ret'])}, "
          f"모든 신호의 평균은 {pct(f['p_all_mean_ret'])} 다. 신호가 한꺼번에 몰리는 날 자리가 모자라 대부분을 놓친다. 종목을 든 날은 {f['p_invested_share']:.0%} 다.", ""]
    yr = [{"해": int(y["year"]), "거래": "" if pd.isna(y["count"]) else int(y["count"]), "거래당 평균": pct(y["mean"]),
           "포트폴리오": pct(y["portfolio"]), "코스피": pct(y["kospi"]), "낙폭 맞춘 코스피 대비": pct(y["diff_risk"])} for _, y in yrs.iterrows()]
    L += ["해마다(거래는 순차 목록, 낙폭 맞춘 비중은 전체 구간 값 하나):", "", md(pd.DataFrame(yr)), ""]

    L += ["## 3. Deflated Sharpe 와 같은 날 무작위 종목", "",
          md(pd.DataFrame([{"판본": x["role"], "샤프(단위당)": num(x["sr_trade"], 3), "T": int(x["T"]), "왜도": num(x["skew"]),
                            "첨도": num(x["kurt"], 1), "N": int(x["n_trials"]), "V": num(x["var_sr"], 4), "SR0": num(x["sr0"], 3),
                            "DSR": num(x["dsr"], 3)} for _, x in dsr.iterrows()])), "",
          "규칙의 거래 단위 샤프는 그리드 1,440개 조합 가운데 1등이다.", ""]
    L += [md(pd.DataFrame([{"비교": x["test"], "규칙 PF": num(x["pf"]), "무작위 PF 중앙값": num(x["pf_rand_median"]),
                            "무작위 PF 상위 5%": num(x["pf_rand_p95"]), "PF 백분위": num(x["pf_pct"], 3),
                            "규칙 거래당 평균": pct(x["mean"]), "무작위 평균 중앙값": pct(x["mean_rand_median"]),
                            "평균 백분위": num(x["mean_pct"], 3)} for _, x in mk.iterrows()])), "",
          f"규칙의 신호 {int(same['trades']):,}건(각 신호를 따로 본 거래)은 {int(same['signal_days'])}개 신호일에 있다. 같은 날 무작위로 산 종목도 거래당 {pct(same['mean_rand_median'])} 를 번다. "
          f"폭락한 날 사는 것만으로 대부분을 벌고, 이격도 ≤ 70 종목 고르기가 그 위에 {pct(same['mean'] - same['mean_rand_median'])}p 를 더한다.", ""]

    L += ["## 4. 이웃 설정 18칸 (5자리 포트폴리오, 전체 구간)", "", "각 칸은 \"연 수익률 / 낙폭 맞춘 코스피 대비\"다. 굵은 칸이 규칙이다.", ""]
    rows = []
    for (thr, hold), g in nb.groupby(["threshold", "hold"], sort=False):
        row = {"이격도": f"≤{thr}", "보유": f"{hold}일"}
        for _, x in g.iterrows():
            s = f"{pct(x['cagr'])} / {pct(x['diff_risk'])}"
            row[f"동반급락 ≤{x['market_filter'][5:]}"] = f"**{s}**" if x["combo"] == S.rule_key(cfg) else s
        rows.append(row)
    L += [md(pd.DataFrame(rows).sort_values(["이격도", "보유"], ascending=[False, True])), "", f"낙폭 맞춘 코스피를 이긴 칸: {n_win}/18.", ""]

    L += ["## 5. 폭락 의존도", "", "그 기간에 신호가 난 거래를 빼고 다시 돌렸다.", "",
          md(pd.DataFrame([{"뺀 기간": x["dropped"], "뺀 거래(순차)": int(x["seq_dropped"]), "남은 거래": int(x["trades"]),
                            "거래당 평균": pct(x["mean_ret"]), "포트폴리오 연": pct(x["cagr"]), "최대 낙폭": pct(x["mdd"]),
                            "낙폭 맞춘 코스피 대비": pct(x["diff_risk"])} for _, x in cr.iterrows()])), ""]
    L += [md(pd.DataFrame([{"가장 좋은 신호일": f"{int(x['top_days'])}일", "그날 거래": int(x["trades_on_days"]),
                            "전체 이익 중 비중": f"{x['share_of_sum']:.0%}", "뺀 뒤 거래당 평균": pct(x["mean_without"]),
                            "날짜": x["days"]} for _, x in top.iterrows()])), ""]

    L += ["## 6. 비용·체결", "", "각 칸은 낙폭 맞춘 코스피 대비 연 수익률 차이다. 수수료와 매도세는 그대로 두고 슬리피지만 바꿨다.", ""]
    rows = []
    for sl, g in cs.groupby("slippage"):
        g = g.set_index("stretch")
        rows.append({"슬리피지(편도)": f"{sl:.1%}", "순차 거래당 평균": pct(g.loc["full", "seq_mean"]),
                     **{NAMES[s]: pct(g.loc[s, "diff_risk"]) for s in ("full", "train", "test")}})
    L += [md(pd.DataFrame(rows)), "",
          f"체결 규모: 포트폴리오가 산 {int(cap['taken'])}건에서 자리 크기 / 20일 평균 거래대금의 중앙값 {cap['share_median']:.2%}, 상위 10% {cap['share_p90']:.2%}, "
          f"최대 {cap['share_max']:.2%}. {cap['limit']:.0%} 를 넘는 거래가 {int(cap['over_limit'])}건이다. 1억 원일 때 숫자라 돈이 커지면 비례해 커진다.", "",
          f"진입일 시가 갭(시가 / 신호일 종가 - 1): 산 거래 중앙값 {pct(cap['gap_taken_median'])}, 하위 10% {pct(cap['gap_taken_p10'])}, 상위 10% {pct(cap['gap_taken_p90'])}.", ""]

    dvd = pd.read_csv(out / "dividend.csv")
    L += [f"코스피 배당 가정 민감도(판정에 쓰지 않음). 판정은 연 {d['kospi_dividend']:.1%} 로 했다. 이 숫자는 어림값이다.", "",
          md(pd.DataFrame([{"배당(연)": f"{x['dividend']:.1%}", "기준 3 전체": pct(x["diff_full"]), "학습기": pct(x["diff_train"]),
                            "검증기": pct(x["diff_test"]), "기준 4 이긴 칸": f"{int(x['neighbors_won'])}/18",
                            "기준 5": pct(x["slip_full"])} for _, x in dvd.iterrows()])), "",
          "배당을 0 으로 두면 기준 3 과 5 는 만족으로 바뀌지만, 기준 4 는 어느 가정에서도 18칸 중 3칸 이하라 판정은 바뀌지 않는다.", ""]
    L += ["## 7. 데이터 끊김으로 뺀 거래", "",
          md(pd.DataFrame([{"판본": x["version"], "순차 거래": int(x["trades"]), "거래당 평균": pct(x["mean_ret"]),
                            "포트폴리오 연": pct(x["cagr"]), "낙폭 맞춘 코스피 대비": pct(x["diff_risk"])} for _, x in br.iterrows()])), "",
          f"끊김을 낀 거래는 순차 목록 {int(br['seq_breaks'].iloc[0])}건, 신호 단위 {int(br['ind_breaks'].iloc[0])}건(모두 한 종목, 2008-09)이고 포트폴리오는 그 거래를 사지 않았다. 결과를 부풀리지 않았다.", ""]

    L += ["## 8. 돌린 설정 수와 봉인", "",
          f"- 새 시행 없음. 이웃 18칸은 그리드 안의 조합이다. 누적 시행 수는 그대로 {overfit.n_trials(cfg):,} 다.",
          f"- 가장 늦은 데이터 날짜 {S.deep_end(cfg).date()}. 그 뒤 데이터는 쓰지 않았다(`checks.md` 1장).", ""]
    failed = [c[0].split(".")[0] for c in crit if not c[1]]
    concl = (f"동반급락 규칙은 {final}이다. 기준 6개 중 {n_ok}개만 만족했다(불만족: 기준 {', '.join(failed)})."
             if failed else f"동반급락 규칙은 {final}이다. 기준 6개를 모두 만족했다.")
    L += ["## 9. 결론", "", concl, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(md(pd.DataFrame([{"기준": a, "값": c, "판정": "만족" if b else "불만족"} for a, b, c in crit])))
    print(concl)


if __name__ == "__main__":
    main()
