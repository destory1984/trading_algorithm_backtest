"""results/report.md from the CSV files (SPEC 4.7 and 6). Run last.
Usage: python -m src.report
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src import overfit

pct, num, md = C.pct, C.num, C.md_table


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    p = cfg["pass"]
    js = list(cfg["tickers"]["judged"])
    met = pd.read_csv(out / "metrics.csv")
    reg = pd.read_csv(out / "regime.csv")
    mk = pd.read_csv(out / "monkey.csv").set_index("ticker")
    dsr = pd.read_csv(out / "dsr.csv")
    bk = pd.read_csv(out / "basket.csv").set_index("version")
    bear = met[met["version"] == "bear"].set_index("ticker")
    own = reg[reg["basis"] == "자기 200일선"].set_index("ticker")

    def per(f):
        return {t: f(t) for t in js}

    c1 = per(lambda t: (own.loc[t, "below_mean"] > own.loc[t, "above_mean"], f"{pct(own.loc[t, 'below_mean'], 2)} / {pct(own.loc[t, 'above_mean'], 2)}"))
    c2 = per(lambda t: (bear.loc[t, "trades"] >= p["min_trades"], f"{int(bear.loc[t, 'trades'])}건"))
    c3 = per(lambda t: (bear.loc[t, "diff_risk"] > 0, pct(bear.loc[t, "diff_risk"])))
    c5 = per(lambda t: (mk.loc[t, "pf_pct"] >= p["monkey_pct"], f"{mk.loc[t, 'pf_pct']:.3f}"))
    be = float(bear.loc[js, "breakeven_bp"].mean())
    d6 = float(dsr.iloc[0]["dsr"])
    crit = [
        (f"1. 아래 평균 > 위 평균 (필터 없는 IBS, 자기 200일선), {p['regime_min']}개 이상", c1, p["regime_min"]),
        (f"2. 약세장 판본 거래 {p['min_trades']}건 이상, 6개 모두", c2, 6),
        (f"3. 낙폭 맞춘 보유보다 연 수익률 높음, {p['beat_min']}개 이상", c3, p["beat_min"]),
        (f"5. 200일선 아래 무작위 진입 대비 PF 백분위 ≥ {p['monkey_pct']}, {p['monkey_min']}개 이상", c5, p["monkey_min"]),
    ]
    rows, ok = [], {}
    for name, per_t, need in crit:
        n = sum(bool(v[0]) for v in per_t.values())
        ok[name[:2]] = n >= need
        rows.append({"기준": name, **{t: ("○ " if per_t[t][0] else "× ") + per_t[t][1] for t in js},
                     "만족": f"{n}/6", "판정": "만족" if n >= need else "불만족"})
    ok["4."] = be >= p["breakeven_bp"]
    ok["6."] = d6 >= p["dsr_min"]
    rows.insert(3, {"기준": f"4. 손익분기 비용 6개 평균 ≥ 편도 {p['breakeven_bp']}bp", **{t: num(bear.loc[t, "breakeven_bp"], 1) for t in js},
                    "만족": f"평균 {be:.1f}bp", "판정": "만족" if ok["4."] else "불만족"})
    rows.append({"기준": f"6. 판정 6개 바구니 Deflated Sharpe ≥ {p['dsr_min']} (N {overfit.n_trials(cfg):,})", **{t: "" for t in js},
                 "만족": f"{d6:.3f}", "판정": "만족" if ok["6."] else "불만족"})
    n_ok = sum(ok.values())
    final = "신호 알림 후보" if n_ok == 6 else "후보 아님"
    L = ["# IBS 약세장 판본 보고서", "",
         f"규칙: IBS < {cfg['rule']['ibs_entry']} 이고 종가 < 자기 {cfg['rule']['sma']}일선이면 다음 날 시가 진입, IBS > {cfg['rule']['ibs_exit']} 이면 다음 날 시가 청산. "
         f"비용 편도 0.12%. 기간 {bear['first_day'].iloc[0]} → {bear['last_day'].iloc[0]}. 판정 종목은 {', '.join(js)}. 지시서는 `SPEC.md`.", "",
         "## 1. 판정", "", md(pd.DataFrame(rows)), "", f"**최종 판정: {final}** (기준 6개 중 {n_ok}개 만족)", ""]

    cols = [("trades", "거래", lambda v: str(int(v))), ("win_rate", "승률", lambda v: f"{v:.0%}"), ("expectancy", "거래당", lambda v: pct(v, 2)),
            ("profit_factor", "PF", num), ("cagr", "연 수익률", pct), ("sharpe", "샤프", num), ("mdd", "최대 낙폭", pct),
            ("exposure", "노출", lambda v: f"{v:.0%}"), ("cagr_zero_cost", "비용 0 연", pct), ("breakeven_bp", "손익분기(bp)", lambda v: num(v, 1)),
            ("hold_cagr", "보유", pct), ("risk_cagr", "낙폭 맞춘 보유", pct), ("diff_risk", "차이", pct), ("exp_cagr", "노출 맞춘 보유", pct),
            ("diff_exp", "차이 ", pct)]
    for grp in ("판정", "참고"):
        g = met[met["group"] == grp]
        t = pd.DataFrame([{"종목": r["ticker"], "판본": "약세장" if r["version"] == "bear" else "필터 없음", **{n: f(r[k]) for k, n, f in cols}}
                          for _, r in g.iterrows()])
        L += [f"## 2. 종목별 지표 ({grp} 종목)", "", md(t), ""]

    t = pd.DataFrame([{"종목": r["ticker"], "구분": r["group"], "국면 기준": r["basis"], "위 거래": int(r["above_n"]), "위 평균": pct(r["above_mean"], 2),
                       "위 t": num(r["above_t"]), "아래 거래": int(r["below_n"]), "아래 평균": pct(r["below_mean"], 2), "아래 t": num(r["below_t"])}
                      for _, r in reg.iterrows()])
    L += ["## 3. 국면별 필터 없는 IBS 거래 (비용 포함)", "", "기준 1 은 \"자기 200일선\" 줄로 판정한다. S&P500 줄은 민감도다.", "", md(t), ""]

    t = pd.DataFrame([{"종목": tk, "구분": r["group"], "거래": int(r["trades"]), "PF": num(r["pf"]), "무작위 PF 중앙값": num(r["pf_rand_median"]),
                       "무작위 PF 상위 5%": num(r["pf_rand_p95"]), "PF 백분위": num(r["pf_pct"], 3), "거래당": pct(r["mean"], 2),
                       "무작위 거래당 중앙값": pct(r["mean_rand_median"], 2)} for tk, r in mk.iterrows()])
    L += ["## 4. 200일선 아래 무작위 진입", "", f"종목마다 {cfg['monkey']['runs']:,}번. 같은 거래 수, 전략 보유일에서 다시 뽑은 보유일, 겹치지 않는 거래, 다음 날 시가 체결, 같은 비용.", "", md(t), ""]

    t = pd.DataFrame([{"대상": r["what"], "판본": r["role"], "일별 샤프": num(r["sr_daily"], 4), "연 환산": num(r["sr_daily"] * 252 ** 0.5),
                       "왜도": num(r["skew"]), "첨도": num(r["kurt"], 1), "V": f"{r['var_sr']:.6f}", "SR0": num(r["sr0"], 4), "DSR": num(r["dsr"], 3)}
                      for _, r in dsr.iterrows()])
    L += ["## 5. Deflated Sharpe", "", f"N = {overfit.n_trials(cfg):,} (미국 누적 {cfg['trials_before']:,} + 이 폴더 12). V = us_shortterm 1차 S1 → S8 56개 설정의 일별 샤프 분산.", "", md(t), ""]

    t = pd.DataFrame([{"판본": "약세장" if v == "bear" else "필터 없음", "연 수익률": pct(r["cagr"]), "최대 낙폭": pct(r["mdd"]), "샤프": num(r["sharpe"]),
                       "노출": f"{r['exposure']:.0%}", "6개 보유": pct(r["hold_cagr"]), "보유 낙폭": pct(r["hold_mdd"]), "낙폭 맞춘 보유": pct(r["risk_cagr"]),
                       "차이": pct(r["diff_risk"]), "노출 맞춘 보유": pct(r["exp_cagr"]), "차이 ": pct(r["diff_exp"])} for v, r in bk.iterrows()])
    L += ["## 6. 판정 6개 바구니 (1/6 씩 따로 굴림)", "", md(t), ""]

    L += ["## 7. 돌린 설정 수와 봉인", "", f"- 이 폴더 시행 12개(판정 6개 × 2판본). 참고 종목 10개 실행은 판정에 쓰지 않았다. 미국 누적 {cfg['trials_before']:,} → {overfit.n_trials(cfg):,}.",
          f"- 가장 늦은 데이터 날짜 {C.end(cfg).date()}. 그 뒤 데이터는 쓰지 않았다(`checks.md` 1장).", ""]
    failed = [k.rstrip(".") for k, v in ok.items() if not v]
    concl = (f"IBS 약세장 판본은 {final}이다. 기준 6개 중 {n_ok}개만 만족했다(불만족: 기준 {', '.join(sorted(failed))})."
             if failed else f"IBS 약세장 판본은 {final}이다.")
    L += ["## 8. 결론", "", concl, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(md(pd.DataFrame(rows)))
    print(concl)


if __name__ == "__main__":
    main()
