"""results/deep/report.md from the CSV files of the other src.deep modules (SPEC2 5 and 7).
Usage: python -m src.deep.report
"""
from __future__ import annotations

import pandas as pd

from src import common
from src.deep import base

pct, num, share, md = common.pct, common.num, common.share, common.md_table


def verdict(cfg: dict, met, dsr, mk, nb) -> tuple[pd.DataFrame, str, dict]:
    d = cfg["deep"]
    tick = list(cfg["screen"]["tickers"])
    dsr = dsr[dsr["role"] == "판정"].set_index("ticker")
    mk = mk.set_index("ticker")
    ho = met[met["stretch"] == "holdout"].set_index("ticker")
    beats = nb.assign(win=nb["diff_risk"] > 0).groupby("ticker")["win"].sum()
    crit = {
        "1. Deflated Sharpe ≥ 0.95 (선별)": {t: (dsr.loc[t, "dsr"] >= d["dsr_min"], f"{dsr.loc[t, 'dsr']:.3f}") for t in tick},
        "2. 무작위 진입 PF 백분위 ≥ 0.95 (선별)": {t: (mk.loc[t, "pf_pct"] >= d["monkey_pct_min"], f"{mk.loc[t, 'pf_pct']:.4f}") for t in tick},
        "3. 연 수익률 > 위험 맞춘 보유 (이미 본 구간)": {t: (ho.loc[t, "diff_risk"] > 0, pct(ho.loc[t, "diff_risk"])) for t in tick},
        f"4. 이웃 25칸 중 이기는 칸 ≥ {d['neighbors_min']}": {t: (beats[t] >= d["neighbors_min"], f"{int(beats[t])}칸") for t in tick},
    }
    rows, ok = [], {}
    for name, per in crit.items():
        n = sum(bool(v[0]) for v in per.values())
        ok[name] = n >= d["min_etfs"]
        rows.append({"기준": name, **{t: ("○ " if per[t][0] else "× ") + per[t][1] for t in tick},
                     "만족 ETF": f"{n}/4", "판정": "만족" if ok[name] else "불만족"})
    names = list(crit)
    if all(ok.values()):
        final = "신호 알림 후보"
    elif not ok[names[2]] and all(ok[k] for k in names if k != names[2]):
        final = "보류 (기준 3 만 불만족)"
    else:
        final = "후보 아님"
    return pd.DataFrame(rows), final, ok


def main() -> None:
    cfg = common.load_config()
    out = common.results_dir("deep")
    tick = list(cfg["screen"]["tickers"])
    met = pd.read_csv(out / "metrics.csv")
    dsr = pd.read_csv(out / "dsr.csv")
    mk = pd.read_csv(out / "monkey.csv")
    nb = pd.read_csv(out / "neighbors.csv")
    yrs = pd.read_csv(out / "years.csv")
    reg = pd.read_csv(out / "regime.csv")
    cst = pd.read_csv(out / "costs.csv")
    park = pd.read_csv(out / "parking.csv")
    table, final, ok = verdict(cfg, met, dsr, mk, nb)
    c = cfg["us"]["costs"]
    L = ["# S1 IBS(open) 2차 심화 보고서", "",
         f"규칙: IBS < {cfg['deep']['rule']['entry']} 이면 다음 날 시가에 사고, IBS > {cfg['deep']['rule']['exit']} 이면 다음 날 시가에 판다. "
         f"비용 편도 {(c['buy_fee'] + c['slippage']) * 100:.2f}%. 지시서는 `SPEC2.md`.", "",
         "**한계**: 2023-01-03 → 2026-09-25 는 ibs_backtest 가 같은 규칙을 이미 돌려 결과를 본 구간이라 깨끗한 표본 밖이 아니다.", ""]

    L += ["## 1. 판정", "", f"기준마다 4개 ETF 중 {cfg['deep']['min_etfs']}개 이상이면 만족. ○ = 그 ETF 만족.", "", md(table), "",
          f"**최종 판정: {final}**", ""]

    cols = [("trades", "거래", lambda v: str(int(v))), ("win_rate", "승률", share), ("expectancy", "거래당 기대값", lambda v: pct(v, 2)),
            ("profit_factor", "손익비", num), ("cagr", "연 수익률", pct), ("sharpe", "샤프", num), ("mdd", "최대 낙폭", pct),
            ("exposure", "노출", share), ("cagr_zero_cost", "비용 0 연 수익률", pct), ("breakeven_bp", "손익분기(bp)", lambda v: num(v, 1)),
            ("hold_cagr", "보유", pct), ("risk_cagr", "위험 맞춘 보유", pct), ("diff_risk", "차이", pct),
            ("exp_cagr", "노출 맞춘 보유", pct), ("diff_exp", "차이 ", pct)]
    L += ["## 2. 구간별 지표", "", "이미 본 구간의 거래는 청산일이 그 구간에 있는 거래다(2022-12-30 에 열려 있던 거래 포함).", ""]
    for s in base.STRETCHES:
        g = met[met["stretch"] == s]
        t = pd.DataFrame([{"ETF": r["ticker"], **{n: f(r[k]) for k, n, f in cols}} for _, r in g.iterrows()])
        L += [f"### {base.NAMES[s]} ({g['first_day'].iloc[0]} → {g['last_day'].iloc[0]})", "", md(t), ""]

    j = dsr[dsr["role"] == "판정"]
    t = pd.DataFrame([{"ETF": r["ticker"], "일별 샤프": num(r["sr_daily"], 4), "연 환산": num(r["sr_daily"] * 252 ** 0.5),
                       "왜도": num(r["skew"]), "첨도": num(r["kurt"], 1), "SR0 (일별)": num(r["sr0"], 4), "DSR": num(r["dsr"], 3)}
                      for _, r in j.iterrows()])
    sens = dsr[dsr["role"] != "판정"].pivot_table(index="ticker", columns=["role", "n_trials"], values="dsr")
    sens_rows = []
    for tk in tick:
        row = {"ETF": tk}
        for (role, n) in sens.columns:
            label = f"N {n}" if role == "민감도" else f"N {n}, V S1 → S8"
            row[label] = num(sens.loc[tk, (role, n)], 3)
        sens_rows.append(row)
    v = j["var_sr"].iloc[0]
    L += ["## 3. Deflated Sharpe 와 무작위 진입", "",
          f"Deflated Sharpe: 선별 구간, N = {int(j['n_trials'].iloc[0]):,}, V = 1차 72개 설정 일별 샤프의 분산 {v:.6f}.", "", md(t), "",
          "민감도(판정에 쓰지 않음). V 가 큰 것은 S9 → S12 의 일별 샤프가 크게 음수라서다. 그것을 뺀 56개로 V 를 잡아도 0.95 에 못 미친다.", "",
          md(pd.DataFrame(sens_rows)), ""]
    t = pd.DataFrame([{"ETF": r["ticker"], "거래": int(r["trades"]), "PF": num(r["pf"]), "무작위 PF 중앙값": num(r["pf_rand_median"]),
                       "무작위 PF 상위 5%": num(r["pf_rand_p95"]), "PF 백분위": num(r["pf_pct"], 4), "연 수익률": pct(r["cagr"]),
                       "무작위 연 수익률 중앙값": pct(r["cagr_rand_median"]), "연 수익률 백분위": num(r["cagr_pct"], 4)}
                      for _, r in mk.iterrows()])
    L += [f"무작위 진입: ETF 마다 {cfg['deep']['monkey_runs']:,}번, 같은 거래 수, 전략 보유일에서 다시 뽑은 보유일, 겹치지 않는 거래, 다음 날 시가 체결, 같은 비용.", "",
          md(t), ""]

    L += ["## 4. 이웃 문턱값 (선별 구간)", "", "각 칸은 위험을 맞춘 보유 대비 연 수익률 차이다. 굵은 칸이 규칙(0.20 / 0.80)이다. 고르는 용도가 아니다.", ""]
    for tk in tick:
        g = nb[nb["ticker"] == tk].pivot_table(index="entry", columns="exit", values="diff_risk")
        rows = []
        for en, r in g.iterrows():
            row = {"진입 \\ 청산": f"{en:.2f}"}
            for ex, val in r.items():
                s = pct(val)
                row[f"{ex:.2f}"] = f"**{s}**" if round(en, 2) == 0.2 and round(ex, 2) == 0.8 else s
            rows.append(row)
        L += [f"### {tk} (이기는 칸 {int((g > 0).sum().sum())}/25)", "", md(pd.DataFrame(rows)), ""]

    L += ["## 5. 해마다 성적과 국면", "", "각 칸은 \"전략 / 위험 맞춘 보유 대비 차이\"다. 위험 맞춘 비중은 전체 구간 값 하나를 모든 해에 쓴다.", ""]
    rows = []
    for y, g in yrs.groupby("year"):
        g = g.set_index("ticker")
        rows.append({"해": y, **{tk: f"{pct(g.loc[tk, 'strategy'])} / {pct(g.loc[tk, 'diff_risk'])}" for tk in tick}})
    L += [md(pd.DataFrame(rows)), ""]
    n_pos = yrs.assign(p=yrs["diff_risk"] > 0).groupby("ticker")["p"].sum()
    L += ["위험 맞춘 보유를 이긴 해: " + ", ".join(f"{tk} {int(n_pos[tk])}/{yrs['year'].nunique()}" for tk in tick) + ".", ""]
    t = pd.DataFrame([{"ETF": r["ticker"], "국면(신호일)": r["regime"], "거래": int(r["trades"]), "거래당 평균": pct(r["mean_ret"], 2),
                       "t": num(r["t"]), "승률": share(r["win_rate"])} for _, r in reg.iterrows()])
    L += ["신호일 국면별 거래 수익(비용 포함, 전체 구간). 보고만 하고 필터로 쓰지 않는다.", "", md(t), ""]

    L += ["## 6. 비용 민감도", "", "각 칸은 위험을 맞춘 보유 대비 연 수익률 차이다. 편도 비용을 매수·매도에 똑같이 붙였다.", ""]
    for s in ("screen", "holdout"):
        g = cst[cst["stretch"] == s].pivot_table(index="ticker", columns="bp", values="diff_risk").loc[tick]
        t = g.map(pct).reset_index().rename(columns={"ticker": "ETF", **{b: f"{b}bp" for b in g.columns}})
        wins = (g > 0).sum()
        L += [f"### {base.NAMES[s]}", "", md(t), "", "이기는 ETF 수: " + ", ".join(f"{b}bp {int(wins[b])}/4" for b in g.columns) + ".", ""]

    L += ["## 7. 쉬는 날 SPY 보유 (참고)", "",
          "포지션이 없는 날 평가금액의 w 만큼 SPY 를 들고, 신호가 나면 전액을 신호 ETF 에 넣는다. 비교는 같은 평균 SPY 비중의 SPY + 현금 고정 보유다. 전체 구간.", ""]
    g = park[park["stretch"] == "full"]
    t = pd.DataFrame([{"ETF": r["ticker"], "w": r["w"], "연 수익률": pct(r["cagr"]), "최대 낙폭": pct(r["mdd"]), "샤프": num(r["sharpe"]),
                       "평균 SPY 비중": share(r["spy_w"]), "고정 보유 연 수익률": pct(r["mix_cagr"]), "고정 보유 샤프": num(r["mix_sharpe"])}
                      for _, r in g.iterrows()])
    L += [md(t), ""]

    n_before = sum(cfg["deep"]["trials_before"].values())
    L += ["## 8. 돌린 설정 수와 봉인", "",
          f"- 이번에 새로 돌린 설정: 이웃 문턱값 25개 × ETF 4개 = 100개. 규칙 자체(0.20 / 0.80)는 1차에서 센 것이다. 누적 시행 수 {n_before:,} → {n_before + 100:,}.",
          f"- 가장 늦은 데이터 날짜: {base.deep_end(cfg).date()}. 그 뒤 데이터는 쓰지 않았다(`checks.md` 1장).",
          "- 이미 본 구간은 깨끗한 표본 밖이 아니다(ibs_backtest 가 먼저 봄).", ""]

    failed = [k.split(".")[0] for k, v in ok.items() if not v]
    concl = (f"S1 IBS(open)는 {final}이다. 판정 기준 4개 중 {4 - len(failed)}개만 만족했다(불만족: 기준 {', '.join(failed)})."
             if failed else f"S1 IBS(open)는 {final}이다. 판정 기준 4개를 모두 만족했다.")
    L += ["## 9. 결론", "", concl, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(md(table))
    print(concl)


if __name__ == "__main__":
    main()
