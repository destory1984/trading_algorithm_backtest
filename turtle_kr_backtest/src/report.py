"""SPEC 6: verdicts and the report. Run after metrics, baseline, dsr, stress and costs.
Usage: python -m src.report   -> results/verdicts.csv, results/report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C

CRIT = {1: "DSR ≥ 0.95", 2: "무작위 95%", 3: "낙폭 맞춘 코스피", 4: "이웃", 5: "슬리피지 0.3%", 6: "상위 5일 빼도 +"}


def best_days_removed(t: pd.DataFrame, n: int) -> float:
    day = t.groupby("entry_day")["ret"].mean()
    top = day.nlargest(n).index
    rest = t[~t["entry_day"].isin(top)]
    return float(rest["ret"].mean())


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    m = pd.read_csv(out / "metrics.csv")
    rnd = pd.read_csv(out / "random.csv")
    order = pd.read_csv(out / "order.csv")
    ds = pd.read_csv(out / "dsr.csv").set_index("variant")
    st = pd.read_csv(out / "stress.csv")
    cs = pd.read_csv(out / "costs.csv")
    tr = pd.read_parquet(out / "trades.parquet")
    ps = cfg["pass"]
    get = lambda k, s: m[(m["variant"] == k) & (m["stretch"] == s)].iloc[0]
    rows, detail = [], {}
    for r, s in cfg["rules"].items():
        k = C.vkey(r, s["entry"])
        f = get(k, "full")
        rr = rnd[rnd["rule"] == r]["cagr"]
        pctile = float((rr < f["cagr"]).mean())
        rest = best_days_removed(tr[tr["variant"] == k], ps["top_days"])
        c5 = cs[(cs["rule"] == r) & np.isclose(cs["slippage"], ps["slippage_check"])].iloc[0]
        ok = {1: ds.loc[k, "dsr"] >= ps["dsr_min"],
              2: pctile >= ps["random_pct_min"],
              3: all(get(k, x)["diff_risk"] > 0 for x in C.STRETCHES),
              4: all(get(C.vkey(r, n), "full")["diff_risk"] > 0 for n in s["neighbors"]),
              5: c5["diff_risk"] > 0,
              6: rest > 0}
        detail[r] = {1: f"DSR {ds.loc[k, 'dsr']:.3f} (진입일 샤프 {ds.loc[k, 'sr']:+.3f}, 기대 최대 {ds.loc[k, 'sr0']:.3f}, 진입일 {int(ds.loc[k, 'T'])}개)",
                     2: f"연 {C.pct(f['cagr'])}, 무작위 200번 중앙값 {C.pct(rr.median())}, 95 백분위 {C.pct(rr.quantile(0.95))}; 규칙은 {pctile:.0%} 위치",
                     3: " / ".join(f"{C.NAMES[x]} {C.pp(get(k, x)['diff_risk'])}" for x in C.STRETCHES),
                     4: ", ".join(f"{n}일 {C.pp(get(C.vkey(r, n), 'full')['diff_risk'])}" for n in s["neighbors"]),
                     5: f"연 {C.pct(c5['cagr'])}, 낙폭 맞춘 코스피 대비 {C.pp(c5['diff_risk'])}",
                     6: f"상위 {ps['top_days']}일을 뺀 거래당 평균 {C.pct(rest)}"}
        rows.append({"rule": r, **{f"c{i}": bool(v) for i, v in ok.items()}, "passed": all(ok.values()),
                     "failed": ", ".join(str(i) for i, v in ok.items() if not v), "random_pctile": pctile,
                     "best_days_removed": rest})
    v = pd.DataFrame(rows)
    v.to_csv(out / "verdicts.csv", index=False)

    L = ["# 한국 종목 터틀 돌파 결과", "",
         f"코스피·코스닥 개별 종목, 매수만. 곡선 {cfg['data']['curve_start']} → {cfg['data']['end']}, 처음 자산 "
         f"{cfg['turtle']['capital'] / 1e8:.0f}억 원, 슬리피지 편도 {cfg['costs']['slippage']:.1%}. 지시서 SPEC.md.", "",
         "## 판정", ""]
    for x in v.itertuples():
        spec = cfg["rules"][x.rule]
        L += [f"### {x.rule} ({spec['entry']}일 돌파, {spec['exit']}일 청산): {'통과' if x.passed else '탈락'}", "",
              C.md_table(pd.DataFrame([{"기준": f"{i} {CRIT[i]}", "결과": "O" if getattr(x, f"c{i}") else "X",
                                        "내용": detail[x.rule][i]} for i in CRIT])), ""]

    L += ["## 전체 표", ""]
    for s in C.STRETCHES:
        sub = m[m["stretch"] == s]
        L += [f"### {C.NAMES[s]} ({sub['first_day'].iloc[0]} → {sub['last_day'].iloc[0]})", "",
              C.md_table(pd.DataFrame([{
                  "판본": r.variant, "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd), "샤프": C.num(r.sharpe),
                  "평균 투자 비율": f"{r.avg_invested:.0%}", "평균 보유 단위": C.num(r.avg_units_held, 1),
                  "거래": r.trades, "한 해 거래": C.num(r.trades_per_year, 0), "승률": C.pct(r.win_rate, 0),
                  "거래당 평균": C.pct(r.avg_ret), "손익비": C.num(r.profit_factor),
                  "같은 날 청산": C.pct(r.same_day_exit, 0), "코스피": C.pct(r.kospi_cagr),
                  "낙폭 맞춘 코스피(w)": f"{C.pct(r.risk_cagr)} ({r.risk_w:.2f})", "그 대비": C.pp(r.diff_risk)}
                  for r in sub.itertuples()])), ""]

    L += ["## 무작위 진입 200번 (판정 기준 2)", "",
          C.md_table(rnd.groupby("rule").agg(중앙값=("cagr", "median"), 하위5=("cagr", lambda x: x.quantile(0.05)),
                                             상위5=("cagr", lambda x: x.quantile(0.95)), 낙폭중앙값=("mdd", "median"),
                                             승률중앙값=("win", "median")).reset_index()
                     .assign(**{c: (lambda c: lambda d: d[c].map(lambda y: C.pct(y)))(c)
                                for c in ["중앙값", "하위5", "상위5", "낙폭중앙값", "승률중앙값"]})), ""]
    L += ["## 민감도: 같은 날 신호를 무작위 순서로 50번 (판정 제외)", "",
          "지시서는 거래대금이 큰 종목부터 사게 정했다. 순서가 결과를 얼마나 바꾸는지 보려고 같은 날 신호의 순서만 섞었다.", "",
          C.md_table(order.groupby("rule").agg(중앙값=("cagr", "median"), 최소=("cagr", "min"), 최대=("cagr", "max"))
                     .reset_index().assign(**{c: (lambda c: lambda d: d[c].map(lambda y: C.pct(y)))(c)
                                              for c in ["중앙값", "최소", "최대"]})), ""]
    L += ["## 슬리피지", "", C.md_table(pd.DataFrame([{
        "규칙": r.rule, "슬리피지(편도)": f"{r.slippage:.1%}", "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd),
        "거래": r.trades, "낙폭 맞춘 코스피 대비": C.pp(r.diff_risk)} for r in cs.itertuples()])), ""]
    L += ["## 큰 구간", "", C.md_table(pd.DataFrame([{
        "구간": r.window, "대상": r.rule, "수익": C.pct(r["return"]), "낙폭": C.pct(r.mdd),
        "투자 비율": "" if pd.isna(r.avg_invested) else f"{r.avg_invested:.0%}",
        "청산 거래": "" if pd.isna(r.trades) else int(r.trades), "이익 상위": "" if pd.isna(r.top5) else r.top5}
        for _, r in st.iterrows()])), ""]
    L += ["## Deflated Sharpe", "",
          "진입일 단위. V 는 이 폴더 6개 판본의 진입일 샤프 분산이다. 오른쪽 두 열은 bnf 그리드의 신호일 샤프 분산으로 잰 민감도(판정 제외)다.", "",
          C.md_table(pd.DataFrame([{"판본": k, "진입일": int(r["T"]), "진입일 샤프": C.num(r["sr"], 3), "왜도": C.num(r["skew"]),
                                    "첨도": C.num(r["kurt"]), "기대 최대": C.num(r["sr0"], 3), "DSR": C.num(r["dsr"], 3),
                                    "DSR (bnf V)": C.num(r.get("dsr_bnf_v"), 3)} for k, r in ds.iterrows()])), ""]
    reasons = tr.groupby(["variant", "reason"]).size().unstack(fill_value=0)
    L += ["## 청산 이유", "", C.md_table(reasons.reset_index()), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
