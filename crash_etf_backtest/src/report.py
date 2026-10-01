"""SPEC 6: verdict and report. Run after run / metrics / monkey / overfit / crashes.
Usage: python -m src.report   -> results/report.md, verdicts.csv
"""
from __future__ import annotations

import math

import pandas as pd

from src import common as C


def ox(b: bool) -> str:
    return "O" if b else "X"


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    mb = pd.read_csv(out / "metrics_basket.csv").set_index(["variant", "period"])
    mt = pd.read_csv(out / "metrics_ticker.csv")
    ex = pd.read_csv(out / "metrics_extra.csv").iloc[0]
    rnd = pd.read_csv(out / "random.csv")
    ds = pd.read_csv(out / "dsr.csv").set_index("variant")
    dr = pd.read_csv(out / "drop_days.csv").iloc[0]
    days = pd.read_csv(out / "signal_days.csv")
    yr = pd.read_csv(out / "yearly.csv", index_col=0)
    st = pd.read_csv(out / "stress.csv")
    p = cfg["pass"]
    main_, p1, p2 = mb.loc[("95", "전체")], *[mb.loc[("95", k)] for k in cfg["data"]["periods"]]
    need = math.ceil(p["ticker_share"] * len(mt) - 1e-9)
    pos = int((mt["mean"] > 0).sum())
    p95 = rnd["cagr"].quantile(p["monkey_pct"])
    place = (rnd["cagr"] < main_["cagr"]).mean()
    n93, n97, c2 = mb.loc[("93", "전체")], mb.loc[("97", "전체")], mb.loc[("95|비용 2배", "전체")]
    d95 = ds.loc[95]
    v = [
        ("1 효과가 있다", main_["mean"] > 0 and pos >= need,
         f"거래 {int(main_['trades'])}건, 거래당 평균 {C.pct(main_['mean'], 2)}, 승률 {main_['win']:.0%}. 평균이 0 보다 큰 종목 {pos} / {len(mt)} (기준 {need})"),
        ("2 무작위 진입 95 백분위", main_["cagr"] >= p95,
         f"연 {C.pct(main_['cagr'], 2)}. 무작위 {len(rnd)}번 중앙값 {C.pct(rnd['cagr'].median(), 2)}, 95 백분위 {C.pct(p95, 2)} → {place:.0%} 위치"),
        ("3 낙폭 맞춘 보유보다 높음", main_["diff"] > 0,
         f"연 {C.pct(main_['cagr'], 2)}, 낙폭 {C.pct(main_['mdd'])}. 보유 {C.pct(main_['hold_cagr'])} / {C.pct(main_['hold_mdd'])}, 낙폭 맞춘 보유(w {main_['w']:.2f}) {C.pct(main_['matched_cagr'])} → {C.pp(main_['diff'])}"),
        ("4 두 기간", p1["diff"] > 0 and p2["diff"] > 0, f"1997 → 2011 {C.pp(p1['diff'])} / 2012 → 2026-09 {C.pp(p2['diff'])}"),
        ("5 이웃 93·97", n93["diff"] > 0 and n97["diff"] > 0, f"93 {C.pp(n93['diff'])} / 97 {C.pp(n97['diff'])}"),
        ("6 비용 2배", c2["diff"] > 0, f"연 {C.pct(c2['cagr'], 2)}, {C.pp(c2['diff'])}"),
        (f"7 가장 좋은 신호일 {p['drop_days']}일 빼도 평균 > 0", dr["mean_without_best"] > 0,
         f"{C.pct(dr['mean_without_best'], 2)} (거래 {int(dr['trades_dropped'])}건 뺌)"),
        (f"8 DSR ≥ {p['dsr']}", d95["dsr"] >= p["dsr"],
         f"DSR {d95['dsr']:.3f} (진입일 샤프 {d95['sr']:+.3f}, 기대 최대 {d95['sr0']:.3f}, 진입일 {int(d95['T'])}개)"),
    ]
    vd = pd.DataFrame([{"기준": a, "결과": ox(b), "내용": c} for a, b, c in v])
    vd.to_csv(out / "verdicts.csv", index=False)
    failed = [a.split()[0] for a, b, _ in v if not b]
    verdict = "통과" if not failed else f"탈락 (못 넘은 기준 {', '.join(failed)})"

    def basket_rows(names):
        rows = []
        for (var, per), r in mb.iterrows():
            if var in names:
                rows.append({"판본": var, "구간": per, "연 수익률": C.pct(r["cagr"], 2), "최대 낙폭": C.pct(r["mdd"]),
                             "샤프": C.num(r["sharpe"]), "보유": f"{C.pct(r['hold_cagr'])} / {C.pct(r['hold_mdd'])}",
                             "낙폭 맞춘 보유(w)": f"{C.pct(r['matched_cagr'])} ({r['w']:.2f})", "그 대비": C.pp(r["diff"]),
                             "거래": int(r["trades"]), "승률": f"{r['win']:.0%}", "거래당 평균": C.pct(r["mean"], 2),
                             "PF": C.num(r["pf"])})
        return pd.DataFrame(rows)

    tick = pd.DataFrame([{"종목": f"{r.ticker} {r.name_}", "거래": r.trades, "승률": f"{r.win:.0%}", "거래당 평균": C.pct(r.mean_, 2),
                          "손익비": C.num(r.payoff), "평균 보유일": C.num(r.hold_days, 1), "칸 연 수익률": C.pct(r.cagr, 2),
                          "칸 낙폭": C.pct(r.mdd), "노출": f"{r.exposure:.0%}", "보유": f"{C.pct(r.hold_cagr)} / {C.pct(r.hold_mdd)}",
                          "낙폭 맞춘 보유 대비": C.pp(r.diff_), "손익분기 비용(편도)": f"{r.breakeven * 1e4:+.0f}bp"}
                         for r in mt.rename(columns={"name": "name_", "mean": "mean_", "diff": "diff_"}).itertuples()])
    dd = days.copy()
    dd["total"], dd["mean"] = dd["total"].map(lambda x: f"{x:+.2f}"), dd["mean"].map(C.pct)
    dd.columns = ["신호일", "거래", "수익률 합", "평균"]
    yy = pd.DataFrame({"해": yr.index, "규칙": yr["rule"].map(C.pct), "보유": yr["hold"].map(C.pct),
                       "청산 거래": yr["trades"].fillna(0).astype(int), "거래당 평균": yr["mean"].map(lambda x: C.pct(x, 2))})
    ss = pd.DataFrame({"구간": st["window"], "규칙 수익": st["rule_ret"].map(C.pct), "규칙 낙폭": st["rule_mdd"].map(C.pct),
                       "보유 수익": st["hold_ret"].map(C.pct), "보유 낙폭": st["hold_mdd"].map(C.pct),
                       "진입 거래": st["trades"], "거래당 평균": st["mean"].map(lambda x: C.pct(x, 2)), "승률": st["win"].map(lambda x: f"{x:.0%}")})
    dsr_t = pd.DataFrame([{"판본": k, "진입일": int(r["T"]), "진입일 샤프": f"{r['sr']:+.3f}", "왜도": C.num(r["skew"]), "첨도": C.num(r["kurt"]),
                           "기대 최대(N 3)": C.num(r["sr0"], 3), "DSR": C.num(r["dsr"], 3),
                           "기대 최대(N 1,443, bnf V)": C.num(r.get("sr0_sens"), 3), "DSR(민감도)": C.num(r.get("dsr_sens"), 3)}
                          for k, r in ds.iterrows()])
    md = ["# 나라별 ETF 폭락일 매수 결과", "",
          f"19개 나라 ETF, 곡선 {cfg['data']['start']} → {cfg['data']['end']}, 편도 비용 {cfg['cost']:.2%}, 현금 0%. 지시서 `SPEC.md`.", "",
          f"**판정: {verdict}**", "", C.md_table(vd), "",
          "## 바구니 (판정 판본과 이웃)", "", C.md_table(basket_rows(["95", "93", "97"])), "",
          f"- 노출(칸이 ETF 를 들고 있는 날의 비율) 평균 {ex['exposure_basket']:.1%}, 한 해 거래 {ex['trades_per_year']:.0f}건(19칸 합).",
          f"- 25일선 회복으로 끝난 거래 {ex['kind_ma']:.0%} 는 평균 {C.pct(ex['mean_ma'], 2)}, 20일째에 끝난 거래 {ex['kind_max']:.0%} 는 평균 {C.pct(ex['mean_max'], 2)}.",
          f"- 합친 거래의 손익분기 비용은 편도 {ex['breakeven_pooled'] * 1e4:.0f}bp 다(실제 12bp).", "",
          "## 비용과 민감도 (95, 민감도는 판정 제외)", "",
          C.md_table(basket_rows(["95|비용 2배", "95|비용 0", "95|현금 2%", "95|하한 없음", "95|2004 시작"])), "",
          "## 종목별 (95)", "", C.md_table(tick), "",
          "## 무작위 진입", "",
          f"종목마다 같은 거래 수, 같은 보유일 분포로 살 수 있는 날에 무작위로 {len(rnd)}번 놓았다. 바구니 연 수익률: 최소 {C.pct(rnd['cagr'].min(), 2)}, "
          f"중앙값 {C.pct(rnd['cagr'].median(), 2)}, 95 백분위 {C.pct(p95, 2)}, 최대 {C.pct(rnd['cagr'].max(), 2)}. 규칙 {C.pct(main_['cagr'], 2)} 는 {place:.0%} 위치다.",
          f"거래당 평균으로 보면 무작위 중앙값 {C.pct(rnd['mean'].median(), 2)}, 규칙 {C.pct(main_['mean'], 2)} 로 {(rnd['mean'] < main_['mean']).mean():.0%} 위치다(참고).", "",
          "## Deflated Sharpe (진입일 단위)", "", C.md_table(dsr_t), "",
          "오른쪽 두 열은 민감도(판정 제외)다.", "",
          "## 신호일 쏠림", "",
          f"신호일 {int(dr['signal_days'])}개, 거래 {int(dr['trades'])}건, 수익률 합 {dr['all_total']:+.2f}. 가장 좋은 5일의 합 {dr['best_total']:+.2f}.",
          f"같은 날 5개 이상의 나라에서 신호가 난 날은 {int(dr['days_with_5plus'])}일(거래 {int(dr['trades_on_5plus_days'])}건)이고 그 거래의 평균은 "
          f"{C.pct(dr['mean_on_5plus_days'], 2)}, 나머지 날의 거래는 {C.pct(dr['mean_on_other_days'], 2)} 다.", "",
          "가장 좋은 5일:", "", C.md_table(dd.head(5)), "", "가장 나쁜 5일:", "", C.md_table(dd.tail(5).iloc[::-1]), "",
          "## 해마다", "", C.md_table(yy), "",
          "## 큰 하락 구간", "", C.md_table(ss), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
