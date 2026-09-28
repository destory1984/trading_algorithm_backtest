"""results/report.md from the csv files of the other stages.

Usage:  python -m src.report
"""
from __future__ import annotations

import pandas as pd

from .common import load_config, md_table, num, pct, results_dir, share

SIZING = {"full": "전액", "0.2": "변동성 0.2", "0.3": "변동성 0.3", "0.4": "변동성 0.4"}


def decision(cfg: dict, verdicts: dict, g: pd.DataFrame, at: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """The spec's pre-set rule, per open / with-costs / vol-target row (both cash variants)."""
    base = at[at["capital_krw"] == cfg["tax"]["base_capital_krw"]].set_index("combo")
    r = g[(g["exec"] == "open") & (g["costs"] == "with") & (g["sizing"] != "full")].copy()
    r["sharpe_ok"] = r["sharpe"] > r["u1x_sharpe"]
    r["mdd_ok"] = r["mdd"] > cfg["decision"]["max_mdd"]
    r["tax_ok"] = base.loc[r.index, "post_cagr"] >= base.loc[r.index, "u1x_post_cagr"]
    r["pass"] = r["sharpe_ok"] & r["mdd_ok"] & r["tax_ok"]
    claims_ok = verdicts.get("C1") == "재현됨" and verdicts.get("C3") == "재현됨"
    passed = r[r["pass"]]
    if claims_ok and len(passed):
        names = ", ".join(f"{x.ticker} {SIZING[x.sizing]} 현금 {x.cash}" for x in passed.itertuples())
        line = f"신호 알림 후보로 올린다: {names}."
    else:
        why = []
        if not claims_ok:
            why.append(f"C1 {verdicts.get('C1')}, C3 {verdicts.get('C3')}")
        if not len(passed):
            why.append(f"기준 셋을 모두 넘는 조합 0/{len(r)} (샤프 {int(r['sharpe_ok'].sum())}, 낙폭 {int(r['mdd_ok'].sum())}, 세후 {int(r['tax_ok'].sum())})")
        line = "신호 알림 후보로 올리지 않는다: " + "; ".join(why) + "."
    return r, line


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    s0 = pd.read_csv(rd / "stage0.csv")
    xc = pd.read_csv(rd / "crosscheck.csv")
    vd = pd.read_csv(rd / "claims_verdicts.csv")
    cl = pd.read_csv(rd / "claims.csv")
    g = pd.read_csv(rd / "lev.csv", index_col="combo", dtype={"sizing": str})
    at = pd.read_csv(rd / "aftertax.csv", dtype={"sizing": str})
    verdicts = dict(zip(vd["claim"], vd["verdict"]))
    dr, line = decision(cfg, verdicts, g, at)  # open / with costs / vol-target rows, both cash variants
    starts = g.groupby("ticker")["start"].first()
    end = g["end"].iloc[0]
    out = ["# 레버리지 ETF IBS 백테스트 결과", "",
           f"**실제 상장 구간만 썼다**: TQQQ {starts['TQQQ']}부터, SPXL {starts['SPXL']}부터, SOXL {starts['SOXL']}부터 (끝 {end}). "
           "닷컴 폭락(2000 → 2002)은 없고, 2008 금융위기는 SPXL 의 상장 뒤 몇 달만 들어 있다. "
           "합성 과거 구간 수치는 단계 0 의 원본 패키지 결과로만 인용한다. "
           f"달러 손익은 고정 환율 {cfg['tax']['fx']:,}원으로 바꿨고 환율 변동은 무시했다. 비용 없는 결과는 원본과 비교용이다. "
           f"단계 1 → 3 은 {cfg['data']['end']} 을 포함해 끝난다. 단계 0 의 원본 패키지는 `--end {cfg['stage0']['end']}` 이 그날을 빼서 "
           f"하루 앞({s0['last'].iloc[0]})에서 끝난다.", ""]

    # 1. stage 0
    t = pd.DataFrame({"구간·사이징": s0["label"], "기간": s0["first"] + " → " + s0["last"],
                      "연 수익률 원본/재현": s0["orig_cagr"].map(pct) + " / " + s0["cagr"].map(pct),
                      "샤프 원본/재현": s0["orig_sharpe"].map(num) + " / " + s0["sharpe"].map(num),
                      "최대 낙폭 원본/재현": s0["orig_mdd"].map(pct) + " / " + s0["mdd"].map(pct),
                      "거래 원본/재현": s0["orig_trades"].map(lambda v: "" if pd.isna(v) else f"{int(v)}") + " / " + s0["trades"].map(lambda v: f"{int(v)}"),
                      "노출 원본/재현": s0["orig_exposure"].map(share) + " / " + s0["exposure"].map(share)})
    x = pd.DataFrame({"출처": xc["source"], "거래": xc["trades"], "연 수익률": xc["cagr"].map(pct),
                      "샤프": xc["sharpe"].map(num), "최대 낙폭": xc["mdd"].map(pct), "노출": xc["exposure"].map(share)})
    out += [f"## 1. 단계 0: 원본 패키지 재현 (--end {cfg['stage0']['end']}, 비용 0, 현금 ^IRX)", "", md_table(t), "",
            "우리 엔진과 원본 패키지 대조 (TQQQ 실제 구간, 비용 0, ^IRX, 전액, open, 1만 달러):", "", md_table(x), ""]

    # 2. claims
    c1a = cl[(cl["claim"] == "C1") & (cl["variant"] == "all")].set_index("ticker")
    gap_txt = ", ".join(f"{k} {int(round(c1a.at[k, 'days'] * c1a.at[k, 'gap_share']))}일" for k in cfg["claims"]["c1_tickers"])
    out += ["## 2. 단계 1: 주장별 판정", "", md_table(vd.rename(columns={"claim": "#", "title": "주장", "verdict": "판정", "original": "원본 수치"})),
            "", "주장별 표와 판정 기준은 `results/claims.md` 에 있다. "
            f"C1 은 원본(raw, 조정 전) 가격으로 갭(시가 = 전날 종가) 여부를 가른다: 갭인 날(시가 = 전날 종가)이 {gap_txt} 이다. "
            "no_gap 변형은 이 날들만 빼고, 십분위 등 구간의 경계값은 전체 표본(all)에서 잡은 그대로 쓴다.", ""]

    # 3. stage 2 with pre/after tax at the base capital
    base = at[at["capital_krw"] == cfg["tax"]["base_capital_krw"]].set_index("combo")
    wu = g[g["costs"] == "with"].set_index(["ticker", "exec", "sizing", "cash"])
    zu = g[g["costs"] == "zero"].set_index(["ticker", "exec", "sizing", "cash"])
    u1x_cost_gap = (zu["u1x_cagr"] - wu["u1x_cagr"]).median()
    capped_n, total_n = int(g["k1x_capped"].sum()), len(g)
    irx_dr = dr[dr["cash"] == "irx"]
    idle_lo, idle_hi = float((1 - irx_dr["exposure"]).min()), float((1 - irx_dr["exposure"]).max())
    vt_dr = dr[dr["sizing"] != "full"]
    held_cash_lo, held_cash_hi = float((1 - vt_dr["avg_weight"]).min()), float((1 - vt_dr["avg_weight"]).max())
    beat = int(dr["sharpe_ok"].sum())
    beat_txt = "그런데도 1배 보유 샤프를 넘는 행은 없다." if beat == 0 else f"그런데도 1배 보유 샤프를 넘는 행은 {beat}/{len(dr)}건뿐이다."
    out += ["## 3. 단계 2: 조합표 (비용 포함)", "", "세후는 1억 원 기준. 1배 = 기초지수 1배 ETF 보유 (TQQQ→QQQ, SPXL→SPY, SOXL→SOXX), "
            "k배 = 1배 ETF 를 전략의 최대 낙폭에 맞춘 비중 k(1 이하, 나머지는 현금)로 보유. k배 곡선에는 매매 비용이 없는데, "
            "종목 보유와 1배 보유는 진입·청산 비용을 내고 k배는 내지 않는다(비용이 1배 보유 연 수익률을 중앙값 "
            f"{pct(u1x_cost_gap, 2)}p 깎는다). "
            f"`1.00*` 는 전략 낙폭이 1배 ETF 보다 깊어 k 가 상한 1 에 걸린 행이다(이때 k배 = 1배 보유, 낙폭은 맞지 않는다; 전체 {total_n}행 중 {capped_n}행). "
            "샤프는 무위험금리를 빼지 않은 값이다(원본과 같은 정의). 현금 ^IRX 행은 대기 중에도 이자를 받아, "
            f"판단 기준 조합(open·비용 포함·변동성 사이징)에서 거래일의 {idle_lo:.0%} → {idle_hi:.0%} 를 현금으로 대기하면서도 그 이자로 1배 보유보다 샤프가 유리해진다. "
            f"변동성 사이징 행은 보유 중에도 비중 w(사이징의 목표 변동성으로 정해진다)만 투자되고 나머지(평균 {held_cash_lo:.0%} → {held_cash_hi:.0%})는 보유일에도 현금으로 남아 마찬가지로 이자를 받는다. "
            + beat_txt, ""]
    for tk in cfg["lev"]["tickers"]:
        r = g[(g["ticker"] == tk) & (g["costs"] == "with")]
        tt = pd.DataFrame({"체결": r["exec"], "사이징": r["sizing"].map(SIZING), "현금": r["cash"],
                           "연 수익률 세전/세후": r["cagr"].map(pct) + " / " + base.loc[r.index, "post_cagr"].map(pct),
                           "샤프": r["sharpe"].map(num), "최대 낙폭": r["mdd"].map(pct), "노출": r["exposure"].map(share),
                           "거래": r["trades"], "종목 보유": r["hold_cagr"].map(pct) + " / " + r["hold_mdd"].map(pct),
                           "1배 보유 연수익/샤프": r["u1x_cagr"].map(pct) + " / " + r["u1x_sharpe"].map(num),
                           "k배": r["k1x"].map(num) + r["k1x_capped"].map({True: "*", False: ""}),
                           "k배 연수익": r["k1x_cagr"].map(pct)})
        out += [f"### {tk}", "", md_table(tt), ""]
    z = g[g["costs"] == "zero"]
    out += ["비용 0 결과는 `results/lev.csv` 의 costs = zero 행에 있다. 비용이 줄인 연 수익률의 중앙값: "
            f"{pct((z.set_index(['ticker','exec','sizing','cash'])['cagr'] - g[g['costs']=='with'].set_index(['ticker','exec','sizing','cash'])['cagr']).median())}p.", ""]

    # 4. stage 3
    sel = at[(at["exec"] == "open") & (at["costs"] == "with") & (at["cash"] == "irx")]
    pv = sel.pivot_table(index=["ticker", "sizing"], columns="capital_krw", values="post_cagr").reset_index()
    bh = sel.pivot_table(index=["ticker", "sizing"], columns="capital_krw", values="u1x_post_cagr").reset_index()
    pv["ticker"] = pd.Categorical(pv["ticker"], categories=cfg["lev"]["tickers"], ordered=True)
    pv["sizing"] = pd.Categorical(pv["sizing"], categories=cfg["lev"]["sizing"], ordered=True)
    pv = pv.sort_values(["ticker", "sizing"]).reset_index(drop=True)
    bh["ticker"] = pd.Categorical(bh["ticker"], categories=cfg["lev"]["tickers"], ordered=True)
    bh["sizing"] = pd.Categorical(bh["sizing"], categories=cfg["lev"]["sizing"], ordered=True)
    bh = bh.sort_values(["ticker", "sizing"]).reset_index(drop=True)
    caps = cfg["tax"]["capitals_krw"]
    name = {30_000_000: "3천만 원", 100_000_000: "1억 원", 300_000_000: "3억 원"}
    tt = pd.DataFrame({"종목": pv["ticker"], "사이징": pv["sizing"].map(SIZING),
                       **{f"세후 {name.get(c, f'{c:,}원')}": pv[c].map(pct) for c in caps},
                       **{f"1배 보유 세후 {name.get(c, f'{c:,}원')}": bh[c].map(pct) for c in caps}})

    base_open = sel[sel["capital_krw"] == cfg["tax"]["base_capital_krw"]]
    base_open_min = base_open["min_cash"].min()
    if base_open_min >= 0:
        cash_line = ("이 표의 열두 조합(1억 원, open·비용·^IRX 행)은 세금 낼 때도 현금이 음수가 된 적이 없다"
                     f"(가장 낮을 때도 {base_open_min:,.0f} 달러; 전액 사이징 행은 진입 때 현금을 모두 써 0에 붙는다).")
    else:
        cash_line = f"가장 낮은 현금(1억 원, open·비용·^IRX 행): {base_open_min:,.0f} 달러."

    neg = at[at["min_cash"] < 0]
    neg_vt = neg[neg["sizing"] != "full"]
    vt_ex = neg_vt.loc[neg_vt["min_cash"].idxmin()] if len(neg_vt) else None
    vt_txt = (f" 전액 사이징뿐 아니라 변동성 사이징에서도 일어난다(예: {vt_ex['ticker']} {vt_ex['exec']} "
              f"{SIZING[vt_ex['sizing']]}, 자본 {int(vt_ex['capital_krw']):,}원, 최저 현금 {vt_ex['min_cash']:,.0f} 달러)."
              if vt_ex is not None else "")
    out += ["## 4. 단계 3: 세후 비교 (open, 비용 포함, 현금 ^IRX)", "",
            f"양도소득세 {cfg['tax']['rate']:.0%}, 해마다 기본공제 {cfg['tax']['deduction_krw']:,}원, 같은 해 손익 통산. "
            f"Y 년 세금은 Y+1 년 {cfg['tax']['pay_month']}월 첫 거래일에 계좌에서 뺀다. 그때까지 그 돈은 계좌에 남아 현금 이자를 받거나 투자된다. "
            f"마지막 해({cfg['data']['end'][:4]}년)는 표본 안에 다음 해 5월이 없어 마지막 날({cfg['data']['end']})에 뺀다. "
            "보유 전략은 마지막 날 한 번에 팔아 그날 한 번 낸다. 현금 이자에 붙는 세금은 넣지 않았다. "
            "세금 납부일에 그전에 산 포지션이 아직 열려 있으면(포지션을 정리해 세금을 낼 수 없으니) 세금만큼 현금이 음수가 된다"
            "(평가금액에서 뺀 것으로 본다; 음수 현금에도 ^IRX 이자가 붙는다)." + vt_txt + " " + cash_line,
            "", md_table(tt), ""]

    all_neg = at.loc[at["min_cash"].idxmin()]
    all_neg_in_decision = all_neg["exec"] == "open" and all_neg["costs"] == "with" and all_neg["sizing"] != "full"
    all_neg_scope = "판단 기준 행(open·비용 포함·변동성 목표 사이징)이다" if all_neg_in_decision else "판단 기준 행(open·비용 포함·변동성 목표 사이징)이 아니다"
    out += [f"세후 계산 전체(48개 조합 x 3개 자본 x 비용 있음/없음)에서 가장 낮은 현금은 {all_neg['ticker']} {all_neg['exec']} "
            f"{SIZING[all_neg['sizing']]} 사이징, 비용 {'포함' if all_neg['costs'] == 'with' else '0'}, 현금 {all_neg['cash']}, "
            f"자본 {int(all_neg['capital_krw']):,}원 조합의 {all_neg['min_cash_date']} 로, 이날 현금 {all_neg['min_cash']:,.0f} 달러는 그날 평가금액의 "
            f"{all_neg['min_cash'] / all_neg['min_cash_equity'] * 100:.1f}% 다. "
            f"이 조합은 {all_neg_scope}.", ""]

    # 5. C3 vs bnf
    c3 = cl[(cl["claim"] == "C3") & (cl["variant"] == "no_gap")].set_index("ticker")
    same = all(c3.at[k, "below_mean"] > c3.at[k, "above_mean"] for k in cfg["claims"]["c1_tickers"])
    parts = [f"{k} 200일선 위 {pct(c3.at[k, 'above_mean'], 3)} (t {num(c3.at[k, 'above_t'])}), 아래 {pct(c3.at[k, 'below_mean'], 3)} (t {num(c3.at[k, 'below_t'])})"
             for k in cfg["claims"]["c1_tickers"]]
    other = [k for k in cfg["claims"]["c3_tickers"] if k not in cfg["claims"]["c1_tickers"]]
    other_same = all(c3.at[k, "below_mean"] > c3.at[k, "above_mean"] for k in other)
    other_txt = (f" 나머지 종목({', '.join(other)})도 같은 방향이다(200일선 아래 평균이 위보다 크다)."
                 if other_same else f" 나머지 종목({', '.join(other)}) 중에는 방향이 다른 종목도 있다.")
    out += ["## 5. C3 와 bnf_backtest 비교", "",
            f"C3 판정은 {verdicts.get('C3')} 이다. {'; '.join(parts)}. "
            "bnf_backtest 에서는 지수 120일선 위(상승장)에서만 사는 필터가 비교한 216쌍 모두에서 졌다. "
            + ("IBS 도 엣지가 지수 이동평균선 아래에 몰려 있어 같은 방향이다. 상승장 필터를 걸면 수익이 나는 매수를 버린다."
               if same else "IBS 에서는 이동평균선 아래 쪽 엣지가 더 크지 않아 bnf 와 방향이 다르다.")
            + other_txt, ""]

    # 6. decision and one line
    r = dr
    dt = pd.DataFrame({"종목": r["ticker"], "사이징": r["sizing"].map(SIZING), "현금": r["cash"],
                       "샤프 > 1배": r["sharpe_ok"].map({True: "예", False: "아니오"}),
                       "낙폭 > -40%": r["mdd_ok"].map({True: "예", False: "아니오"}),
                       "세후 ≥ 1배 세후": r["tax_ok"].map({True: "예", False: "아니오"})})
    out += ["## 6. 미리 정한 판단 기준과 한 줄 결론", "", md_table(dt), "", f"**{line}**", ""]
    (rd / "report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
