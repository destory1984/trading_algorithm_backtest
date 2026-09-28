"""results/report.md from metrics.csv, neighbors.csv, stress.csv (SPEC 6). Run last.
Usage: python -m src.report   -> results/report.md, verdicts.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C

pct, num, md = C.pct, C.num, C.md_table


def pick(met, m, r, p, s, dv):
    x = met[(met["market"] == m) & (met["rule"] == r) & (met["stretch"] == s)]
    x = x[x["param"].astype(str).isin([str(p), str(float(p)) if p != "" else ""])] if r != "HOLD" else x
    x = x[x["dividend"].isna()] if dv is None else x[x["dividend"].round(6) == round(dv, 6)]
    assert len(x) == 1, (m, r, p, s, dv, len(x))
    return x.iloc[0]


def judge(cfg, met, nb, m, r, dv):
    p = cfg["pass"]
    main = cfg["rules"][r]["main"]
    f = pick(met, m, r, main, "full", dv)
    halves = [pick(met, m, r, main, s, dv) for s in ("first", "second")]
    nbr = nb[(nb["market"] == m) & (nb["rule"] == r)]
    c = {
        "1": (abs(f["mdd"]) <= p["mdd_ratio"] * abs(f["hold_mdd"]), f"{pct(f['mdd'])} (보유의 {f['mdd_ratio']:.0%})"),
        "2": (f["cagr"] >= f["hold_cagr"] - p["cagr_gap"], f"{pct(f['cagr'])} ({pct(f['diff_hold'])}p)"),
        "3": (f["cagr"] > f["risk_cagr"], f"{pct(f['diff_risk'])}p (고정 비중 w {f['risk_w']:.2f})"),
        "4": (all(abs(h["mdd"]) < abs(h["hold_mdd"]) and h["cagr"] >= h["hold_cagr"] - p["half_cagr_gap"] for h in halves),
              " / ".join(f"낙폭 {pct(h['mdd'], 0)} 대 {pct(h['hold_mdd'], 0)}, 수익 {pct(h['diff_hold'])}p" for h in halves)),
        "5": (bool(nbr["ok"].all()) and len(nbr) == 2, ", ".join(f"{v:g}: {'○' if ok else '×'}" for v, ok in zip(nbr["param"], nbr["ok"]))),
        "6": (f["trades_per_year"] <= p["trades_per_year"], f"{f['trades_per_year']:.1f}번"),
    }
    return c, f


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    met = pd.read_csv(out / "metrics.csv")
    nb = pd.read_csv(out / "neighbors.csv")
    stress = pd.read_csv(out / "stress.csv")
    kr_dv = cfg["markets"]["KR"]["dividend"]
    names = {"1": "낙폭 ≤ 보유의 70%", "2": "수익 차이 ≥ -1.5%p", "3": "낙폭 맞춘 고정 비중보다 수익 높음",
             "4": "전반·후반 모두", "5": "이웃 2개 (기준 1·3)", "6": "한 해 12번 이하"}
    L = ["# 지수 낙폭 줄이기 보고서", "",
         "규칙: T1 200일선, T2 10개월선(월말), T3 변동성 목표 15%(주말, 10%p 넘게 바뀔 때만). 신호일 종가로 정하고 다음 날 시가에 사고판다. 쉬는 돈은 현금 0%. 지시서는 `SPEC.md`.", "",
         f"미국은 SPY(배당 반영, 편도 비용 0.12%), 한국은 코스피 가격 지수 + 연 {kr_dv:.1%} 배당 근사(편도 비용 0.02%)다.", ""]
    verdicts, rows = [], []
    for m in C.MARKETS:
        dv = None if m == "US" else kr_dv
        for r, rc in cfg["rules"].items():
            c, f = judge(cfg, met, nb, m, r, dv)
            ok = all(v[0] for v in c.values())
            note = ""
            if m == "KR":
                flips, moved = [], []
                for sdv in cfg["markets"]["KR"]["dividend_sens"]:
                    cs, _ = judge(cfg, met, nb, m, r, sdv)
                    if all(v[0] for v in cs.values()) != ok:
                        flips.append(f"배당 {sdv:.1%}")
                    elif (cs["2"][0], cs["3"][0]) != (c["2"][0], c["3"][0]):
                        moved.append(f"{sdv:.1%}")
                note = ("배당 가정에 달림: " + ", ".join(flips)) if flips else                     (f"기준 2·3 이 배당 {', '.join(moved)} 에서 바뀌지만 판정은 같다" if moved else "")
            verdicts.append({"market": m, "rule": r, "pass": ok, **{f"c{k}": v[0] for k, v in c.items()}, "note": note})
            rows.append({"시장": cfg["markets"][m]["name"], "규칙": f"{r} {rc['name']}", **{f"{k}. {names[k]}": ("○ " if v[0] else "× ") + v[1] for k, v in c.items()},
                         "판정": "통과" if ok else "탈락", "메모": note})
    vd = pd.DataFrame(verdicts)
    vd.to_csv(out / "verdicts.csv", index=False)
    passed = vd[vd["pass"]]
    L += ["## 1. 판정", "", md(pd.DataFrame(rows)), ""]
    L += [("**통과: " + ", ".join(f"{cfg['markets'][x['market']]['name']} {x['rule']} {cfg['rules'][x['rule']]['name']}" for _, x in passed.iterrows()) + "**")
          if len(passed) else "**통과한 규칙 × 시장이 없다.**", ""]

    cols = [("cagr", "연 수익률", pct), ("mdd", "최대 낙폭", pct), ("sharpe", "샤프", num), ("calmar", "칼마", num),
            ("avg_weight", "평균 비중", lambda v: f"{v:.0%}"), ("trades_per_year", "한 해 거래", lambda v: f"{v:.1f}"),
            ("underwater_days", "가장 긴 회복(일)", lambda v: str(int(v))), ("hold_cagr", "보유 연", pct), ("hold_mdd", "보유 낙폭", pct),
            ("risk_cagr", "낙폭 맞춘 고정 비중", pct), ("diff_risk", "차이", pct), ("avgw_cagr", "평균 비중 고정", pct), ("diff_avgw", "차이 ", pct)]
    for m in C.MARKETS:
        dv = None if m == "US" else kr_dv
        L += [f"## 2. {cfg['markets'][m]['name']} 지표", ""]
        for s in C.STRETCHES:
            t = []
            for r in ["HOLD", *cfg["rules"]]:
                x = pick(met, m, r, "" if r == "HOLD" else cfg["rules"][r]["main"], s, dv)
                t.append({"규칙": "보유" if r == "HOLD" else f"{r} {cfg['rules'][r]['name']}", **{n: f(x[k]) for k, n, f in cols}})
            L += [f"### {C.NAMES[s]} ({x['first_day']} → {x['last_day']})", "", md(pd.DataFrame(t)), ""]

    t = pd.DataFrame([{"시장": cfg["markets"][x["market"]]["name"], "규칙": x["rule"], "값": f"{x['param']:g}", "연 수익률": pct(x["cagr"]),
                       "최대 낙폭": pct(x["mdd"]), "보유 대비 낙폭": f"{x['mdd_ratio']:.0%}", "고정 비중 대비": pct(x["diff_risk"]),
                       "기준 1·3": "○" if x["ok"] else "×"} for _, x in nb.iterrows()])
    L += ["## 3. 이웃 값", "", md(t), ""]

    rows = []
    for sdv in [kr_dv, *cfg["markets"]["KR"]["dividend_sens"]]:
        for r in cfg["rules"]:
            x = pick(met, "KR", r, cfg["rules"][r]["main"], "full", sdv)
            rows.append({"배당(연)": f"{sdv:.1%}", "규칙": r, "연 수익률": pct(x["cagr"]), "보유": pct(x["hold_cagr"]),
                         "수익 차이": pct(x["diff_hold"]), "고정 비중 대비": pct(x["diff_risk"]), "최대 낙폭": pct(x["mdd"])})
    L += ["## 4. 한국 배당 가정 민감도 (전체 구간)", "", md(pd.DataFrame(rows)), ""]

    t = pd.DataFrame([{"시장": cfg["markets"][x["market"]]["name"], "구간": x["window"], "규칙": x["rule"], "보유 낙폭": pct(x["hold_dd"]),
                       "규칙 낙폭": pct(x["rule_dd"]), "평균 비중": f"{x['avg_weight']:.0%}", "보유 수익": pct(x["hold_ret"]),
                       "규칙 수익": pct(x["rule_ret"]), "수익 끝날": x["to"]} for _, x in stress.iterrows()])
    L += ["## 5. 큰 하락 구간", "", f"낙폭은 구간 안, 수익은 구간 첫날부터 구간이 끝난 뒤 {cfg['stress']['after_days']}거래일까지다.", "", md(t), ""]

    L += ["## 6. 참고 (판정 제외)", "", f"- {cfg['reference']['bnf_walkforward']}", "",
          "## 7. 시행 수와 봉인", "",
          f"- 본 규칙 6개 + 이웃 12개 = 18개. 한국 배당 민감도는 같은 규칙의 가정만 바꾼 것이라 세지 않았다.",
          f"- 가장 늦은 데이터 날짜: 미국 {cfg['markets']['US']['end']}, 한국 {cfg['markets']['KR']['end']}. 그 뒤 데이터는 쓰지 않았다(`checks.md` 1장).", ""]
    concl = ("통과: " + ", ".join(f"{cfg['markets'][x['market']]['name']} {x['rule']} {cfg['rules'][x['rule']]['name']}" for _, x in passed.iterrows())
             + f". 나머지 {len(vd) - len(passed)}개는 탈락이다.") if len(passed) else "6개 규칙 × 시장 모두 탈락이다."
    L += ["## 8. 결론", "", concl, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(vd.to_string(index=False))
    print(concl)


if __name__ == "__main__":
    main()
