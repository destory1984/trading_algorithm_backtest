"""SPEC 6: verdicts under the main assumption (and, not judged, under each sensitivity), and the report.
Run after metrics.py and stress.py.
Usage: python -m src.report   -> results/verdicts.csv, results/report.md
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.metrics import stretches
from src.run import assumptions

CRIT = {1: "낙폭 ≤ 보유 70%", 2: "수익 -1.5%p 이내", 3: "공짜 아님", 4: "시대 4개", 5: "이웃", 6: "거래 ≤ 12"}


def verdict(cfg: dict, m: pd.DataFrame, a: str) -> dict:
    ps = cfg["pass"]
    rc = cfg["rule"]
    sub = m[(m["rule"] == "T3") & (m["assumption"] == a)]
    get = lambda t, s: sub[(sub["target"] == t) & (sub["stretch"] == s)].iloc[0]
    c1 = lambda x: x["mdd_ratio"] <= ps["mdd_ratio"]
    c3 = lambda x: x["diff_risk"] > 0
    f = get(rc["main"], "전체")
    eras = [get(rc["main"], e) for e in cfg["data"]["eras"]]
    ok = {1: c1(f), 2: f["diff_hold"] >= -ps["cagr_gap"], 3: c3(f),
          4: all(x["mdd"] > x["hold_mdd"] and x["diff_hold"] >= -ps["era_cagr_gap"] for x in eras),
          5: all(c1(get(t, "전체")) and c3(get(t, "전체")) for t in rc["neighbors"]),
          6: f["trades_per_year"] <= ps["trades_per_year"]}
    return {"assumption": a, **{f"c{i}": bool(v) for i, v in ok.items()}, "passed": all(ok.values()),
            "failed": ", ".join(str(i) for i, v in ok.items() if not v)}


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    m = pd.read_csv(out / "metrics.csv")
    st = pd.read_csv(out / "stress.csv")
    spy = pd.read_parquet(out / "spy_daily.parquet")
    spy_n = pd.read_csv(out / "spy_trades.csv", index_col=0)["trades"]
    rc = cfg["rule"]
    asm = assumptions(cfg)
    vs = pd.DataFrame([verdict(cfg, m, a) for a in asm])
    vs.to_csv(out / "verdicts.csv", index=False)
    v0 = vs.iloc[0]
    main_a = m[(m["assumption"] == "본")]
    f = main_a[(main_a["rule"] == "T3") & (main_a["target"] == rc["main"]) & (main_a["stretch"] == "전체")].iloc[0]
    label = lambda r, t: "보유" if r == "HOLD" else f"T3 {t:.0%}"

    L = ["# 변동성 목표 과거 검증 결과", "",
         f"^GSPC {cfg['data']['start']} → {cfg['data']['end']}. 본 가정: 배당 연 {cfg['dividend']:.1%}, "
         f"현금 연 {cfg['cash']:.1%}, 편도 비용 {cfg['cost']:.2%}, 신호일 종가로 정하고 다음 날 종가에 사고판다. "
         "지시서 SPEC.md.", "", "## 판정 (본 가정)", ""]
    ps = cfg["pass"]
    nb = {t: main_a[(main_a["target"] == t) & (main_a["stretch"] == "전체")].iloc[0] for t in rc["neighbors"]}
    eras = main_a[(main_a["rule"] == "T3") & (main_a["target"] == rc["main"]) & (main_a["stretch"] != "전체")]
    detail = {
        1: f"최대 낙폭 {C.pct(f['mdd'])}, 보유 {C.pct(f['hold_mdd'])}의 {f['mdd_ratio']:.1%} (한도 {ps['mdd_ratio']:.0%})",
        2: f"연 수익률 {C.pct(f['cagr'])}, 보유 {C.pct(f['hold_cagr'])} 대비 {C.pp(f['diff_hold'])} (한도 -{ps['cagr_gap']:.1%}p)",
        3: f"낙폭 맞춘 고정 비중(w {f['risk_w']:.2f}) {C.pct(f['risk_cagr'])} 대비 {C.pp(f['diff_risk'])}",
        4: " / ".join(f"{e.stretch}: 낙폭 {C.pct(e.mdd, 0)} 대 {C.pct(e.hold_mdd, 0)}, 수익 {C.pp(e.diff_hold)}"
                      for e in eras.itertuples()),
        5: ", ".join(f"{t:.0%}: 낙폭 비율 {x['mdd_ratio']:.1%}, 고정 비중 대비 {C.pp(x['diff_risk'])}" for t, x in nb.items()),
        6: f"한 해 {f['trades_per_year']:.1f}번",
    }
    L += [C.md_table(pd.DataFrame([{"기준": f"{i} {CRIT[i]}", "결과": "O" if v0[f"c{i}"] else "X", "내용": detail[i]}
                                   for i in CRIT])), "",
          f"**판정: {'통과' if v0['passed'] else '탈락'}**" + ("" if v0["passed"] else f" (못 넘은 기준 {v0['failed']})"), ""]

    L += ["## 가정 민감도 (판정 제외)", ""]
    rows = []
    for x in vs.itertuples():
        r = m[(m["rule"] == "T3") & (m["target"] == rc["main"]) & (m["assumption"] == x.assumption) & (m["stretch"] == "전체")].iloc[0]
        rows.append({"가정": x.assumption, "T3 연 수익률": C.pct(r["cagr"]), "보유": C.pct(r["hold_cagr"]),
                     "T3 낙폭": C.pct(r["mdd"]), "보유 낙폭": C.pct(r["hold_mdd"]), "낙폭 비율": f"{r['mdd_ratio']:.1%}",
                     "고정 비중 대비": C.pp(r["diff_risk"]),
                     **{f"{i}": "O" if getattr(x, f"c{i}") else "X" for i in CRIT},
                     "기준대로 보면": "통과" if x.passed else f"탈락 ({x.failed})"})
    L += [C.md_table(pd.DataFrame(rows)), ""]
    flips = [f"{i}" for i in (2, 3) if len(set(vs[f"c{i}"])) > 1]
    L += [("기준 " + "·".join(flips) + " 이 가정에 따라 바뀐다: 배당·현금 가정에 달림." if flips
           else "기준 2·3 은 세 가정에서 모두 같다.") +
          (" 기준 1 은 가정에 따라 바뀐다(지시서가 '가정에 달림'을 적게 한 것은 기준 2·3 이라 참고로 적는다)."
           if len(set(vs["c1"])) > 1 else ""), ""]

    L += ["## 전체 표 (본 가정)", ""]
    for s in stretches(cfg):
        sub = main_a[main_a["stretch"] == s]
        rows = [{"규칙": label(r.rule, r.target), "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd),
                 "샤프": C.num(r.sharpe), "칼마": C.num(r.calmar), "평균 비중": f"{r.avg_weight:.0%}",
                 "한 해 거래": C.num(r.trades_per_year, 1), "최장 낙폭 기간(일)": r.underwater_days,
                 "보유 대비": C.pp(r.diff_hold), "낙폭 비율": f"{r.mdd_ratio:.0%}",
                 "낙폭 맞춘 고정 비중 대비": C.pp(r.diff_risk), "평균 비중 맞춘 고정 비중 대비": C.pp(r.diff_avgw)}
                for r in sub.itertuples()]
        L += [f"### {s} ({sub['first_day'].iloc[0]} → {sub['last_day'].iloc[0]})", "", C.md_table(pd.DataFrame(rows)), ""]

    L += ["## 큰 하락 구간 (본 가정)", "",
          f"낙폭은 구간 안, 수익은 구간 첫날부터 구간이 끝난 뒤 {cfg['stress']['after_days']}거래일까지.", ""]
    rows = []
    for w in cfg["stress"]["windows"]:
        h = st[(st["window"] == w) & (st["rule"] == "HOLD")].iloc[0]
        t = st[(st["window"] == w) & (st["target"] == rc["main"])].iloc[0]
        rows.append({"구간": w, "보유 낙폭": C.pct(h["mdd"]), "T3 낙폭": C.pct(t["mdd"]),
                     "보유 수익": C.pct(h["return_to_after"]), "T3 수익": C.pct(t["return_to_after"]),
                     "T3 평균 비중": f"{t['avg_weight']:.0%}"})
    L += [C.md_table(pd.DataFrame(rows)), ""]

    L += ["## 참고: 2008 → 2026 SPY (판정 제외)", "",
          "이 폴더 코드로 SPY 를 돌린 것. 다음 날 시가 결과가 index_timing 의 T3 와 같은지는 checks.md 검증 2 에 있다.", ""]
    rows = []
    for k in spy.columns:
        _, t, ex = k.split("|")
        d = spy[k]
        rows.append({"규칙": "보유" if t == "HOLD" else f"T3 {float(t):.0%}", "실행": "다음 날 " + ("시가" if ex == "open" else "종가"),
                     "연 수익률": C.pct(d.pipe(C.cagr), 2), "최대 낙폭": C.pct(C.mdd(d), 2), "거래": int(spy_n[k])})
    L += [C.md_table(pd.DataFrame(rows)), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(vs.to_string(index=False))


if __name__ == "__main__":
    main()
