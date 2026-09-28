"""SPEC 7: verdicts, and the report. Run after metrics.py, stress.py and dsr.py.
Usage: python -m src.report   -> results/verdicts.csv, results/report.md
"""
from __future__ import annotations

import pandas as pd

from src import common as C

CRIT = {1: "공짜 아님", 2: "샤프 +0.10", 3: "낙폭 ≤ SPY 절반", 4: "두 기간", 5: "이웃", 6: "비용 ≤ 1%p"}


def verdicts(cfg: dict, m: pd.DataFrame, cash: float) -> pd.DataFrame:
    ps = cfg["pass"]
    net = m[m["net"] & (m["cash"] == cash)]
    get = lambda r, p, s: net[(net["rule"] == r) & (net["param"] == p) & (net["stretch"] == s)].iloc[0]
    c1 = lambda x: x["diff_match"] > 0
    c3 = lambda x: x["mdd"] >= ps["mdd_ratio"] * x["spy_mdd"]
    rows = []
    for r, spec in cfg["rules"].items():
        f, a, b = (get(r, spec["main"], s) for s in C.STRETCHES)
        nb = [get(r, p, "full") for p in spec["neighbors"]]
        ok = {1: c1(f),
              2: f["diff_ew5_sharpe"] >= ps["sharpe_gap"],
              3: c3(f),
              4: all(c1(x) and x["cagr"] > 0 for x in (a, b)),
              5: all(c1(x) and c3(x) for x in nb),
              6: f["cost_drag"] <= ps["cost_drag"]}
        rows.append({"rule": r, "name": spec["name"], "param": spec["main"], "cash": cash,
                     **{f"c{i}": bool(v) for i, v in ok.items()}, "passed": all(ok.values()),
                     "failed": ", ".join(str(i) for i, v in ok.items() if not v)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    m = pd.read_csv(out / "metrics.csv")
    m["param"] = m["param"].astype("Int64")
    st = pd.read_csv(out / "stress.csv")
    ds = pd.read_csv(out / "dsr.csv")
    tick = cfg["data"]["tickers"]
    v = verdicts(cfg, m, cfg["cash"])
    vs = pd.concat([verdicts(cfg, m, c) for c in cfg["cash_sens"]])
    pd.concat([v, vs]).to_csv(out / "verdicts.csv", index=False)

    net0 = m[m["net"] & (m["cash"] == cfg["cash"])]
    names = {**{r: s["name"] for r, s in cfg["rules"].items()}, **{b: s["name"] for b, s in cfg["baselines"].items()}}
    label = lambda r, p: f"{r} {names[r]}" + ("" if pd.isna(p) else f" ({p})")
    L = ["# 여러 자산 배분 결과", "",
         f"기간 {cfg['data']['start']} → {cfg['data']['end']}, 종목 {', '.join(tick)}, 편도 비용 {cfg['cost']:.2%}, "
         f"현금 연 {cfg['cash']:.1%}. 모든 수치는 비용 뒤다. 지시서 SPEC.md.", ""]

    L += ["## 판정", ""]
    f0 = net0[net0["stretch"] == "full"].set_index(["rule", "param"])
    rows = []
    for x in v.itertuples():
        r = f0.loc[(x.rule, x.param)]
        rows.append({"규칙": label(x.rule, x.param), "연 수익률": C.pct(r["cagr"]), "최대 낙폭": C.pct(r["mdd"]),
                     "샤프 (EW5 대비)": f"{C.num(r['sharpe'])} ({r['diff_ew5_sharpe']:+.2f})",
                     "낙폭 맞춘 고정 비중 대비": C.pp(r["diff_match"]), "SPY 낙폭 대비": f"{r['mdd_ratio_spy']:.0%}",
                     "비용": C.pp(-r["cost_drag"], 2),
                     **{f"{i} {CRIT[i]}": "O" if getattr(x, f"c{i}") else "X" for i in CRIT},
                     "판정": "통과" if x.passed else "탈락"})
    L += [C.md_table(pd.DataFrame(rows)), "",
          f"통과 {int(v['passed'].sum())} / {len(v)}. 기준: (1) 연 수익률 > 낙폭을 맞춘 EW5 × k + 현금, "
          f"(2) 샤프 ≥ EW5 + {cfg['pass']['sharpe_gap']:.2f}, (3) 최대 낙폭 ≤ SPY 의 {cfg['pass']['mdd_ratio']:.0%}, "
          "(4) 전반·후반 각각 기준 1 과 연 수익률 > 0, (5) 이웃 2개 모두 기준 1·3, "
          f"(6) 비용으로 빠지는 연 수익률 ≤ {cfg['pass']['cost_drag']:.1%}p.", ""]

    L += ["## 전체 표 (현금 0%, 비용 뒤)", ""]
    for s in C.STRETCHES:
        sub = net0[net0["stretch"] == s]
        rows = [{"규칙": label(r.rule, r.param), "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd),
                 "샤프": C.num(r.sharpe), "칼마": C.num(r.calmar), "낙폭 맞춘 k": C.num(r.match_k),
                 "그 연 수익률": C.pct(r.match_cagr), "차이": C.pp(r.diff_match),
                 "한 해 비중 변경": C.num(r.changes_per_year, 1), "한 해 회전율": f"{r.turnover_per_year:.0%}",
                 "비용": C.pp(-r.cost_drag, 2), "최장 낙폭 기간(일)": r.underwater_days} for r in sub.itertuples()]
        L += [f"### {C.NAMES[s]} ({sub['first_day'].iloc[0]} → {sub['last_day'].iloc[0]})", "",
              C.md_table(pd.DataFrame(rows)), ""]

    L += ["## 평균 비중 (전체)", ""]
    full = net0[net0["stretch"] == "full"]
    rows = [{"규칙": label(r.rule, r.param), **{t: f"{getattr(r, 'w_' + t):.0%}" for t in tick},
             "현금": f"{r.w_cash:.0%}"} for r in full.itertuples()]
    L += [C.md_table(pd.DataFrame(rows)), ""]

    for c in cfg["cash_sens"]:
        L += [f"## 민감도: 현금 연 {c:.1%} (판정 제외)", ""]
        sub = m[m["net"] & (m["cash"] == c) & (m["stretch"] == "full")]
        vc = vs[vs["cash"] == c].set_index("rule")
        rows = [{"규칙": label(r.rule, r.param), "연 수익률": C.pct(r.cagr), "최대 낙폭": C.pct(r.mdd),
                 "샤프": C.num(r.sharpe), "EW5 대비 샤프": f"{r.diff_ew5_sharpe:+.2f}",
                 "현금 초과 샤프 (EW5 대비)": f"{r.sharpe_ex:.2f} ({r.sharpe_ex - r.ew5_sharpe_ex:+.2f})",
                 "낙폭 맞춘 고정 비중 대비": C.pp(r.diff_match),
                 "기준대로 보면": ("" if r.rule not in vc.index or r.param != cfg["rules"][r.rule]["main"]
                             else ("통과" if vc.loc[r.rule, "passed"] else f"탈락 ({vc.loc[r.rule, 'failed']})"))}
                for r in sub.itertuples()]
        L += [C.md_table(pd.DataFrame(rows)), "",
              "기준 2 의 샤프는 지시서대로 현금 0% 기준이다. 현금에 이자가 붙으면 현금을 많이 드는 규칙의 샤프가 "
              "그만큼 올라가므로, 그 이자를 뺀 초과 샤프를 옆에 적었다.", ""]

    L += ["## 큰 하락 구간 (현금 0%, 비용 뒤)", ""]
    order = [*cfg["rules"], *cfg["baselines"]]
    pv = st.pivot(index="window", columns="rule", values="mdd").reindex(columns=order)
    pr = st.pivot(index="window", columns="rule", values="return").reindex(columns=order)
    rows = [{"구간": w, **{f"{c} 낙폭 / 수익": f"{C.pct(pv.loc[w, c])} / {C.pct(pr.loc[w, c])}" for c in order}}
            for w in cfg["stress"]["windows"]]
    L += [C.md_table(pd.DataFrame(rows)), "", "구간 안의 보유(시작 → 바뀐 날):", ""]
    for w in cfg["stress"]["windows"]:
        for r in cfg["rules"]:
            h = st[(st["window"] == w) & (st["rule"] == r)]["holdings"].iloc[0]
            L.append(f"- {w} {r}: {h}")
    L += [""]

    L += ["## Deflated Sharpe (참고, 판정 제외)", "",
          f"EW5 를 넘는 월 수익의 샤프. 월 {int(ds['months'].iloc[0])}개, 시행 수 {cfg['dsr']['trials']}, "
          f"기대 최대 월 샤프 {ds['sr0_month'].iloc[0]:.3f}.", ""]
    rows = [{"규칙": label(r.rule, r.param), "월 샤프": C.num(r.sr_month, 3), "연환산": C.num(r.sr_year),
             "왜도": C.num(r.skew), "첨도": C.num(r.kurt), "PSR(0)": C.num(r.psr0), "DSR": C.num(r.dsr)}
            for r in ds.itertuples()]
    L += [C.md_table(pd.DataFrame(rows)), ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(v[["rule", *[f"c{i}" for i in CRIT], "passed", "failed"]].to_string(index=False))
    print(vs[["rule", "cash", "passed", "failed"]].to_string(index=False))


if __name__ == "__main__":
    main()
