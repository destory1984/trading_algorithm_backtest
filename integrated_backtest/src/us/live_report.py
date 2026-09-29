"""SPEC2 5.4 / 7: the paper account against TQQQ held from the same open, the 1-year review and the 2-year verdict.
Usage: python -m src.us.live_report   -> results/us/live/report.md, paper.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C
from src.us import vr_live_rule as VR
from src.us.live_vr import load_prices, load_state, out_dir, read


def verdict(cfg, d: pd.Series, h: pd.Series, n: int) -> list[tuple[str, str]]:
    v = cfg["live"]["verdict"]
    w = C.risk_weight(h, C.mdd(d))
    hold_deep = C.mdd(h) < v["min_hold_dd"]
    log = read("log.csv")
    dec = read("decisions.csv")
    main_b = cfg["live"]["bands"][0]
    orders = dec[(dec["band"] == main_b) & dec["side"].notna() & (dec["side"] != "")] if len(dec) else dec
    late = int(orders["late"].astype(str).eq("True").sum()) if len(orders) else 0
    bad = int((~log["self_check"].fillna("").isin(["일치", ""])).sum()) if len(log) else 0
    c1 = f"종이 {C.pct(C.cagr(d))} vs 낙폭 맞춘 보유(w {w:.2f}) {C.pct(C.cagr(w * h))}"
    c2 = f"종이 낙폭 {C.pct(C.mdd(d))}, 보유 {C.pct(C.mdd(h))}, 비율 {C.mdd(d) / C.mdd(h):.0%} (한도 {v['mdd_ratio']:.0%})" if C.mdd(h) < 0 else "보유 낙폭 없음"
    ok1 = C.cagr(d) > C.cagr(w * h)
    ok2 = C.mdd(h) < 0 and C.mdd(d) / C.mdd(h) <= v["mdd_ratio"]
    tag = lambda ok: ("O" if ok else "X") if hold_deep else "미룸(보유 낙폭이 얕음)"
    return [("1 공짜 아님", f"{tag(ok1)} {c1}"), ("2 낙폭", f"{tag(ok2)} {c2}"),
            ("3 실행", f"{'O' if late == 0 else 'X'} 늦은 알림 {late}건"), ("4 일치", f"{'O' if bad == 0 else 'X'} 불일치 실행 {bad}번")]


def main() -> None:
    cfg = C.load_config()
    st = load_state()
    L = ["# 밸류 리밸런싱 TQQQ 기록 장치", "", "지시서 `SPEC2.md`. 주문은 내지 않는다.", ""]
    if "paper_start" not in st:
        L.append("아직 시작하지 않았다.")
    else:
        px = load_prices(cfg)
        p = px[px.index >= pd.Timestamp(st["paper_start"])]
        L += [f"첫 실행 {st['start_run_kst']}(그때 마지막 봉 {st['start_last_bar']}), 종이 계좌 시작 {st['paper_start']} 시가.", ""]
        if len(p) < 2:
            L.append("종이 계좌 시작 전이거나 첫날이다.")
        else:
            cap = cfg["us"]["capital"]
            r = VR.run(cfg, p, cfg["live"]["bands"][0])
            h = VR.hold(cfg, p)
            pd.DataFrame({"paper": r["equity"], "hold": h}).to_csv(out_dir() / "paper.csv")
            d, hh = C.from_equity(r["equity"], cap), C.from_equity(h, cap)
            n = len(p) - 1
            L += [f"{n}거래일째. 종이 계좌 {(1 + d).prod() - 1:+.1%}(최대 낙폭 {C.mdd(d):.1%}), TQQQ 보유 {(1 + hh).prod() - 1:+.1%}"
                  f"(최대 낙폭 {C.mdd(hh):.1%}). 현금 풀 ${r['state']['pool']:,.0f}, 목표 V ${r['state']['V']:,.0f}.", ""]
            if n >= cfg["live"]["review_days"]:
                title = "2년 판정" if n >= cfg["live"]["verdict_days"] else "1년 점검(보고만)"
                L += [f"## {title}", "", C.md_table(pd.DataFrame(verdict(cfg, d, hh, n), columns=["기준", "결과"])), ""]
    log = read("log.csv")
    if len(log):
        L += ["## 최근 실행", "", C.md_table(log.tail(10)), ""]
    (out_dir() / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
