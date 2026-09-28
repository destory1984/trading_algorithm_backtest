"""SPEC2 4.2: the paper account — what following the logged rule since the start would have done — against SPY hold.

The paper account runs src.sim.simulate on the live prices with the backtest's decisions (src.rules.t3) from
state.paper_start (the first NYSE open after the first run): the target decided before it is bought at that open, as
in the backtest. SPY hold buys at the same open without costs. Milestones: live.review_days (report only) and
live.verdict_days (SPEC2 6). Run after src.live.
Usage: python -m src.paper   -> results/live/paper.csv, report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C
from src import rules as R
from src import sim as S
from src.live import load_prices, load_state, out_dir, read


def curves(cfg: dict, px: pd.DataFrame) -> pd.DataFrame | None:
    st = load_state()
    if "start_bar" not in st:
        return None
    after = px.index[px.index >= pd.Timestamp(st["paper_start"])]
    if not len(after):
        return None
    rc, lv = cfg["rules"]["T3"], cfg["live"]
    out = {}
    for t in lv["targets"]:
        dec = R.t3(px["close"], float(t), int(rc["vol_window"]), float(rc["band"]))
        d, w, _ = S.simulate(px, dec, after[0], cfg["markets"]["US"]["cost"])
        out[f"rule_{t:g}"], out[f"weight_{t:g}"] = d, w
    hold, _, _ = S.simulate(px, R.decisions(px["close"], "HOLD", None, cfg), after[0], 0.0)
    out["hold"] = hold
    return pd.DataFrame(out)


def stats(d: pd.Series) -> dict:
    days = len(d)
    total = float((1 + d).prod() - 1)
    ann = (1 + total) ** (252 / days) - 1 if days else np.nan
    return {"days": days, "total": total, "ann": ann, "mdd": C.mdd(d)}


def summary(cfg: dict, px: pd.DataFrame) -> str:
    c = curves(cfg, px)
    if c is None or not len(c):
        return "[기록 장치] 종이 계좌는 아직 시작 전이다."
    t = cfg["live"]["targets"][0]
    r, h = stats(c[f"rule_{t:g}"]), stats(c["hold"])
    return (f"[기록 장치 월간] {c.index[0].date()} → {c.index[-1].date()} ({r['days']}거래일). 규칙 {r['total']:+.1%} "
            f"(낙폭 {r['mdd']:.1%}), SPY 보유 {h['total']:+.1%} (낙폭 {h['mdd']:.1%}). 지금 목표 비중 {c[f'weight_{t:g}'].iloc[-1]:.2f}.")


def verdict(cfg: dict, c: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    v = cfg["live"]["verdict"]
    t = cfg["live"]["targets"][0]
    x = c.iloc[:n]
    r, h = stats(x[f"rule_{t:g}"]), stats(x["hold"])
    alerts, dec = read("alerts.csv"), read("decisions.csv")
    late = int(dec[(dec["target_vol"] == t) & dec["changed"] & dec["late"]].shape[0]) if len(dec) else 0
    log = read("log.csv")
    mism = int(log["self_check"].astype(str).str.startswith("불일치").sum()) if len(log) else 0
    c1 = r["ann"] >= h["ann"] - v["cagr_gap"]
    if h["mdd"] <= v["min_hold_dd"]:
        c2 = "만족" if abs(r["mdd"]) <= v["mdd_ratio"] * abs(h["mdd"]) else "불만족"
    else:
        c2 = "미룸(보유 낙폭이 얕음)"
    return [("1. 연 수익 차이 ≥ -3%p", f"{'만족' if c1 else '불만족'} ({r['ann'] - h['ann']:+.1%}p)"),
            ("2. 낙폭 ≤ 보유의 70%", f"{c2} (규칙 {r['mdd']:.1%}, 보유 {h['mdd']:.1%})"),
            ("3. 늦은 알림 0건", f"{'만족' if late == 0 else '불만족'} ({late}건)"),
            ("4. 자가 점검 불일치 0건", f"{'만족' if mism == 0 else '확인 필요'} ({mism}번)")]


def main() -> None:
    cfg = C.load_config()
    px = load_prices(cfg)
    c = curves(cfg, px)
    st = load_state()
    L = ["# 변동성 목표 기록 장치", "", f"첫 실행 {st.get('start_run_kst', '')} (그때 마지막 봉 {st.get('start_bar', '')}), 종이 계좌 시작 {st.get('paper_start', '')} 시가. 지시서는 `SPEC2.md`.", ""]
    if c is None or not len(c):
        L += [f"종이 계좌는 아직 시작 전이다({st.get('paper_start', '')} 시가에 시작한다).", ""]
    else:
        c.to_csv(out_dir() / "paper.csv")
        rows = [{"대상": f"규칙 {t:g}" + (" (판정)" if i == 0 else ""), **{k: v for k, v in stats(c[f"rule_{t:g}"]).items()},
                 "지금 비중": c[f"weight_{t:g}"].iloc[-1]} for i, t in enumerate(cfg["live"]["targets"])]
        rows.append({"대상": "SPY 보유", **stats(c["hold"]), "지금 비중": 1.0})
        t = pd.DataFrame([{"대상": r["대상"], "거래일": r["days"], "누적": C.pct(r["total"]), "연 환산": C.pct(r["ann"]),
                           "최대 낙폭": C.pct(r["mdd"]), "지금 비중": f"{r['지금 비중']:.2f}"} for r in rows])
        L += [f"## 종이 계좌 ({c.index[0].date()} → {c.index[-1].date()})", "", C.md_table(t), ""]
        for name, n in (("1년 중간 점검(보고만)", cfg["live"]["review_days"]), ("2년 판정", cfg["live"]["verdict_days"])):
            if len(c) >= n:
                L += [f"## {name}", "", C.md_table(pd.DataFrame(verdict(cfg, c, n), columns=["기준", "결과"])), ""]
            else:
                L += [f"## {name}", "", f"{n}거래일 중 {len(c)}거래일 지났다.", ""]
    log = read("log.csv")
    if len(log):
        L += ["## 최근 실행", "", C.md_table(log.tail(10)), ""]
    (out_dir() / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
