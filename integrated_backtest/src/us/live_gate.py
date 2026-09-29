"""SPEC2 3: the gate before the forward log. Value rebalancing with next-open execution on TQQQ over live.gate,
against the drawdown-matched TQQQ hold (w x hold + cash). The log starts only if the full-period CAGR is higher.
Halves and the same-close backtest (integrated_backtest U2) are shown for reference.
Usage: python -m src.us.live_gate   -> results/us/live_gate.md (exit code 1 if the gate is not passed)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import common as C
from src.us import data as D
from src.us import strategies as S
from src.us import vr_live_rule as VR


def compare(cfg, eq: pd.Series, hold_eq: pd.Series) -> dict:
    cap = cfg["us"]["capital"]
    d, h = C.from_equity(eq, cap), C.from_equity(hold_eq, cap)
    mid = eq.index[0] + (eq.index[-1] - eq.index[0]) / 2
    out = {}
    for s, m in {"전체": np.ones(len(d), bool), "앞 절반": np.asarray(d.index <= mid), "뒤 절반": np.asarray(d.index > mid)}.items():
        x, y = d[m], h[m]
        w = C.risk_weight(y, C.mdd(x))
        out[s] = {"기간": f"{x.index[0].date()} → {x.index[-1].date()}", "cagr": C.cagr(x), "mdd": C.mdd(x),
                  "hold_cagr": C.cagr(y), "hold_mdd": C.mdd(y), "w": w, "risk_cagr": C.cagr(w * y), "diff": C.cagr(x) - C.cagr(w * y)}
    return out


def main() -> None:
    cfg = C.load_config()
    g = cfg["live"]["gate"]
    px = D.prices(cfg, cfg["live"]["ticker"])
    px = px[(px.index >= g["start"]) & (px.index <= g["end"])]
    rows, res = [], {}
    for band in cfg["live"]["bands"]:
        r = VR.run(cfg, px, band)
        res[band] = compare(cfg, r["equity"], VR.hold(cfg, px))
        n_orders = int((r["orders"]["side"] != "").sum())
        for s, x in res[band].items():
            rows.append({"판본": f"다음 날 시가, 밴드 {band:.0%}", "구간": s, **x, "주문": n_orders if s == "전체" else ""})
    same = compare(cfg, S.value_rebalance(cfg, px, cfg["us"]["U2"]["band"])["equity"], S.hold(cfg, px))
    for s, x in same.items():
        rows.append({"판본": "같은 날 종가(백테스트 U2)", "구간": s, **x, "주문": ""})
    main_band = cfg["live"]["bands"][0]
    passed = res[main_band]["전체"]["diff"] > 0
    t = pd.DataFrame(rows)
    L = ["# 기록 장치 관문 (SPEC2 3절)", "",
         f"TQQQ {g['start']} → {g['end']}, 밸류 리밸런싱(50% 시작, 연 10% 성장, 10거래일, 밴드), 갱신일 종가로 판단하고 다음 날 시가에 체결. "
         "비교 기준: 같은 낙폭의 TQQQ 보유(w × 보유 + 현금). 보유는 첫날 시가에 전부 산다.", "",
         f"**관문: {'통과' if passed else '실패'}** (밴드 {main_band:.0%} 전체 구간 차이 {C.pp(res[main_band]['전체']['diff'], 2)})", "",
         C.md_table(pd.DataFrame([{"판본": r["판본"], "구간": r["구간"], "기간": r["기간"], "연 수익률": C.pct(r["cagr"]),
                                   "최대 낙폭": C.pct(r["mdd"]), "보유": f"{C.pct(r['hold_cagr'])} / {C.pct(r['hold_mdd'])}",
                                   "낙폭 맞춘 보유(w)": f"{C.pct(r['risk_cagr'])} ({r['w']:.2f})", "차이": C.pp(r["diff"], 2),
                                   "주문": r["주문"]} for _, r in t.iterrows()])), "",
         "관문은 전체 구간 하나로만 정했다. 앞·뒤 절반과 같은 날 종가 판본은 참고다."]
    (C.results_dir("us") / "live_gate.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
