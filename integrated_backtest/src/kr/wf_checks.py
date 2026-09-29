"""SPEC3 4: checks of the walk-forward study. Run after src.kr.wf.
  1 seal; 2 no trade sold after the training end was used for a pick (leak = 0 for every year and mode);
  3 one year's pick recomputed from the raw grid-trade file with a separately written cost formula;
  4 three positions held across a year end match the candidate row of the cell picked in their entry year;
  5 three walk-forward trades recomputed from raw prices (breakout, value multiple, entry, exit, costs).
Usage: python -m src.kr.wf_checks   -> results/kr/wf/checks.md (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import common as C
from src.checks import raw_valid
from src.kr import build as B
from src.kr.kernel import BRK, HOLD, LINE, REASONS, STOP
from src.kr.portfolio import Panel

OUT = "kr/wf"


def main() -> None:
    cfg = C.load_config()
    d = C.results_dir(OUT)
    P = Panel(in_ram=False)
    cal = P.cal
    eq = pd.read_parquet(d / "equity.parquet")
    pk = pd.read_csv(d / "picks.csv")
    tr = pd.read_parquet(d / "trades.parquet")
    res = []
    res.append(("1 봉인", eq.index.max() <= pd.Timestamp(cfg["data"]["end"]) and cal[int(tr["exit_i"].max())] <= pd.Timestamp(cfg["data"]["end"]),
                f"곡선 마지막 날 {eq.index.max().date()}"))
    res.append(("2 미래 정보 없음", int(pk["leak"].sum()) == 0, f"고를 때 쓴 거래 중 학습 끝 뒤에 판 것 {int(pk['leak'].sum())}건(해 {pk['year'].nunique()}개 × 방식 {pk['mode'].nunique()}개)"))
    # 3 pick reproduction for 2019
    y = 2019
    g = pd.read_parquet(C.results_dir("kr") / "grid_trades.parquet")
    g = g[(g["strategy"].astype(str) == "B") & (g["reason"] != BRK)]
    cs = cfg["costs"]
    end_day = pd.Timestamp(f"{y - 1}-12-31")
    xd = cal[g["exit_i"].to_numpy()]
    x = g[(xd <= end_day) & (cal[g["entry_i"].to_numpy()] >= pd.Timestamp(cfg["wf"]["train_start"]))].copy()
    taxes = np.array([C.sell_tax(cfg, dd) for dd in cal[x["exit_i"].to_numpy()]])
    x["r"] = x["exit_px"] * (1 - cs["slippage"]) * (1 - cs["sell_fee"] - taxes) / (x["entry_px"] * (1 + cs["slippage"]) * (1 + cs["buy_fee"])) - 1
    st = x.groupby("combo")["r"].agg(["mean", "size"])
    st = st[st["size"] >= cfg["wf"]["min_trades"]].sort_index()
    top = int(st["mean"].idxmax())
    want = int(pk[(pk["year"] == y) & (pk["mode"] == "main")]["combo"].iloc[0])
    res.append(("3 고른 칸 재현", top == want, f"{y} 년: 손으로 다시 잰 1등 칸 {top}, 기록 {want} (거래당 {st.loc[top, 'mean']:+.3%})"))
    # 4 seams
    m = tr[tr["run"] == "main|main"].copy()
    m["ey"], m["xy"] = cal.year[m["entry_i"].to_numpy()], cal.year[m["exit_i"].to_numpy()]
    across = m[m["ey"] != m["xy"]].head(3)
    from src.kr.wf import build_cands  # noqa: F401  (candidates are rebuilt only for the three cells involved)
    cells = sorted({int(pk[(pk["year"] == int(r.ey)) & (pk["mode"] == "main")]["combo"].iloc[0]) for r in across.itertuples()})
    cand, _ = build_cands(cfg, cells)
    seam_ok, seam_txt = True, []
    for r in across.itertuples():
        cell = int(pk[(pk["year"] == int(r.ey)) & (pk["mode"] == "main")]["combo"].iloc[0])
        c = cand[(cand["combo"] == cell) & (cand["tid"] == r.tid) & (cand["entry_i"] == r.entry_i)]
        ok = len(c) == 1 and int(c["exit_i"].iloc[0]) == int(r.exit_i) and abs(float(c["exit_px"].iloc[0]) - r.exit_px) < 1e-6
        seam_ok &= ok
        seam_txt.append(f"{P.ticker[int(r.tid)]} {cal[int(r.entry_i)].date()} → {cal[int(r.exit_i)].date()} 칸 {cell} {'일치' if ok else '불일치'}")
    res.append(("4 이음새", seam_ok and len(across) > 0, "; ".join(seam_txt)))
    # 5 hand check
    lines, hand_ok = [], True
    grid = B.b_grid(cfg)
    for want_reason in (LINE, STOP, HOLD):
        cand_rows = m[m["reason"] == want_reason]
        if not len(cand_rows):
            continue
        r = cand_rows.iloc[len(cand_rows) // 2]
        cell = int(pk[(pk["year"] == int(r["ey"])) & (pk["mode"] == "main")]["combo"].iloc[0])
        cb = grid[cell]
        tk = P.ticker[int(r["tid"])]
        v = raw_valid(cfg, tk)
        ie, ix = v.index.get_loc(cal[int(r["entry_i"])]), v.index.get_loc(cal[int(r["exit_i"])])
        c = v["close"]
        hh = float(v["high"].iloc[ie - 1 - cb["n"]: ie - 1].max())
        val = c * v["volume"]
        mult = float(val.iloc[ie - 1] / val.iloc[ie - 21: ie - 1].mean())
        chk = {"신호": c.iloc[ie - 1] > hh and mult >= cb["mult"], "진입가": abs(v["open"].iloc[ie] - r["entry_px"]) < 1e-6}
        if cb["stop"] == "atr2":
            pc = c.shift(1)
            trr = pd.concat([v["high"] - v["low"], (v["high"] - pc).abs(), (pc - v["low"]).abs()], axis=1).max(axis=1)
            stop = v["open"].iloc[ie] - 2 * float(trr.rolling(20).mean().iloc[ie - 1])
        else:
            stop = v["open"].iloc[ie] * (1 + cb["stop"])
        line = {"ma20": c < c.rolling(20).mean(), "ll10": c < v["low"].shift(1).rolling(10).min(),
                "ll20": c < v["low"].shift(1).rolling(20).min()}[cb["exit"]]
        if r["reason"] == LINE:
            chk["청산"] = bool(line.iloc[ix - 1]) and abs(v["open"].iloc[ix] - r["exit_px"]) < 1e-6
        elif r["reason"] == STOP:
            chk["청산"] = v["low"].iloc[ix] <= stop + 1e-9 and abs(min(v["open"].iloc[ix], stop) - r["exit_px"]) < 1e-6
        else:
            chk["청산"] = abs(c.iloc[ix] - r["exit_px"]) < 1e-6
        tax = C.sell_tax(cfg, cal[int(r["exit_i"])])
        cost = r["shares"] * r["entry_px"] * (1 + cs["slippage"]) * (1 + cs["buy_fee"])
        proc = r["shares"] * r["exit_px"] * (1 - cs["slippage"]) * (1 - cs["sell_fee"] - tax)
        chk["비용"] = abs(cost - r["cost"]) < 1e-6 * cost and abs(proc - r["proceeds"]) < 1e-6 * proc
        hand_ok &= all(chk.values())
        lines.append(f"- {tk} {cal[int(r['entry_i'])].date()} → {cal[int(r['exit_i'])].date()} ({REASONS[int(r['reason'])]}, 칸 {cell}: "
                     f"{cb['n']}일 신고가, 배수 {cb['mult']}, 손절 {cb['stop']}, 이탈 {cb['exit']}): 종가 {c.iloc[ie - 1]:.0f} > {hh:.0f}, "
                     f"배수 {mult:.2f}, 산 값 {r['entry_px']:.0f}, 손절선 {stop:.0f}, 판 값 {r['exit_px']:.0f}; "
                     + ", ".join(f"{k} {'일치' if b else '불일치'}" for k, b in chk.items()))
    res.append(("5 손 검산", hand_ok, "아래 (필터 RS·템플릿은 SPEC 검증 3 에서 같은 코드로 확인했고 여기서는 다시 재지 않는다)"))
    L = ["# 걸어가며 고르기 검증 (SPEC3 4절)", "",
         C.md_table(pd.DataFrame([{"검증": n, "결과": "통과" if o else "실패", "내용": t} for n, o, t in res])), "", "## 5 손 검산", "", *lines]
    (d / "checks.md").write_text("\n".join(L), encoding="utf-8")
    for n, o, t in res:
        print(("PASS " if o else "FAIL ") + n + ": " + t)
    for x in lines:
        print(x)
    sys.exit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
