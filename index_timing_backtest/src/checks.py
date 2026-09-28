"""SPEC 5: verification -> results/checks.md. Run after run.py.
Usage: python -m src.checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C
from src import rules as R
from src import sim as S
from src.run import one


def yes(ok: bool) -> str:
    return "통과" if ok else "실패"


def base_div(cfg, m):
    return None if m == "US" else cfg["markets"]["KR"]["dividend"]


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    start = pd.Timestamp(cfg["data"]["start"])
    L = ["# 검증", ""]
    st = {}

    # 1 seal
    daily = pd.read_parquet(out / "daily.parquet")
    lines, ok = [], True
    for m in C.MARKETS:
        last_raw = C.raw(cfg, m).index[-1]
        last_run = max(daily[[c for c in daily.columns if c.startswith(m + "|")]].dropna(how="all").index)
        ok &= last_raw <= C.end(cfg, m) and last_run <= C.end(cfg, m)
        lines.append(f"{cfg['markets'][m]['name']}: 원표 {last_raw.date()}, 곡선 {last_run.date()} (봉인 {C.end(cfg, m).date()})")
    st["1. 봉인"] = ok
    L += ["## 1. 봉인", "", ". ".join(lines) + f". **{yes(ok)}**", ""]

    # 2 always-in at zero cost == buy and hold
    ok, lines = True, []
    for m in C.MARKETS:
        px = C.prices(cfg, m, base_div(cfg, m) if m == "KR" else "config")
        d, _, _ = S.simulate(px, R.decisions(px["close"], "HOLD", None, cfg), start, 0.0)
        g = px[px.index >= start]
        h = g["close"].pct_change()
        h.iloc[0] = g["close"].iloc[0] / g["open"].iloc[0] - 1
        diff = float((d - h).abs().max())
        ok &= diff < 1e-12
        lines.append(f"{cfg['markets'][m]['name']} 최대 차이 {diff:.1e}")
    st["2. 항상 보유 = 보유"] = ok
    L += ["## 2. 비중 100% 고정, 비용 0 = 보유", "", ", ".join(lines) + f". **{yes(ok)}**", ""]

    # 3 hand check
    rows, ok = [], True
    for m in C.MARKETS:
        px = C.prices(cfg, m, base_div(cfg, m) if m == "KR" else "config")
        c = px["close"]
        rate = cfg["markets"][m]["cost"]
        for r, rc in cfg["rules"].items():
            p = rc["main"]
            d, w, trades = S.simulate(px, R.decisions(c, r, p, cfg), start, rate)
            for day in trades[1:1 + int(cfg["checks"]["hand_changes"])]:
                i = px.index.get_loc(day)
                sd = px.index[i - 1]
                if r == "T1":
                    ind = c.iloc[:i].rolling(int(p)).mean().iloc[-1]
                    want = float(c[sd] > ind)
                    txt = f"종가 {c[sd]:.2f}, {p}일선 {ind:.2f}"
                elif r == "T2":
                    me = R.month_end(px.index)
                    mc = c[me & (px.index <= sd)]
                    ind = mc.iloc[-int(p):].mean()
                    want = float(mc.iloc[-1] > ind)
                    txt = f"월말 종가 {mc.iloc[-1]:.2f}, {p}개월 평균 {ind:.2f}"
                else:
                    ret = c.iloc[:i].pct_change().iloc[-int(rc['vol_window']):]
                    vol = ret.std(ddof=1) * np.sqrt(252)
                    want = min(1.0, float(p) / vol)
                    txt = f"20일 변동성 {vol:.1%}"
                prev_w = w.iloc[w.index.get_loc(day) - 1]
                match = abs(w[day] - want) < 1e-9
                ok &= match
                rows.append({"시장": m, "규칙": r, "신호일": sd.date(), "지표": txt, "새 비중": f"{w[day]:.3f}", "손계산 비중": f"{want:.3f}",
                             "이전 비중": f"{prev_w:.3f}", "체결일": day.date(), "체결가(시가)": f"{px.loc[day, 'open']:.2f}",
                             "비용률": f"{rate:.2%}", "일치": "예" if match else "아니오"})
    st["3. 손 검산"] = ok
    L += ["## 3. 손 검산 (규칙 × 시장마다 처음 시작 뒤 비중이 바뀐 날 3번)", "", C.md_table(pd.DataFrame(rows)), "", f"**{yes(ok)}**", ""]

    # 4 T2 only after month-ends, T3 only after week-ends
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    ok, lines = True, []
    for m in C.MARKETS:
        idx = C.prices(cfg, m).index
        for r, fn in (("T2", R.month_end), ("T3", R.week_end)):
            flag = pd.Series(fn(idx), index=idx)
            ks = [k for k in tr["run"].unique() if k.startswith(f"{m}|{r}|")]
            bad = 0
            for k in ks:
                for day in tr.loc[tr["run"] == k, "date"].iloc[1:]:
                    bad += not bool(flag.iloc[idx.get_loc(day) - 1])
            ok &= bad == 0
            lines.append(f"{m} {r} 어긋난 거래 {bad}건")
    st["4. 월말·주말에만"] = ok
    L += ["## 4. T2 는 월말 다음 날, T3 는 주말 다음 날에만 바뀐다", "", ", ".join(lines) + f" (각 곡선의 첫 매수 제외). **{yes(ok)}**", ""]

    # 5 look-ahead
    rng = np.random.default_rng(int(cfg["checks"]["seed"]))
    ok, lines = True, []
    n = int(cfg["checks"]["perturb_days"])
    for m in C.MARKETS:
        raw = C.raw(cfg, m)
        bad = raw.copy()
        last = raw.index[-n:]
        k = pd.Series(rng.uniform(0.9, 1.1, n), index=last)
        for col in [c for c in ("open", "high", "low", "close", "adj_close") if c in raw]:
            bad.loc[last, col] = bad.loc[last, col] * k
        for r, rc in cfg["rules"].items():
            _, w1, _ = one(cfg, m, r, rc["main"], base_div(cfg, m))
            _, w2, _ = one(cfg, m, r, rc["main"], base_div(cfg, m), table=bad)
            before = w1.index < last[0]
            same = bool((w1[before] == w2[before]).all())
            ok &= same
            lines.append(f"{m} {r} {'같음' if same else '다름'}")
    st["5. 미래 데이터 금지"] = ok
    L += ["## 5. 미래 데이터 금지", "", f"시장마다 마지막 {n}거래일 가격에 0.9 → 1.1 사이 무작위 배수를 곱하고 다시 돌렸다. 그 전의 비중: " + ", ".join(lines) + f". **{yes(ok)}**", ""]

    L += ["## 요약", "", C.md_table(pd.DataFrame([{"검사": a, "결과": yes(b)} for a, b in st.items()])), ""]
    (out / "checks.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"{a}: {yes(b)}" for a, b in st.items()))


if __name__ == "__main__":
    main()
