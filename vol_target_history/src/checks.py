"""SPEC 5: the five checks. Run after run.py and data_check.py.
Usage: python -m src.checks   -> results/checks.md (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import common as C
from src import rules as R
from src.run import assumptions, key, one, runs

# Days the close moved more than 10%, with the known event (listed after data_check.py printed them; check 3 asks
# that every such day be a known one)
KNOWN = {
    "1929-10-28": "1929 폭락 검은 월요일",
    "1929-10-29": "1929 폭락 검은 화요일",
    "1929-10-30": "폭락 뒤 첫 반등",
    "1931-06-22": "후버 모라토리엄(전쟁 부채 상환 유예) 발표 뒤 첫 거래일",
    "1931-10-06": "후버의 은행 신용 공동기금 계획 발표",
    "1932-09-21": "1932 여름 반등 중의 급등",
    "1933-03-15": "은행 휴업(1933-03-06 → 14) 뒤 재개장 첫날",
    "1939-09-05": "제2차 세계대전 개전 뒤 첫 거래일(군수 기대)",
    "1987-10-19": "1987 폭락 검은 월요일",
}


def seal_check(cfg) -> tuple[bool, str]:
    out = C.results_dir()
    full = pd.read_parquet(C.hist_dir(cfg) / cfg["data"]["file"])
    daily = pd.read_parquet(out / "daily.parquet")
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    cut = pd.Timestamp("2007-01-01")
    n = int((full.index >= cut).sum() + (daily.index >= cut).sum() + (tr["date"] >= cut).sum())
    return n == 0, (f"2007-01-01 이후 날짜: 받은 파일 {int((full.index >= cut).sum())}개, 곡선·거래 표 합 {n}개. "
                    f"받은 파일 마지막 날 {full.index.max().date()}")


def replicate_check(cfg) -> tuple[bool, str]:
    out = C.results_dir()
    spy = pd.read_parquet(out / "spy_daily.parquet")
    n = pd.read_csv(out / "spy_trades.csv", index_col=0)["trades"]
    it = pd.read_csv(C.ROOT / cfg["replicate"]["index_timing_metrics"])
    it = it[(it["market"] == "US") & (it["stretch"] == "full")]
    lines, ok = [], True
    for r, t in runs(cfg):
        k = f"SPY|{'' if t is None else t}|open" if r == "T3" else "SPY|HOLD|open"
        ref = it[(it["rule"] == r) & ((it["param"].astype(str) == str(t)) if t is not None else True)].iloc[0]
        mine = (C.cagr(spy[k]), C.mdd(spy[k]), int(n[k]))
        same = abs(mine[0] - ref["cagr"]) < 5e-5 and abs(mine[1] - ref["mdd"]) < 5e-5 and mine[2] == int(ref["trades"])
        ok &= same
        lines.append(f"{'보유' if r == 'HOLD' else f'T3 {t:.0%}'} 연 {mine[0]:.4f}/{ref['cagr']:.4f}, "
                     f"낙폭 {mine[1]:.4f}/{ref['mdd']:.4f}, 거래 {mine[2]}/{int(ref['trades'])}")
    return ok, "이 폴더 / index_timing: " + "; ".join(lines)


def data_check(cfg) -> tuple[bool, str]:
    big = pd.read_csv(C.results_dir() / "big_moves.csv")
    unknown = [d for d in big["date"] if d not in KNOWN]
    return not unknown, (f"하루 ±{cfg['checks']['big_move']:.0%} 넘게 움직인 날 {len(big)}개, 알려진 날과 대조해 "
                         f"모르는 날 {len(unknown)}개" + (f": {unknown}" if unknown else ""))


def hand_check(cfg) -> tuple[bool, list[str]]:
    """T3 15% (main assumption, cash 0), recomputed by hand from the raw closes:
    - the first `hand_changes` target changes: 20-day volatility with plain numpy, week-end from the next trading
      date, the band test;
    - every day's return, from the weight traced day by day: drift between trades, and on a trade day equity before
      the trade V = 1 + a x g, fee = |target x V - a x (1 + g)| x cost, and the weight right after the trade
      target x V / (V - fee) (the fee comes out of cash, as in index_timing's sim)."""
    raw = C.raw(cfg)
    a = assumptions(cfg)["본"]
    assert a["cash"] == 0.0
    close = C.total_return(raw["close"], a["dividend"])
    t, rc = rc_main(cfg)
    k = key("T3", t, "본")
    d = pd.read_parquet(C.results_dir() / "daily.parquet")[k]
    w = pd.read_parquet(C.results_dir() / "weights.parquet")[k]
    idx = close.index
    c = close.to_numpy()
    s0 = idx.searchsorted(pd.Timestamp(cfg["data"]["start"]))
    changed = w.diff().fillna(w.iloc[0]).ne(0)
    exp = np.empty(len(w))
    a_w, fees = 0.0, {}
    for j, day in enumerate(w.index):
        i = s0 + j
        g = c[i] / c[i - 1] - 1 if j > 0 else 0.0
        if changed.iloc[j]:
            tgt = float(w.iloc[j])
            v = 1 + a_w * g if j > 0 else 1.0
            held = a_w * (1 + g) if j > 0 else 0.0
            fee = abs(tgt * v - held) * cfg["cost"]
            exp[j] = v - fee - 1  # the first day is not compared (it starts from cash at the close)
            a_w = tgt * v / (v - fee)
            fees[day] = (fee, v, held)
        else:
            exp[j] = a_w * g
            a_w = a_w * (1 + g) / (1 + a_w * g)
    diff = float(np.abs(exp[1:] - d.to_numpy()[1:]).max())
    lines, ok = [f"- 모든 날({len(w) - 1}일)의 수익을 손 계산과 비교한 최대 차이 {diff:.1e}"], diff < 1e-9
    for ex in w.index[changed.to_numpy()][1: cfg["checks"]["hand_changes"] + 1]:
        i = idx.get_loc(ex)
        dec_day, nxt = idx[i - 1], idx[i]
        wk_end = dec_day.isocalendar()[:2] != nxt.isocalendar()[:2]
        rets = c[i - 20: i] / c[i - 21: i - 1] - 1  # the 20 returns ending on dec_day
        vol = float(np.std(rets, ddof=1) * np.sqrt(252))
        want = min(1.0, rc["main"] / vol)
        prev_t = float(w.iloc[w.index.get_loc(ex) - 1])
        band_ok = abs(want - prev_t) > rc["band"]
        same_w = abs(float(w.loc[ex]) - want) < 1e-12
        fee, v, held = fees[ex]
        ok &= wk_end and band_ok and same_w
        lines.append(f"- 결정 {dec_day.date()}(다음 거래일 {nxt.date()}, 주 끝 {wk_end}): 20일 변동성 {vol:.1%} → "
                     f"min(1, {rc['main']:.0%} / {vol:.1%}) = {want:.3f}, 이전 목표 {prev_t:.3f}, "
                     f"문턱 {rc['band']:.0%}p 넘음 {band_ok}. 기록된 목표 {float(w.loc[ex]):.3f} ({'일치' if same_w else '불일치'}). "
                     f"실행 {ex.date()} 종가(배당 근사 {c[i]:.4f}), 거래 전 지수 몫 {held:.4f} / 자산 {v:.4f}, "
                     f"비용 {fee:.6f} = |{want:.3f} × {v:.4f} - {held:.4f}| × {cfg['cost']:.2%}, "
                     f"그날 수익 {float(d.loc[ex]):+.6f}")
    return ok, lines


def rc_main(cfg):
    return cfg["rule"]["main"], cfg["rule"]


def lookahead_check(cfg) -> tuple[bool, str]:
    raw = C.raw(cfg)
    n = cfg["checks"]["perturb_days"]
    rng = np.random.default_rng(cfg["checks"]["seed"])
    pert = raw.copy()
    pert.iloc[-n:, pert.columns.get_loc("close")] *= rng.uniform(0.7, 1.3, size=n)
    cut = raw.index[-n]
    bad, off_days, count = 0, 0, 0
    we = R.week_end(raw.index)
    start = pd.Timestamp(cfg["data"]["start"])
    for name, a in assumptions(cfg).items():
        for r, t in runs(cfg):
            _, w0, tr = one(cfg, r, t, a)
            _, w1, _ = one(cfg, r, t, a, pert)
            bad += int((w0[w0.index < cut] != w1[w1.index < cut]).sum())
            pos = raw.index.get_indexer(tr)
            off_days += int(sum(1 for p, x in zip(pos, tr) if x != raw.index[raw.index.searchsorted(start)] and not we[p - 1]))
            count += 1
    return bad == 0 and off_days == 0, (f"마지막 {n}거래일({cut.date()} →) 종가를 0.7 → 1.3 배로 흔든 뒤 그 전 비중이 "
                                        f"바뀐 날 {bad}개 ({count}개 실행). 주 끝 다음 날이 아닌 거래 {off_days}건")


def main() -> None:
    cfg = C.load_config()
    ok1, t1 = seal_check(cfg)
    ok2, t2 = replicate_check(cfg)
    ok3, t3 = data_check(cfg)
    ok4, t4 = hand_check(cfg)
    ok5, t5 = lookahead_check(cfg)
    res = [("1 봉인", ok1, t1), ("2 재현(SPY 2008 → 2026)", ok2, t2), ("3 데이터 큰 움직임", ok3, t3),
           ("4 손 검산", ok4, "아래"), ("5 미래 데이터 금지, 주 끝에만 바뀜", ok5, t5)]
    L = ["# 검증 (SPEC 5)", "", C.md_table(pd.DataFrame(
        [{"검증": n, "결과": "통과" if o else "실패", "내용": t} for n, o, t in res])), "",
        "## 3 큰 움직임과 알려진 사건", "", *[f"- {d}: {e}" for d, e in KNOWN.items()], "",
        "## 4 손 검산 (T3 15%, 본 가정)", "", *t4]
    (C.results_dir() / "checks.md").write_text("\n".join(L), encoding="utf-8")
    for n, o, t in res:
        print(("PASS " if o else "FAIL ") + n + ": " + t)
    for x in t4:
        print(x)
    sys.exit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
