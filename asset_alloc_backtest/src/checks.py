"""SPEC 6: the five checks. Run after run.py.
Usage: python -m src.checks   -> results/checks.md (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import common as C
from src.rules import fixed
from src.run import targets
from src.sim import simulate


def seal_check(cfg, out) -> tuple[bool, str]:
    last = pd.Timestamp(cfg["data"]["end"])
    daily = pd.read_parquet(out / "daily.parquet")
    tg = pd.read_csv(out / "targets.csv", parse_dates=["date"])
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    days = [daily.index.max(), tg["date"].max(), tr["date"].max()]
    ok = all(d <= last for d in days)
    return ok, f"마지막 날: 곡선 {days[0].date()}, 목표 비중 {days[1].date()}, 거래 {days[2].date()} (봉인선 {last.date()})"


def hold_check(cfg, opn, cls) -> tuple[bool, str]:
    start = pd.Timestamp(cfg["data"]["start"])
    me = cls.loc[C.month_ends(cls.index)]
    d, _ = simulate(opn, cls, fixed(me, {"SPY": 1.0}), start, 0.0)
    s = cls.index.searchsorted(start)
    hold = cls["SPY"].iloc[s:] / opn["SPY"].iloc[s]
    gap = float(((1 + d).cumprod() / hold - 1).abs().max())
    return gap < 1e-12, f"SPY 100% 월간 계좌와 SPY 보유 곡선의 최대 차이 {gap:.1e}"


def month_end_closes(r: dict) -> pd.DataFrame:
    """Recomputed from the raw files a second way: the last row of each calendar month, minus the data's last
    (incomplete) month."""
    cl = pd.DataFrame({t: df["adj_close"] for t, df in r.items()})
    last = cl.groupby([cl.index.year, cl.index.month]).tail(1)
    return last.iloc[:-1]


def hand_check(cfg, opn, cls) -> tuple[bool, list[str]]:
    """For each main rule, the first `hand_changes` months (from the start) whose target changed: recompute the
    signal from the raw month-end closes, then look up the execution day, its open and the cost paid."""
    r = C.raw(cfg)
    me = month_end_closes(r)
    tick = cfg["data"]["tickers"]
    tg = pd.read_csv(C.results_dir() / "targets.csv", parse_dates=["date"])
    tr = pd.read_csv(C.results_dir() / "trades.csv", parse_dates=["date"])
    start = pd.Timestamp(cfg["data"]["start"])
    lines, ok = [], True
    for rule, spec in cfg["rules"].items():
        n = spec["main"]
        k = C.key(rule, n, cfg["cash"], True)
        t = tg[tg["run"] == k].set_index("date")[tick]
        chg = t.index[(t.diff().abs().sum(axis=1) > 1e-9) & (t.index >= start)][: cfg["checks"]["hand_changes"]]
        lines.append(f"### {rule} {spec['name']} ({n})")
        lines.append("")
        for day in chg:
            i = me.index.get_loc(day)
            row = me.iloc[i]
            if rule == "A2":
                ma = me.iloc[i - n + 1: i + 1].mean()
                want = {x: (0.2 if row[x] > ma[x] else 0.0) for x in tick}
                why = ", ".join(f"{x} {row[x]:.2f} {'>' if row[x] > ma[x] else '≤'} {ma[x]:.2f}" for x in tick)
            else:
                ret = row / me.iloc[i - n] - 1
                if rule == "A1":
                    pick = ("SPY" if ret["SPY"] >= ret["EFA"] else "EFA") if ret["SPY"] > 0 else "TLT"
                    want = {x: (1.0 if x == pick else 0.0) for x in tick}
                else:
                    best = ret.sort_values(ascending=False).index[: cfg["rules"]["A3"]["top"]]
                    want = {x: (0.5 if x in best and ret[x] > 0 else 0.0) for x in tick}
                why = ", ".join(f"{x} {ret[x]:+.1%}" for x in tick) + f" ({me.index[i - n].date()} 대비)"
            got = t.loc[day].to_dict()
            same = all(abs(got[x] - want[x]) < 1e-12 for x in tick)
            ex = cls.index[cls.index.searchsorted(day) + 1]
            trow = tr[(tr["run"] == k) & (tr["date"] == ex)]
            cost_ok = len(trow) == 1 and abs(trow["cost"].iloc[0] - trow["turnover"].iloc[0] * cfg["cost"]) < 1e-12
            ok &= same and cost_ok
            wtxt = ", ".join(f"{x} {v:.0%}" for x, v in got.items() if v > 0) or "현금"
            opens = ", ".join(f"{x} {opn.loc[ex, x]:.2f}" for x in tick if got[x] > 0 or t.shift(1).loc[day, x] > 0)
            lines.append(f"- 결정 {day.date()}: {why} → 목표 {wtxt} ({'일치' if same else '불일치'}). "
                         f"실행 {ex.date()} 시가({opens}), 회전율 {trow['turnover'].iloc[0]:.1%}, "
                         f"비용 {trow['cost'].iloc[0]:.4%} = 회전율 × {cfg['cost']:.2%} ({'일치' if cost_ok else '불일치'})")
        lines.append("")
    return ok, lines


def timing_check(cfg, cls) -> tuple[bool, str]:
    tg = pd.read_csv(C.results_dir() / "targets.csv", parse_dates=["date"])
    tr = pd.read_csv(C.results_dir() / "trades.csv", parse_dates=["date"])
    tick = cfg["data"]["tickers"]
    start = pd.Timestamp(cfg["data"]["start"])
    me = set(C.month_ends(cls.index))
    prev = {cls.index[i]: cls.index[i - 1] for i in range(1, len(cls.index))}
    bad_days = [d for d in tr["date"] if d != start and prev[d] not in me]
    w = tg[tick]
    ok_w = bool((w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-12).all())
    ok_me = bool(tg["date"].isin(me).all())
    return (not bad_days and ok_w and ok_me), (
        f"거래 {len(tr)}건 중 첫날·월초 아닌 날 {len(bad_days)}건, 결정이 모두 월말 {ok_me}, "
        f"비중 음수 없음·합 ≤ 1 {ok_w}")


def lookahead_check(cfg) -> tuple[bool, str]:
    r = C.raw(cfg)
    opn, cls = C.prices(cfg, r)
    base = targets(cfg, opn, cls)
    rng = np.random.default_rng(cfg["checks"]["seed"])
    n = cfg["checks"]["perturb_days"]
    cut = cls.index[-n]
    pert = {}
    for t, df in r.items():
        df = df.copy()
        f = rng.uniform(0.7, 1.3, size=n)
        for col in ("open", "high", "low", "close", "adj_close"):
            df.iloc[-n:, df.columns.get_loc(col)] *= f
        pert[t] = df
    o2, c2 = C.prices(cfg, pert)
    new = targets(cfg, o2, c2)
    bad = 0
    for k in base:
        a, b = base[k], new[k]
        a, b = a[a.index < cut], b[b.index < cut]
        bad += int(((a - b).abs().fillna(0) > 0).any(axis=1).sum() + (a.isna() != b.isna()).any(axis=1).sum())
    return bad == 0, f"마지막 {n}거래일({cut.date()} →) 가격을 0.7 → 1.3 배로 흔든 뒤 그 전 결정이 바뀐 달 {bad}개 (12개 실행)"


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    opn, cls = C.prices(cfg)
    res = []
    ok1, t1 = seal_check(cfg, out)
    ok2, t2 = hold_check(cfg, opn, cls)
    ok3, t3 = hand_check(cfg, opn, cls)
    ok4, t4 = timing_check(cfg, cls)
    ok5, t5 = lookahead_check(cfg)
    res = [("1 봉인", ok1, t1), ("2 SPY 100% = 보유", ok2, t2), ("3 손 검산", ok3, "아래"),
           ("4 월초에만 바뀜, 비중 범위", ok4, t4), ("5 미래 데이터 금지", ok5, t5)]
    L = ["# 검증 (SPEC 6)", "", C.md_table(pd.DataFrame(
        [{"검증": n, "결과": "통과" if o else "실패", "내용": t} for n, o, t in res])), "", "## 3 손 검산", "", *t3]
    (out / "checks.md").write_text("\n".join(L), encoding="utf-8")
    for n, o, t in res:
        print(("PASS " if o else "FAIL ") + n + ": " + t)
    sys.exit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
