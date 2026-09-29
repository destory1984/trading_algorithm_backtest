"""SPEC 5: the six checks. Run after portfolio.py and baseline.py.
Usage: python -m src.checks   -> results/checks.md (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import common as C
from src import signals as SG
from src.portfolio import Panel, simulate


def raw_valid(cfg: dict, ticker: str) -> pd.DataFrame:
    """Raw prices, sealed, traded rows only (what krxbt keeps as valid)."""
    from krxbt.data import load_prices
    px = load_prices(cfg, ticker)
    px = px[~px.index.duplicated(keep="last")]
    px = px[px.index <= C.end(cfg)]
    ok = (px["volume"] > 0) & (px[["open", "high", "low", "close"]] > 0).all(axis=1)
    return px[ok]


def wilder_n(v: pd.DataFrame, n: int) -> pd.Series:
    """N by a plain loop, independently of signals.wilder."""
    pc = v["close"].shift(1)
    tr = pd.concat([v["high"] - v["low"], (v["high"] - pc).abs(), (pc - v["low"]).abs()], axis=1).max(axis=1)
    out = pd.Series(np.nan, index=v.index)
    vals = tr.to_numpy()
    if len(vals) > n:
        cur = vals[1: n + 1].mean()
        out.iloc[n] = cur
        for i in range(n + 1, len(vals)):
            cur = ((n - 1) * cur + vals[i]) / n
            out.iloc[i] = cur
    return out


def seal_check(cfg) -> tuple[bool, str]:
    out = C.results_dir()
    eq = pd.read_parquet(out / "equity.parquet")
    tr = pd.read_parquet(out / "trades.parquet")
    cal = SG.load()["cal"]
    days = [eq.index.max(), tr["exit_day"].max(), cal.max()]
    return all(d <= C.end(cfg) for d in days), f"마지막 날: 곡선 {days[0].date()}, 거래 {days[1].date()}, 달력 {days[2].date()}"


def hand_check(cfg) -> tuple[bool, list[str]]:
    tr = pd.read_parquet(C.results_dir() / "trades.parquet")
    cs = cfg["costs"]
    tc = cfg["turtle"]
    buy_k = (1 + cs["slippage"]) * (1 + cs["buy_fee"])
    lines, ok = [], True
    for r, s in cfg["rules"].items():
        k = C.vkey(r, s["entry"])
        t = tr[(tr["variant"] == k) & (tr["reason"] != "break_void") & (tr["reason"] != "open_at_end")]
        picks = [t[t["units"] >= 2].iloc[0], t[t["reason"] == "stop"].iloc[0], t[t["reason"] == "channel"].iloc[0]]
        lines += [f"### {r}", ""]
        for x in picks[: cfg["checks"]["hand_trades"]]:
            v = raw_valid(cfg, x["ticker"])
            n_ser = wilder_n(v, tc["n_window"])
            e0 = x["entry_day"]
            i0 = v.index.get_loc(e0)
            n_exp = float(n_ser.iloc[i0 - 1])
            fills = [f.split("@") for f in x["fills"].split(";")]
            f_days = [pd.Timestamp(d) for d, _ in fills]
            f_px = [float(p.split("x")[0]) for _, p in fills]
            f_sh = [int(p.split("x")[1]) for _, p in fills]
            hh = float(v["high"].iloc[i0 - s["entry"]: i0].max())
            hh55 = float(v["high"].iloc[i0 - tc["failsafe_breakout"]: i0].max())
            o0 = float(v["open"].iloc[i0])
            lvl_ok = abs(f_px[0] - max(o0, hh)) < 1 or abs(f_px[0] - max(o0, hh55)) < 1
            add_ok = all(abs(f_px[j] - max(float(v.loc[f_days[j], "open"]), f_px[j - 1] + tc["add_n"] * n_exp)) < 1
                         for j in range(1, len(f_px)))
            stop = f_px[-1] - tc["stop_n"] * n_exp
            ix = v.index.get_loc(x["exit_day"])
            ox, lx = float(v["open"].iloc[ix]), float(v["low"].iloc[ix])
            llx = float(v["low"].iloc[ix - s["exit"]: ix].min())
            if x["reason"] == "stop":
                exp_px = min(ox, stop)
            elif x["reason"] == "channel":
                exp_px = min(ox, llx)
            elif x["reason"].startswith("same_day"):
                exp_px = llx if "channel" in x["reason"] else stop
            elif x["reason"] == "stop_after_add":
                exp_px = stop
            else:
                exp_px = x["exit_px"]
            cost = sum(sh * px * buy_k for sh, px in zip(f_sh, f_px))
            tax = C.sell_tax(cfg, x["exit_day"])
            proceeds = x["shares"] * x["exit_px"] * (1 - cs["slippage"]) * (1 - cs["sell_fee"] - tax)
            checks = {"N": abs(n_exp - x["n"]) < 1e-6 * max(1, n_exp), "진입가": lvl_ok, "추가가": add_ok,
                      "청산가": abs(exp_px - x["exit_px"]) < 1e-6 * exp_px, "비용": abs(cost - x["cost"]) < 1e-8 * cost,  # fills are stored to 4 decimals
                      "매도금": abs(proceeds - x["proceeds"]) < 1e-8 * proceeds}
            ok &= all(checks.values())
            lines.append(f"- {x['ticker']} {e0.date()} → {x['exit_day'].date()} ({x['reason']}, {x['units']}단위): "
                         f"N {n_exp:.1f} / 기록 {x['n']:.1f}; 직전 {s['entry']}일 최고가 {hh:.0f}, 시가 {o0:.0f}, 산 값 "
                         f"{', '.join(f'{p:.0f}' for p in f_px)}; 마지막 손절선 {stop:.0f}; 청산일 시가 {ox:.0f}, 저가 {lx:.0f}, "
                         f"직전 {s['exit']}일 최저가 {llx:.0f} → 판 값 {exp_px:.0f} / 기록 {x['exit_px']:.0f}; "
                         f"매도세 {tax:.2%}; " + ", ".join(f"{a} {'일치' if b else '불일치'}" for a, b in checks.items()))
        lines.append("")
    return ok, lines


def filter_check(cfg) -> tuple[bool, list[str]]:
    p = SG.load()
    meta, cal = p["meta"], p["cal"]
    tc = cfg["turtle"]
    lines, ok = [], True
    sk = p["skipped"][C.vkey("S1", cfg["rules"]["S1"]["entry"])]
    n = cfg["rules"]["S1"]["entry"]
    for x in sk.sample(3, random_state=cfg["checks"]["seed"]).itertuples():
        tk = meta.loc[x.tid, "ticker"]
        v = raw_valid(cfg, tk)
        n_ser = wilder_n(v, tc["n_window"])
        d, din, dout = cal[x.day], cal[x.prev_in], cal[x.prev_out]
        i = v.index.get_loc(d)
        is_brk = float(v["high"].iloc[i]) > float(v["high"].iloc[i - n: i].max())
        j = v.index.get_loc(din)
        buy = max(float(v["open"].iloc[j]), float(v["high"].iloc[j - n: j].max()))
        stop = buy - tc["stop_n"] * float(n_ser.iloc[j - 1])
        # walk the hypothetical trade forward by hand
        sell, sell_day = None, None
        for q in range(j, len(v)):
            lo, op = float(v["low"].iloc[q]), float(v["open"].iloc[q])
            ll = float(v["low"].iloc[q - cfg["rules"]["S1"]["exit"]: q].min())
            s_hit, c_hit = lo <= stop, lo < ll
            if s_hit or c_hit:
                lvl = max(stop if s_hit else -np.inf, ll if c_hit else -np.inf)
                sell = lvl if q == j else min(op, lvl)
                sell_day = v.index[q]
                break
        good = is_brk and abs(buy - x.prev_buy) < 1 and sell_day == dout and abs(sell - x.prev_sell) < 1 and sell > buy
        ok &= good
        lines.append(f"- {tk} {d.date()} 20일 돌파 {is_brk}; 직전 가상 거래 {din.date()} 에 {buy:.0f} 매수(기록 {x.prev_buy:.0f}), "
                     f"{sell_day.date() if sell_day is not None else '-'} 에 {sell:.0f} 매도(기록 {dout.date()} {x.prev_sell:.0f}) → "
                     f"이익이므로 건너뜀 {'일치' if good else '불일치'}")
    return ok, lines


def account_check(cfg) -> tuple[bool, str]:
    out = C.results_dir()
    eq = pd.read_parquet(out / "equity.parquet")
    un = pd.read_parquet(out / "units.parquet")
    tr = pd.read_parquet(out / "trades.parquet")
    cap = cfg["turtle"]["capital"]
    cs = cfg["costs"]
    tax = C.sell_tax(cfg, eq.index[-1])
    keep = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - tax)
    gaps, ok = [], True
    for k in eq.columns:
        t = tr[tr["variant"] == k]
        open_ = t[t["reason"] == "open_at_end"]
        # final equity values open positions at the close without sale costs; the trade list sells them
        expect = cap + t["pnl"].sum() + (open_["shares"] * open_["exit_px"] * (1 - keep)).sum()
        gap = abs(eq[k].iloc[-1] - expect)
        gaps.append(gap)
        ok &= gap < 1.0 and un[k].max() <= cfg["turtle"]["max_units_total"] and t["units"].max() <= cfg["turtle"]["max_units_per_stock"]
    return ok, (f"마지막 자산 = 처음 자산 + 거래 손익 합(끝에 들고 있던 것의 매도 비용 되돌림) 차이 최대 {max(gaps):.4f}원. "
                f"단위 수 최대 {int(un.max().max())}(한도 {cfg['turtle']['max_units_total']}), 종목당 최대 {int(tr['units'].max())}"
                f"(한도 {cfg['turtle']['max_units_per_stock']}). 현금 음수는 계산 중 assert 로 막는다")


def lookahead_check(cfg) -> tuple[bool, str]:
    from krxbt.data import load_prices
    cal = C.calendar(cfg)
    n = cfg["checks"]["perturb_days"]
    cut = cal[-n]
    rng = np.random.default_rng(cfg["checks"]["seed"])

    def load(c, tk):
        px = load_prices(c, tk)
        if px is None:
            return px
        px = px.copy()
        m = px.index >= cut
        f = rng.uniform(0.7, 1.3, size=m.sum())
        for col in ("open", "high", "low", "close"):
            px.loc[m, col] = px.loc[m, col] * f
        return px

    base = SG.load(mmap=False)
    pert = SG.build(cfg, load)
    bad = 0
    for r, s in cfg["rules"].items():
        k = C.vkey(r, s["entry"])
        a = simulate(cfg, Panel(base), s["exit"], base["cands"][k])["trades"]
        b = simulate(cfg, Panel(pert), s["exit"], pert["cands"][k])["trades"]
        key = ["ticker", "entry_day", "exit_day", "reason", "shares", "exit_px"]
        a = a[a["exit_day"] < cut][key].reset_index(drop=True)
        b = b[b["exit_day"] < cut][key].reset_index(drop=True)
        bad += 0 if a.equals(b) else max(1, int((a != b).any(axis=1).sum()) if len(a) == len(b) else abs(len(a) - len(b)))
    return bad == 0, f"마지막 {n}거래일({cut.date()} →) 가격을 0.7 → 1.3 배로 흔든 뒤 그 전에 끝난 거래가 바뀐 수 {bad}"


def random_check(cfg) -> tuple[bool, str]:
    p = SG.load(mmap=False)
    P = Panel(p)
    out = C.results_dir()
    s = cfg["rules"]["S1"]
    ks = pd.read_parquet(out / "entries.parquet")[C.vkey("S1", s["entry"])].to_numpy()
    res = simulate(cfg, P, s["exit"], None, random_k=ks, elig_days=p["eligible_days"], seed=cfg["random"]["seed"])
    t = res["trades"]
    e = p["eligible_days"]
    elig = set(zip(e[:, 0].tolist(), e[:, 1].tolist()))
    day_i = {d: i for i, d in enumerate(P.cal)}
    not_elig = sum((day_i[d], tid) not in elig for d, tid in zip(t["entry_day"], t["tid"]))
    overlap = 0
    for tid, g in t.sort_values("entry_day").groupby("tid"):
        prev_exit = None
        for x in g.itertuples():
            if prev_exit is not None and x.entry_day <= prev_exit:
                overlap += 1
            prev_exit = x.exit_day
    return not_elig == 0 and overlap == 0, (f"무작위 1번(시드 {cfg['random']['seed']}) 거래 {len(t)}건 중 그날 살 수 없는 종목 {not_elig}건, "
                                            f"들고 있거나 그날 판 종목을 다시 산 경우 {overlap}건")


def main() -> None:
    cfg = C.load_config()
    ok1, t1 = seal_check(cfg)
    ok2, t2 = hand_check(cfg)
    ok3, t3 = filter_check(cfg)
    ok4, t4 = account_check(cfg)
    ok6, t6 = random_check(cfg)
    ok5, t5 = lookahead_check(cfg)
    res = [("1 봉인", ok1, t1), ("2 손 검산", ok2, "아래"), ("3 S1 거르기 검산", ok3, "아래"), ("4 계좌 대조", ok4, t4),
           ("5 미래 데이터 금지", ok5, t5), ("6 무작위 진입", ok6, t6)]
    L = ["# 검증 (SPEC 5)", "", C.md_table(pd.DataFrame(
        [{"검증": n, "결과": "통과" if o else "실패", "내용": t} for n, o, t in res])), "", "## 2 손 검산", "", *t2,
        "## 3 S1 거르기 검산", "", *t3]
    (C.results_dir() / "checks.md").write_text("\n".join(L), encoding="utf-8")
    for n, o, t in res:
        print(("PASS " if o else "FAIL ") + n + ": " + t)
    for x in t2 + t3:
        print(x)
    sys.exit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
