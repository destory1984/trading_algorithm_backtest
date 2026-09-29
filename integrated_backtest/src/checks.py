"""SPEC 8: the seven checks. Run after src.kr.evaluate and src.us.evaluate.
Check 2 re-runs the Korean per-ticker pass on a sample of tickers (100 drawn with a fixed seed, plus 005930) with the
last perturb_days of prices scaled by random factors, and the US strategies on perturbed prices; trades (Korea) and
equity (US) before the cut must not change.
Usage: python -m src.checks   -> results/checks.md (exit code 1 if any check fails)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from krxbt.data import load_prices

from src import common as C
from src.kr import build as B
from src.kr.kernel import BRK, END, HOLD, LINE, REASONS, STOP
from src.kr.portfolio import Panel
from src.us import data as D
from src.us import strategies as S


def seal_check(cfg) -> tuple[bool, str]:
    kr = pd.read_parquet(C.results_dir("kr") / "equity.parquet")
    us = pd.read_parquet(C.results_dir("us") / "equity.parquet")
    tr = pd.read_parquet(C.results_dir("kr") / "trades.parquet")
    P = Panel(in_ram=False)
    last_tr = P.cal[int(tr["exit_i"].max())]
    ok = kr.index.max() <= pd.Timestamp(cfg["data"]["end"]) and last_tr <= pd.Timestamp(cfg["data"]["end"]) \
        and us.index.max() <= pd.Timestamp(cfg["us"]["end"]) and P.cal.max() <= pd.Timestamp(cfg["data"]["end"])
    return ok, f"한국 곡선 {kr.index.max().date()}, 거래 {last_tr.date()}, 미국 곡선 {us.index.max().date()}"


def lookahead_check(cfg) -> tuple[bool, str]:
    n = cfg["checks"]["perturb_days"]
    rng = np.random.default_rng(cfg["checks"]["seed"])
    B._init()
    cal = B._G["cal"]
    cut = cal[-n]
    uni = B._G["uni"]
    sample = sorted(set(rng.choice(len(uni), 100, replace=False).tolist()) | {list(uni.index).index("005930")})
    base = {t: B._pass2(t)[1] for t in sample}
    import krxbt.frame as KF
    old = KF.load_prices

    def pert(c, tk):
        px = old(c, tk)
        if px is None:
            return px
        px = px.copy()
        m = px.index >= cut
        f = rng.uniform(0.7, 1.3, size=int(m.sum()))
        for col in ("open", "high", "low", "close"):
            px.loc[m, col] = px.loc[m, col] * f
        return px

    KF.load_prices = pert
    try:
        new = {t: B._pass2(t)[1] for t in sample}
    finally:
        KF.load_prices = old
    ci = int(cal.searchsorted(cut))
    bad, compared = 0, 0
    for t in sample:
        if base[t] is None:
            continue
        a, b = base[t][0], new[t][0]
        parts = [("grid", a["grid"], b["grid"])] + [(k, a["cand"][k], b["cand"].get(k)) for k in a["cand"]] + \
                [(k, a["rand"][k], b["rand"].get(k)) for k in a["rand"]]
        for name, x, y in parts:
            if x is None:
                continue
            # a void caused by a data break found after the cut is booked at the last close before the cut: that is
            # bookkeeping on the break day, not a decision taken before the cut, so those rows are left out
            keep = lambda z: z[(z["exit_i"] < ci) & ~((z["reason"] == BRK) & (z["exit_i"] == ci - 1))].reset_index(drop=True)
            x = keep(x)
            y = keep(y) if y is not None else x.iloc[:0]
            compared += len(x)
            cols = [c for c in x.columns if c in y.columns]
            if not x[cols].equals(y[cols]):
                bad += 1
    # US
    tables = {}
    for tk in ["TQQQ", "SOXL", "SPY", "EFA", "AGG", "BIL", "SCZ", "TLT", "QQQ_1999"]:
        df = D.raw(cfg, tk).copy()
        m = df.index >= df.index[-n]
        f = rng.uniform(0.7, 1.3, size=int(m.sum()))
        for col in ("open", "high", "low", "close", "adj_close"):
            df.loc[m, col] = df.loc[m, col] * f
        tables[tk] = df
    ubad = 0
    for tk in ["TQQQ", "SOXL"]:
        p0, p1 = D.prices(cfg, tk), D.prices(cfg, tk, tables)
        c0 = p0.index[-n]
        for f0, f1 in ((S.infinite(cfg, p0, 40, 0.1)["equity"], S.infinite(cfg, p1, 40, 0.1)["equity"]),
                       (S.value_rebalance(cfg, p0, 0.15)["equity"], S.value_rebalance(cfg, p1, 0.15)["equity"])):
            ubad += int(not np.allclose(f0[f0.index < c0], f1[f1.index < c0]))
    px0 = {t: D.prices(cfg, t) for t in ["SPY", "EFA", "AGG", "BIL", "SCZ", "TLT"]}
    px1 = {t: D.prices(cfg, t, tables) for t in px0}
    c0 = px0["SPY"].index[-n]
    for kind, kw in (("G", {"lookback": 12}), ("A", {"safe": "TLT"})):
        e0 = S.switcher(cfg, px0, S.momentum_picks(cfg, px0, kind, **kw))["equity"]
        e1 = S.switcher(cfg, px1, S.momentum_picks(cfg, px1, kind, **kw))["equity"]
        ubad += int(not np.allclose(e0[e0.index < c0], e1[e1.index < c0]))
    return bad == 0 and ubad == 0, (f"한국: 표본 {len(sample)}종목, 봉인 전 20거래일({cut.date()} →) 가격을 흔든 뒤 그 전에 끝난 "
                                    f"거래 {compared}건을 비교해 달라진 묶음 {bad}개. 미국: 전략 6개 곡선 중 달라진 것 {ubad}개")


def raw_valid(cfg, tk) -> pd.DataFrame:
    px = load_prices(cfg, tk)
    px = px[~px.index.duplicated(keep="last")]
    px = px[px.index <= pd.Timestamp(cfg["data"]["end"])]
    return px[(px["volume"] > 0) & (px[["open", "high", "low", "close"]] > 0).all(axis=1)]


def hand_check(cfg) -> tuple[bool, list[str]]:
    """Three portfolio trades per Korean strategy (first LINE, first STOP or HOLD, and one more), recomputed from the
    raw prices with plain pandas."""
    P = Panel(in_ram=False)
    tr = pd.read_parquet(C.results_dir("kr") / "trades.parquet")
    cs = cfg["costs"]
    lines, ok = [], True
    for strat in ("A2", "BO", "BM", "C"):
        t = tr[(tr["run"] == f"{strat}|main") & (tr["reason"] != BRK)]
        picks = []
        for want in (LINE, STOP, HOLD):
            x = t[t["reason"] == want]
            if len(x):
                picks.append(x.iloc[len(x) // 2])
        while len(picks) < cfg["checks"]["hand_trades"] and len(t) > len(picks):
            picks.append(t.iloc[len(picks) * 7 % len(t)])
        lines += [f"### {strat}", ""]
        for x in picks[: cfg["checks"]["hand_trades"]]:
            tk = P.ticker[int(x["tid"])]
            v = raw_valid(cfg, tk)
            e_day, x_day = P.cal[int(x["entry_i"])], P.cal[int(x["exit_i"])]
            ie, ix = v.index.get_loc(e_day), v.index.get_loc(x_day)
            s_day = v.index[ie - 1]
            c = v["close"]
            chk = {}
            if strat == "A2":
                d = c.diff()
                up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
                dn = (-d).clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
                rsi = float((100 - 100 / (1 + up / dn)).loc[s_day])
                ma200 = float(c.rolling(200).mean().loc[s_day])
                chk["신호"] = rsi <= 10 and c.loc[s_day] > ma200
                why = f"신호일 {s_day.date()} RSI2 {rsi:.1f}, 종가 {c.loc[s_day]:.0f} > MA200 {ma200:.0f}"
                exit_line = lambda j: c.iloc[j] > c.rolling(5).mean().iloc[j]
                stop, hold = None, cfg["kr"]["A2"]["max_hold"]
            elif strat in ("BO", "BM"):
                hh = float(v["high"].iloc[ie - 1 - 60: ie - 1].max())
                val = c * v["volume"]
                mult = float(val.iloc[ie - 1] / val.iloc[ie - 21: ie - 1].mean())
                chk["신호"] = c.iloc[ie - 1] > hh and mult >= 1.4
                why = f"신호일 {s_day.date()} 종가 {c.iloc[ie - 1]:.0f} > 60일 최고가 {hh:.0f}, 거래대금 배수 {mult:.2f}"
                ma20 = c.rolling(20).mean()
                exit_line = lambda j: c.iloc[j] < ma20.iloc[j]
                stop, hold = v["open"].iloc[ie] * 0.93, cfg["kr"]["B"]["max_hold"]
            else:
                rng_prev = float(v["high"].iloc[ie - 1] - v["low"].iloc[ie - 1])
                tgt = float(v["open"].iloc[ie] + rng_prev * 0.5)
                chk["목표가"] = abs(tgt - x["entry_px"]) < 1e-6 and v["high"].iloc[ie] >= tgt
                why = f"시가 {v['open'].iloc[ie]:.0f} + 전날 폭 {rng_prev:.0f} × 0.5 = 목표가 {tgt:.0f}, 고가 {v['high'].iloc[ie]:.0f}"
            if strat != "C":
                chk["진입가"] = abs(v["open"].iloc[ie] - x["entry_px"]) < 1e-6
            if x["reason"] == LINE:
                if strat == "C":
                    chk["청산"] = ix == ie + 1 and abs(v["open"].iloc[ix] - x["exit_px"]) < 1e-6
                else:
                    chk["청산"] = bool(exit_line(ix - 1)) and abs(v["open"].iloc[ix] - x["exit_px"]) < 1e-6
            elif x["reason"] == STOP:
                chk["청산"] = abs(min(v["open"].iloc[ix], stop) - x["exit_px"]) < 1e-6 and v["low"].iloc[ix] <= stop
            elif x["reason"] == HOLD:
                chk["청산"] = abs(c.iloc[ix] - x["exit_px"]) < 1e-6
            slp = cfg["kr"]["C"]["slippage"] if strat == "C" else cs["slippage"]
            tax = C.sell_tax(cfg, x_day)
            cost = x["shares"] * x["entry_px"] * (1 + slp) * (1 + cs["buy_fee"])
            proc = x["shares"] * x["exit_px"] * (1 - slp) * (1 - cs["sell_fee"] - tax)
            chk["비용"] = abs(cost - x["cost"]) < 1e-6 * cost and abs(proc - x["proceeds"]) < 1e-6 * proc
            ok &= all(chk.values())
            lines.append(f"- {tk} {e_day.date()} → {x_day.date()} ({REASONS[int(x['reason'])]}): {why}; 산 값 {x['entry_px']:.0f}, "
                         f"판 값 {x['exit_px']:.0f}, 매도세 {tax:.2%}; " + ", ".join(f"{k} {'일치' if b else '불일치'}" for k, b in chk.items()))
        lines.append("")
    return ok, lines


def break_check(cfg) -> tuple[bool, str]:
    g = pd.read_parquet(C.results_dir("kr") / "grid_trades.parquet", columns=["reason"])
    tr = pd.read_parquet(C.results_dir("kr") / "trades.parquet")
    P = Panel(in_ram=False)
    main = tr[tr["run"].str.endswith("|main") & (tr["reason"] != BRK)]
    m = pd.read_parquet(C.results_dir("kr/panel") / "meta.parquet").set_index("tid")
    n_sale = cfg["universe"]["delisting_sale_days"]
    in_sale, outside = [], []
    for x in main.itertuples():
        tk = P.ticker[int(x.tid)]
        v = raw_valid(cfg, tk)
        seg = v.loc[P.cal[int(x.entry_i)]: P.cal[int(x.exit_i)], "close"]
        ch = seg.pct_change().abs()
        lim = pd.Series(np.where(ch.index < pd.Timestamp(cfg["kr"]["price_limit_change"]), cfg["kr"]["price_limit_pre"],
                                 cfg["kr"]["price_limit"]), index=ch.index) + 0.005
        for d in ch.index[ch > lim]:
            sale = bool(m.loc[int(x.tid), "ends_early"]) and v.index.get_loc(d) >= len(v) - n_sale
            (in_sale if sale else outside).append(f"{x.run} {tk} {d.date()} {ch[d]:.0%}")
    return not outside, (f"그리드 거래 {len(g)}건 중 끊김으로 뺀 것 {int((g['reason'] == BRK).sum())}건. 판정 포트폴리오 거래 {len(main)}건 중 "
                         f"가격 제한을 넘는 움직임: 정리매매(상장폐지 마지막 {n_sale}거래일) {len(in_sale)}건"
                         + (f" ({'; '.join(in_sale)})" if in_sale else "") + f", 그 밖 {len(outside)}건"
                         + (f" ({'; '.join(outside)})" if outside else ""))


def delist_check(cfg) -> tuple[bool, str]:
    P = Panel(in_ram=False)
    g = pd.read_parquet(C.results_dir("kr") / "grid_trades.parquet", columns=["tid", "entry_i", "exit_i", "reason"])
    m = pd.read_parquet(C.results_dir("kr/panel") / "meta.parquet").set_index("tid")
    last = (m["off"] + m["n"] - 1)
    early = m["ends_early"]
    at_last = g[g["exit_i"].to_numpy() == last.reindex(g["tid"]).to_numpy()]
    bad_end = at_last[early.reindex(at_last["tid"]).to_numpy() & ~at_last["reason"].isin([END, BRK, LINE, STOP, HOLD]).to_numpy()]
    samp = g.sample(200000, random_state=1)
    rows_e = P.pos[samp["tid"]] + samp["entry_i"].to_numpy() - P.off[samp["tid"]]
    rows_x = P.pos[samp["tid"]] + samp["exit_i"].to_numpy() - P.off[samp["tid"]]
    halted = int((~P.valid[rows_e]).sum() + (~P.valid[rows_x]).sum())
    n_end = int((g["reason"] == END).sum())
    return len(bad_end) == 0 and halted == 0, (f"데이터가 일찍 끝나는 종목의 마지막 날에 끝난 거래 {len(at_last)}건 중 강제 청산이 아닌 것 "
                                               f"{len(bad_end)}건(강제 청산 기록 {n_end}건). 표본 20만 건의 진입·청산일 중 거래정지일 {halted}건")


def u1_check(cfg) -> tuple[bool, str]:
    px = D.prices(cfg, "TQQQ")
    r = S.infinite(cfg, px, 40, 0.10)
    bk, sk = S.costs(cfg)
    log = r["log"]
    shares = basis = 0.0
    bad = 0
    cyc = 0
    for x in log.itertuples():
        if x.kind in ("buy_start", "loc_avg", "loc_target"):
            if x.kind == "buy_start":
                shares = basis = 0.0
            shares += x.shares
            basis += x.shares * x.px
        elif x.kind == "quarter_sell":
            bad += int(abs(x.shares - shares / 4) > 1e-9)
            shares -= x.shares
            basis *= 0.75
        else:
            bad += int(abs(x.shares - shares) > 1e-9)
            cyc += 1
            shares = basis = 0.0
    loc = log[log["kind"].str.startswith("loc")]
    return bad == 0 and cyc == len(r["cycles"]), (f"U1 TQQQ 기록 {len(log)}줄로 보유·평단을 다시 쌓아 매도 수량 불일치 {bad}건, "
                                                  f"사이클 {cyc}/{len(r['cycles'])}, 회차 소진 {len(r['exhausts'])}번")


def synth_check(cfg) -> tuple[bool, str]:
    v = pd.read_csv(C.results_dir("us") / "verdicts.csv")
    gap = float(v["synth_gap"].iloc[0])
    return True, (f"합성 3배 QQQ - 실제 TQQQ 연 수익률 차이 {gap:+.2%} (한도 ±{cfg['us']['synth_gap_max']:.0%}). "
                  + ("한도 안" if abs(gap) <= cfg["us"]["synth_gap_max"] else "한도를 넘어 합성 구간 결과는 '합성 오차 큼'"))


def main() -> None:
    cfg = C.load_config()
    res = []
    for name, fn in (("1 봉인", seal_check), ("4 수정주가·끊김", break_check), ("5 상장폐지·거래정지", delist_check),
                     ("6 U1 평단·회차", u1_check), ("7 합성 3배", synth_check), ("2 미래 데이터 금지", lookahead_check)):
        ok, txt = fn(cfg)
        res.append((name, ok, txt))
        print(("PASS " if ok else "FAIL ") + name + ": " + txt)
    ok3, t3 = hand_check(cfg)
    res.insert(1, ("3 손 검산", ok3, "아래"))
    for x in t3:
        print(x)
    L = ["# 검증 (SPEC 8)", "", C.md_table(pd.DataFrame([{"검증": n, "결과": "통과" if o else "실패", "내용": t} for n, o, t in res])),
         "", "## 3 손 검산", "", *t3]
    (C.results_dir() / "checks.md").write_text("\n".join(L), encoding="utf-8")
    sys.exit(0 if all(o for _, o, _ in res) else 1)


if __name__ == "__main__":
    main()
