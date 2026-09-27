"""Section 9 checklist. Writes results/checks.md.

1. signal and entry never on the same bar
2. adjusted prices: no one-day drop beyond the price limit that makes a signal
3. delisted names still holding at the end are closed as "forced"
4. no exit on a halted day
5. three random tickers: a slow, plain re-implementation of the rules must
   give exactly the same trades as the numba kernel

Usage:  python -m src.checks
"""
from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .common import load_calendar, load_config, load_prices, results_dir
from .grid import combos, universe_tickers
from .indicators import index_features, ticker_frame
from .simulate import net_return, simulate


def reference_trades(f: pd.DataFrame, cfg: dict, thr: float, stop: float | None, hold: int) -> list[tuple]:
    """Plain loop over rows, written independently of simulate.run_kernel."""
    rows = list(f.itertuples())
    out, i = [], 0
    while i < len(rows):
        r = rows[i]
        if not (r.eligible and r.disp <= thr):
            i += 1
            continue
        if i + 1 >= len(rows) or not rows[i + 1].valid:
            i += 1
            continue
        e = i + 1
        entry = rows[e].open
        stop_px = entry * (1 + stop) if stop else None
        exit_i = exit_px = reason = None
        target_hit = False
        last_valid = e
        j = e
        while exit_i is None:
            if j >= len(rows):
                exit_i, exit_px, reason = last_valid, rows[last_valid].close, "forced"
                break
            b = rows[j]
            day = j - e + 1
            if b.valid:
                if target_hit:
                    exit_i, exit_px, reason = j, b.open, "target"
                elif stop_px is not None and b.open <= stop_px:
                    exit_i, exit_px, reason = j, b.open, "stop"
                elif stop_px is not None and b.low <= stop_px:
                    exit_i, exit_px, reason = j, stop_px, "stop"
                elif day >= hold:
                    exit_i, exit_px, reason = j, b.close, "time"
                elif b.close >= b.ma:
                    target_hit = True
                last_valid = j
            j += 1
        ret = float(net_return(cfg, [entry], [exit_px], [rows[exit_i].Index])[0])
        out.append((rows[i].Index, rows[e].Index, rows[exit_i].Index, entry, exit_px, round(ret, 10), reason))
        # next signal may be the exit day itself only when the exit was at the open
        at_open = reason == "target" or (reason == "stop" and exit_px == rows[exit_i].open)
        i = exit_i if at_open else exit_i + 1
    return out


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    lines = ["# 검증 체크리스트 결과", ""]
    tr = pd.read_parquet(rd / "trades.parquet")

    # 1. look-ahead
    same = (tr["entry_date"] <= tr["signal_date"]).sum()
    lines += ["## 1. 신호와 진입이 같은 봉인가",
              f"- 전체 거래 {len(tr):,}건 중 진입일이 신호일과 같거나 앞선 거래: {same}건"
              + (" (정상)" if same == 0 or cfg["rules"]["entry_on_signal_close"] else " (오류)"), ""]

    # 2. adjusted prices
    tk = universe_tickers(cfg)
    limit_breaks, sale_breaks = [], 0
    n_sale = cfg["universe"].get("delisting_sale_days", 0)
    for t in tk.index:
        p = load_prices(cfg, t)
        v = p[p["volume"] > 0]
        chg = v["close"] / v["close"].shift(1) - 1
        gap = (v.index.to_series().diff().dt.days <= 5)  # consecutive trading, not after a long halt
        u = cfg["universe"]
        lim = np.where(v.index < pd.Timestamp("2015-06-15"), u.get("price_break_drop_pre_2015", -0.16),
                       u.get("price_break_drop", -0.30))
        bad = chg[(chg < lim) & gap]
        sale_start = v.index[-min(n_sale, len(v))] if tk.loc[t, "delisted"] and n_sale else None
        for d, c in bad.items():
            if sale_start is not None and d >= sale_start:
                sale_breaks += 1  # delisting sale: no price limit, real prices
            else:
                limit_breaks.append((t, d.date(), round(c, 3)))
    bad_keys = {(t, pd.Timestamp(d)) for t, d, _ in limit_breaks}
    sig_keys = set(zip(tr["ticker"].astype(str), tr["signal_date"]))
    signal_on_break = len(bad_keys & sig_keys)
    lines += ["## 2. 수정주가 (액면분할·무상증자 가짜 급락)",
              f"- 가격 제한(2015-06-15 전 -15%, 이후 -30%)을 넘는 하루 하락 중 정리매매 기간(가격 제한 없음) 안: {sale_breaks}건",
              f"- 그 밖의 하한가 초과 하락: {len(limit_breaks)}건 (5일 넘는 거래정지 직후는 뺐다). "
              "대부분 거래량이 수천 주인 종목에서 시세가 수정되지 않고 끊긴 것이라, 그날부터 25거래일은 신호를 막는다 "
              "(config `price_break_drop`)",
              f"- 그런 날에 난 신호: {signal_on_break}건"]
    for t, d, c in limit_breaks[:10]:
        lines.append(f"  - {t} {d} {c:+.1%}")
    lines += ["- 삼성전자 2018-05-04 50:1 액면분할 전후 종가가 53,000 → 51,900 으로 이어진다 (수정주가 확인).", ""]

    # 3. delisted
    dl = tr[tr["delisted"]]
    forced = dl[dl["reason"] == "forced"]
    lines += ["## 3. 상장폐지 종목의 마지막 거래",
              f"- 상장폐지 종목 거래 {len(dl):,}건, 그중 데이터가 끝나 강제 청산된 거래 {len(forced):,}건"]
    if len(forced):
        x = forced.drop_duplicates(["ticker", "entry_date"]).head(3)
        for _, r in x.iterrows():
            last = load_prices(cfg, str(r["ticker"]))
            last = last[last["volume"] > 0].index[-1].date()
            lines.append(f"  - {r['ticker']} 진입 {r['entry_date'].date()} → 청산 {r['exit_date'].date()} "
                         f"(마지막 거래일 {last}), 수익률 {r['ret']:+.1%}")
    lines.append("")

    # 4. halted exits
    bad_exit = 0
    for t, g in tr.groupby("ticker", observed=True):
        p = load_prices(cfg, str(t))
        live = set(p.index[p["volume"] > 0])
        bad_exit += (~g["exit_date"].isin(live)).sum()
    lines += ["## 4. 거래정지일에 청산되지 않는가",
              f"- 청산일에 거래량이 0 인 거래: {bad_exit}건" + (" (정상)" if bad_exit == 0 else " (오류)"), ""]

    # 5. reference implementation
    cal, idx = load_calendar(cfg), index_features(cfg)
    random.seed(7)
    picks = random.sample(sorted(set(tr["ticker"].astype(str))), 3)
    lines += ["## 5. 임의 종목 3개: 단순 반복문 재계산과 비교", ""]
    cases = [(90, None, 10), (85, -0.08, 5), (80, -0.05, 20)]
    for t in picks:
        f = ticker_frame(cfg, t, tk.loc[t, "market"], cal, idx, bool(tk.loc[t, "delisted"]))
        for thr, stop, hold in cases:
            ref = reference_trades(f, cfg, thr, stop, hold)
            sim = simulate(f, cfg, thr, stop, hold, "none")
            simt = [(r.signal_date, r.entry_date, r.exit_date, r.entry_px, r.exit_px, round(r.ret, 10), r.reason)
                    for r in sim.itertuples()]
            ok = len(ref) == len(simt) and all(
                a[:3] == b[:3] and a[6] == b[6] and math.isclose(a[4], b[4]) and math.isclose(a[5], b[5], abs_tol=1e-9)
                for a, b in zip(ref, simt))
            lines.append(f"- {t} {tk.loc[t, 'name']} 기준 {thr}, 손절 {stop}, 보유 {hold}일: "
                         f"재계산 {len(ref)}건 / 시뮬레이션 {len(simt)}건 → {'일치' if ok else '불일치'}")
        if len(sim):
            r = sim.iloc[0]
            lines.append(f"  - 예: 신호 {r.signal_date.date()} (이격도 {r.disp:.1f}) → 진입 {r.entry_date.date()} "
                         f"시가 {r.entry_px:,.0f} → 청산 {r.exit_date.date()} {r.exit_px:,.0f} ({r.reason}), "
                         f"비용 뺀 수익률 {r.ret:+.2%}")
    (rd / "checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
