"""Verification (SPEC 12) -> results/screen/checks.md. Run after run.py.

1 seal            every table's last date <= data.screen_end
2 no look-ahead   randomise SPY's last 20 trading days of 2022 and rerun all 18 settings: trades that entered before
                  that window keep signal day, entry day and entry price; trades that also exited before it are
                  identical in every column
3 hand check      3 SPY trades per setting against prices rebuilt from the raw file, with the indicator values
                  on the signal day and on the exit-condition day
4 curve check     curve's last value = prod(1 + ret) to 1e-8 for all 72 runs
5 engine vs loop  S1 (open) on SPY with a plain loop instead of the engine
6 close mask      trades changed by `target & ~cand` in the close versions
7 flat bars       days with high = low per ETF
Usage: python -m src.screen.checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from krxbt import engine

from src import common
from src.indicators import ibs, pct_b, rsi, sma
from src.screen.run import label, settings, trades_for
from src.screen.strategies import s01_ibs, s08_tom

KEY = ["signal_date", "entry_date", "exit_date", "entry_px", "exit_px", "ret"]


def adjusted(raw: pd.DataFrame) -> pd.DataFrame:
    """Dividend-adjusted OHLC rebuilt from the raw file (open x adj_close / close, ...)."""
    k = raw["adj_close"] / raw["close"]
    return pd.DataFrame({"open": raw["open"] * k, "high": raw["high"] * k, "low": raw["low"] * k,
                         "close": raw["adj_close"]})


def indicator_table(p: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    c = p["close"]
    first, last = s08_tom.month_positions(cal)
    return pd.DataFrame({
        "close": c, "high": p["high"], "low": p["low"], "sma200": sma(c, 200), "sma5": sma(c, 5), "rsi2": rsi(c, 2),
        "ibs": ibs(p["high"], p["low"], c), "pctb": pct_b(c), "min7": c.rolling(7).min(), "max7": c.rolling(7).max(),
        "mpos": first.reindex(p.index) + 1, "mpos_end": last.reindex(p.index) + 1,
    })


EXPLAIN = {  # (signal-day text, exit-condition-day text)
    "S1": (lambda r, h: f"IBS {r.ibs:.3f}", lambda r, h: f"IBS {r.ibs:.3f}"),
    "S2": (lambda r, h: f"C {r.close:.2f} SMA200 {r.sma200:.2f} RSI2 {r.rsi2:.1f}",
           lambda r, h: f"C {r.close:.2f} SMA5 {r.sma5:.2f}"),
    "S3": (lambda r, h: f"C {r.close:.2f} SMA200 {r.sma200:.2f} RSI2 {h.rsi2.iloc[-2]:.1f}+{r.rsi2:.1f}",
           lambda r, h: f"RSI2 {r.rsi2:.1f}"),
    "S4": (lambda r, h: f"C>SMA200 {r.close > r.sma200} RSI2 t-3..t " + "/".join(f"{v:.1f}" for v in h.rsi2.iloc[-4:]),
           lambda r, h: f"RSI2 {r.rsi2:.1f}"),
    "S5": (lambda r, h: f"C>SMA200 {r.close > r.sma200} %b t-2..t " + "/".join(f"{v:.3f}" for v in h.pctb.iloc[-3:]),
           lambda r, h: f"%b {r.pctb:.3f}"),
    "S6": (lambda r, h: f"C {r.close:.2f} min7 {r.min7:.2f} SMA200 {r.sma200:.2f}",
           lambda r, h: f"C {r.close:.2f} max7 {r.max7:.2f}"),
    "S7": (lambda r, h: f"C {r.close:.2f} SMA5 {r.sma5:.2f} SMA200 {r.sma200:.2f} H t-3..t "
                        + "/".join(f"{v:.2f}" for v in h.high.iloc[-4:]) + " L " + "/".join(f"{v:.2f}" for v in h.low.iloc[-4:]),
           lambda r, h: f"C {r.close:.2f} SMA5 {r.sma5:.2f}"),
    "S8": (lambda r, h: f"{int(r.mpos_end)}번째 끝 거래일", lambda r, h: f"{int(r.mpos)}번째 거래일"),
    "S9": (lambda r, h: "매일", lambda r, h: "매일"),
}


def expected_prices(sid: str, version: str, p: pd.DataFrame, t: pd.Series) -> tuple[float, float]:
    """Entry and exit price from the rebuilt table, by the rule text."""
    s, e, x = t["signal_date"], t["entry_date"], t["exit_date"]
    i = p.index.get_loc(e)
    if sid == "S10":
        return max(p["open"].iloc[i], p["open"].iloc[i] + 0.5 * (p["high"].iloc[i - 1] - p["low"].iloc[i - 1])), p.loc[x, "open"]
    if sid == "S11":
        return max(p["open"].iloc[i], p["high"].iloc[i - 1]), p.loc[x, "close"]
    if sid == "S12":
        return p.loc[e, "open"], p.loc[x, "close"]
    entry = p.loc[s, "close"] if version in ("close", "co") else p.loc[e, "open"]
    if t.get("reason") == "forced" or version == "close":
        return entry, p.loc[x, "close"]
    return entry, p.loc[x, "open"]


def hand_check(cfg, cal, frames, n: int) -> tuple[str, bool]:
    t = cfg["checks"]["perturb_ticker"]
    raw = common.raw_prices(cfg, t)
    p = adjusted(raw)
    ind = indicator_table(p, cal)
    f = frames[t]
    a = engine.arrays(f)
    rows, ok = [], True
    for mod, v in settings():
        tr = trades_for(mod, v, f, a, cal, cfg)
        picks = sorted({0, len(tr) // 2, len(tr) - 1})[:n] if len(tr) else []
        for j in picks:
            r = tr.iloc[j]
            ep, xp = expected_prices(mod.ID, v, p, r)
            match = bool(np.isclose(ep, r["entry_px"], rtol=1e-12) and np.isclose(xp, r["exit_px"], rtol=1e-12))
            ok &= match
            sig_txt = exit_txt = ""
            if mod.ID in EXPLAIN:
                s = r["signal_date"]
                hist = ind.loc[:s]
                sig_txt = EXPLAIN[mod.ID][0](ind.loc[s], hist)
                # exit-condition day: the exit day for close exits, the priced day before it for open exits
                x = r["exit_date"]
                cd = x if (v == "close" or r["reason"] == "forced") else ind.index[ind.index.get_loc(x) - 1]
                exit_txt = f"{cd.date()}: " + EXPLAIN[mod.ID][1](ind.loc[cd], ind.loc[:cd])
            rows.append({"설정": f"{mod.ID} {v}", "신호일": r["signal_date"].date(), "신호일 지표": sig_txt,
                         "진입일": r["entry_date"].date(), "진입가": f"{r['entry_px']:.4f}", "원표 진입가": f"{ep:.4f}",
                         "청산 조건일 지표": exit_txt, "청산일": r["exit_date"].date(),
                         "청산가": f"{r['exit_px']:.4f}", "원표 청산가": f"{xp:.4f}", "일치": "예" if match else "아니오"})
    return common.md_table(pd.DataFrame(rows)), ok


def lookahead(cfg, cal, frames) -> tuple[str, bool]:
    t, n = cfg["checks"]["perturb_ticker"], int(cfg["checks"]["perturb_days"])
    raw = common.raw_prices(cfg, t)
    last = raw.index[raw.index.year == 2022][-n:]
    rng = np.random.default_rng(int(cfg["checks"]["seed"]))
    bad = raw.copy()
    k = pd.Series(rng.uniform(0.9, 1.1, len(last)), index=last)  # one factor per day keeps each bar consistent
    for col in ("open", "high", "low", "close", "adj_close"):
        bad.loc[last, col] = bad.loc[last, col] * k
    _, frames2 = common.load_frames(cfg, overrides={t: bad})
    f1, f2 = frames[t], frames2[t]
    assert not np.allclose(f1.loc[last, "close"], f2.loc[last, "close"])
    a1, a2 = engine.arrays(f1), engine.arrays(f2)
    w0 = last[0]
    rows, ok = [], True
    for mod, v in settings():
        x1 = trades_for(mod, v, f1, a1, cal, cfg)
        x2 = trades_for(mod, v, f2, a2, cal, cfg)
        e1, e2 = x1[x1["entry_date"] < w0], x2[x2["entry_date"] < w0]
        same_entry = len(e1) == len(e2) and (e1[["signal_date", "entry_date", "entry_px"]].to_numpy()
                                             == e2[["signal_date", "entry_date", "entry_px"]].to_numpy()).all()
        d1, d2 = x1[x1["exit_date"] < w0], x2[x2["exit_date"] < w0]
        same_done = len(d1) == len(d2) and (d1[KEY].to_numpy() == d2[KEY].to_numpy()).all()
        changed_after = len(pd.concat([x1, x2])[KEY].astype(str).drop_duplicates()) - len(x1)
        ok &= bool(same_entry and same_done)
        rows.append({"설정": f"{mod.ID} {v}", "창 전 진입 거래": len(e1), "진입 같음": "예" if same_entry else "아니오",
                     "창 전 청산 거래": len(d1), "전부 같음": "예" if same_done else "아니오",
                     "창에 걸쳐 바뀐 거래": changed_after})
    head = f"SPY 의 {last[0].date()} → {last[-1].date()} ({n}거래일) 가격에 날마다 0.9 → 1.1 사이 무작위 배수를 곱했다(seed {cfg['checks']['seed']}).\n\n"
    return head + common.md_table(pd.DataFrame(rows)), ok


def loop_s1(f: pd.DataFrame, a: dict, cfg: dict) -> pd.DataFrame:
    """S1 open with a plain loop: signal IBS < 0.2 on an eligible day -> buy the next open; exit condition IBS > 0.8
    -> sell the next open; a signal on the exit day's close is taken; data end -> last close."""
    from krxbt.costs import cost_factors
    x = ibs(f["high"], f["low"], f["close"]).to_numpy()
    o, c, el, dates = a["open"], a["close"], a["eligible"], a["dates"]
    bc, sk = cost_factors(cfg)
    n, i, out = len(c), 0, []
    while i < n:
        if el[i] and x[i] < 0.2 and i + 1 < n:
            e = i + 1
            j = e
            while j < n and not (x[j] > 0.8 and j + 1 < n):
                j += 1
            if j >= n:
                xi, xp, nxt = n - 1, c[n - 1], n
            else:
                xi, xp, nxt = j + 1, o[j + 1], j + 1
            out.append((dates[i], dates[e], dates[xi], o[e], xp, xp * sk / (o[e] * bc) - 1))
            i = nxt
            continue
        i += 1
    return pd.DataFrame(out, columns=KEY)


def main() -> None:
    cfg = common.load_config()
    cal, frames = common.load_frames(cfg)
    end = common.screen_end(cfg)
    out = common.results_dir()
    met = pd.read_csv(out / "metrics.csv")
    lines = ["# 검증", "", f"기준: 봉인선 {end.date()}, 선별 기간 {cfg['data']['start']} → {end.date()}.", ""]
    status = {}

    # 1 seal
    rows = [{"표": "SPX 거래일 달력", "마지막 날": cal[-1].date()}]
    for name in ("SPX", "NDX", "DJI"):
        with common.sealed_loaders(cfg):
            import krxbt.us
            rows.append({"표": f"index_{name}", "마지막 날": krxbt.us.load_index(cfg, name).index[-1].date()})
    for t, f in frames.items():
        rows.append({"표": f"{t} 프레임", "마지막 날": f.index[-1].date()})
    for (name, v, t) in [(r.label, r.version, r.ticker) for r in met.itertuples()]:
        tr = pd.read_parquet(out / f"trades_{name}_{v}_{t}.parquet")
        if len(tr):
            rows.append({"표": f"거래 {name}_{v}_{t}", "마지막 날": pd.Timestamp(tr["exit_date"].max()).date()})
    seal = pd.DataFrame(rows)
    ok1 = bool((pd.to_datetime(seal["마지막 날"]) <= end).all())
    status[1] = ok1
    lines += ["## 1. 봉인", "", f"가격·지수 표 {3 + 1 + len(frames)}개와 거래 목록 {len(seal) - 4 - len(frames)}개의 마지막 날이 "
              f"모두 {end.date()} 이하인가: **{'예' if ok1 else '아니오'}**. 가장 늦은 날 {pd.to_datetime(seal['마지막 날']).max().date()}.", "",
              common.md_table(seal.head(4 + len(frames))), ""]

    # 2 look-ahead
    txt, ok2 = lookahead(cfg, cal, frames)
    status[2] = ok2
    lines += ["## 2. 미래 데이터 금지", "", txt, "", f"통과: **{'예' if ok2 else '아니오'}**", ""]

    # 3 hand check
    txt, ok3 = hand_check(cfg, cal, frames, int(cfg["checks"]["hand_trades"]))
    status[3] = ok3
    lines += ["## 3. 손 검산 (SPY, 설정마다 첫 거래·가운데 거래·마지막 거래)", "",
              "원표 가격은 원래 파일의 시가·고가·저가에 adj_close / close 를 곱해 다시 만든 값이다. "
              "청산 조건일은 종가 청산이면 청산일, 시가 청산이면 그 전 거래일이다.", "", txt, "",
              f"모든 거래의 진입가·청산가가 원표와 같은가: **{'예' if ok3 else '아니오'}**", ""]

    # 4 curve check
    diff = (met["curve_last"] - met["prod_ret"]).abs()
    ok4 = bool(met["curve_check"].all() and len(met) == 72)
    status[4] = ok4
    lines += ["## 4. 자산곡선 곱셈 검산", "", f"{len(met)}개 중 {int(met['curve_check'].sum())}개 통과. "
              f"자산곡선 마지막 값과 Π(1 + 거래 수익률)의 차이 최대 {diff.max():.1e}.", ""]

    # 5 engine vs loop
    t = cfg["checks"]["perturb_ticker"]
    f = frames[t]
    a = engine.arrays(f)
    eng = trades_for(s01_ibs, "open", f, a, cal, cfg)[KEY].reset_index(drop=True)
    lp = loop_s1(f, a, cfg)
    ok5 = len(eng) == len(lp) and bool((eng[KEY[:3]].to_numpy() == lp[KEY[:3]].to_numpy()).all()) \
        and np.allclose(eng[KEY[3:]].to_numpy(np.float64), lp[KEY[3:]].to_numpy(np.float64), rtol=1e-12, atol=0)
    status[5] = ok5
    lines += ["## 5. 엔진 대조 (S1 open, SPY)", "", f"엔진 {len(eng)}건, 단순 반복문 {len(lp)}건. "
              f"신호일·진입일·청산일·진입가·청산가·수익률이 모두 같은가: **{'예' if ok5 else '아니오'}**", ""]

    # 6 close mask
    rows = []
    for tk, fr in frames.items():
        ar = engine.arrays(fr)
        for mod, v in settings():
            if v != "close" or mod.KIND != "engine":
                continue
            m = trades_for(mod, v, fr, ar, cal, cfg)
            u = trades_for(mod, v, fr, ar, cal, cfg, mask_close=False)
            km = set(map(tuple, m[KEY[:3]].astype(str).to_numpy()))
            ku = set(map(tuple, u[KEY[:3]].astype(str).to_numpy()))
            rows.append({"설정": f"{mod.ID} close", "종목": tk, "처리 뒤 거래": len(m), "처리 전 거래": len(u),
                         "처리 전 0일 거래": int((u["entry_date"] == u["exit_date"]).sum()),
                         "처리 뒤에만 있는 거래": len(km - ku), "처리 전에만 있는 거래": len(ku - km)})
    mask = pd.DataFrame(rows)
    lines += ["## 6. close 판본의 `target & np.logical_not(cand)` 처리", "",
              "처리 전은 엔진에 청산 배열을 그대로 넘긴 결과다. 0일 거래는 신호일 종가에 사서 같은 종가에 판 거래다. "
              "거래는 (신호일, 진입일, 청산일)로 비교한다.", "",
              f"달라진 거래 합계: 처리 뒤에만 있는 거래 {int(mask['처리 뒤에만 있는 거래'].sum())}건, "
              f"처리 전에만 있는 거래 {int(mask['처리 전에만 있는 거래'].sum())}건.", "", common.md_table(mask), ""]

    # 7 flat bars
    rows = []
    start = pd.Timestamp(cfg["data"]["start"])
    for tk, fr in frames.items():
        flat = fr["valid"] & (fr["high"] == fr["low"])
        rows.append({"종목": tk, "고가 = 저가 (2007 → 봉인)": int(flat.sum()),
                     "그중 선별 기간": int(flat[fr.index >= start].sum())})
    lines += ["## 7. 고가 = 저가인 날", "", common.md_table(pd.DataFrame(rows)), ""]

    names = {1: "봉인", 2: "미래 데이터 금지", 3: "손 검산", 4: "자산곡선 곱셈 검산", 5: "엔진 대조"}
    summary = [{"검사": f"{k}. {names[k]}", "결과": "통과" if v else "실패"} for k, v in status.items()]
    summary += [{"검사": "6. close 처리로 달라진 거래", "결과": f"{int(mask['처리 전에만 있는 거래'].sum())}건 (기록)"},
                {"검사": "7. 고가 = 저가인 날", "결과": f"{sum(r['고가 = 저가 (2007 → 봉인)'] for r in rows)}일 (기록)"}]
    lines += ["## 요약", "", common.md_table(pd.DataFrame(summary)), ""]
    (out / "checks.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(f"check {k}: {'ok' if v else 'FAIL'}" for k, v in status.items()))


if __name__ == "__main__":
    main()
