"""SPEC 5: verification -> results/checks.md. Run after the other modules.
Usage: python -m src.checks
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from krxbt import engine
from krxbt.costs import cost_factors

from src import common as C
from src import monkey, overfit


def yes(ok: bool) -> str:
    return "통과" if ok else "실패"


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    end = C.end(cfg)
    cal, frames, spx = C.load_frames(cfg)
    L = ["# 검증", ""]
    st = {}

    # 1 seal
    lasts = [cal[-1], spx.index[-1], *[f.index[-1] for f in frames.values()]]
    lasts += [pd.read_parquet(out / f"trades_{t}_{v}.parquet")["exit_date"].max() for t in C.tickers(cfg) for v in C.VERSIONS]
    ok = max(lasts) <= end
    st["1. 봉인"] = ok
    L += ["## 1. 봉인", "", f"달력, S&P500, 종목 프레임 {len(frames)}개, 거래 목록 {2 * len(frames)}개의 가장 늦은 날 {max(lasts).date()} (봉인 {end.date()}). **{yes(ok)}**", ""]

    # 2 SPY unfiltered vs ibs_backtest combo 126 and us_shortterm deep
    ours = pd.read_parquet(out / "trades_SPY_all.parquet")
    p = C.ROOT / cfg["checks"]["ibs_backtest_trades"]
    ib = pd.read_parquet(p, filters=[("combo", "==", int(cfg["checks"]["ibs_backtest_combo"]))])
    ib = ib[ib["exit_date"] <= end]
    k = ["signal_date", "entry_date", "exit_date"]
    same = len(ours) == len(ib) and bool((ours[k].to_numpy() == ib[k].to_numpy()).all()) and \
        np.allclose(ours[["entry_px", "exit_px"]].to_numpy(), ib[["entry_px", "exit_px"]].to_numpy(), rtol=1e-9)
    us = pd.read_csv(C.ROOT / cfg["checks"]["us_shortterm_metrics"])
    n_us = int(us[(us["stretch"] == "full") & (us["ticker"] == "SPY")]["trades"].iloc[0])
    ok = same and n_us == len(ours)
    st["2. 필터 없는 SPY 재현"] = ok
    L += ["## 2. 필터 없는 IBS 의 SPY 거래", "", f"이번 {len(ours)}건, ibs_backtest 조합 126 {len(ib)}건(날짜·가격 같음: {'예' if same else '아니오'}), "
          f"us_shortterm 2차 전체 구간 {n_us}건. us_shortterm 2차 검증 3 이 같은 조합 126 과 529건 모두 같음을 이미 확인했다. **{yes(ok)}**", ""]

    # 3 bear trades are below the SMA; overlap with the unfiltered list
    rows, ok = [], True
    for t in C.tickers(cfg):
        b = pd.read_parquet(out / f"trades_{t}_bear.parquet")
        a = pd.read_parquet(out / f"trades_{t}_all.parquet")
        ok &= bool(b["below"].all())
        only = len(set(b["signal_date"]) - set(a["signal_date"]))
        rows.append({"종목": t, "약세장 판본 거래": len(b), "신호일 모두 200일선 아래": "예" if b["below"].all() else "아니오",
                     "필터 없는 판본에 없는 신호일": only})
    st["3. 약세장 거래"] = ok
    L += ["## 3. 약세장 판본 거래", "", "필터 없는 판본에서는 보유 중이라 무시된 신호가 약세장 판본에서는 거래가 될 수 있어, 약세장 판본이 필터 없는 판본의 부분집합은 아니다.", "",
          C.md_table(pd.DataFrame(rows)), "", f"**{yes(ok)}**", ""]

    # 4 hand check
    rows, ok = [], True
    for t in cfg["tickers"]["judged"]:
        raw = C.raw_prices(cfg, t)
        kf = raw["adj_close"] / raw["close"]
        adj = pd.DataFrame({"open": raw["open"] * kf, "high": raw["high"] * kf, "low": raw["low"] * kf, "close": raw["adj_close"]})
        s200 = adj["close"].rolling(int(cfg["rule"]["sma"])).mean()
        x = (adj["close"] - adj["low"]) / (adj["high"] - adj["low"])
        tr = pd.read_parquet(out / f"trades_{t}_bear.parquet")
        for j in (0, len(tr) // 2)[: int(cfg["checks"]["hand_trades"])]:
            r = tr.iloc[j]
            sd, e, xd = r["signal_date"], r["entry_date"], r["exit_date"]
            cond = adj.index[adj.index.get_loc(xd) - 1]
            m = (x[sd] < 0.2 and adj.loc[sd, "close"] < s200[sd] and x[cond] > 0.8
                 and np.isclose(adj.loc[e, "open"], r["entry_px"], rtol=1e-12) and np.isclose(adj.loc[xd, "open"], r["exit_px"], rtol=1e-12))
            ok &= bool(m)
            rows.append({"종목": t, "신호일": sd.date(), "IBS": f"{x[sd]:.3f}", "종가": f"{adj.loc[sd, 'close']:.2f}",
                         "200일선": f"{s200[sd]:.2f}", "진입일": e.date(), "진입가": f"{r['entry_px']:.4f}",
                         "원표 시가": f"{adj.loc[e, 'open']:.4f}", "청산 조건일 IBS": f"{cond.date()} {x[cond]:.3f}",
                         "청산일": xd.date(), "청산가": f"{r['exit_px']:.4f}", "원표 시가 ": f"{adj.loc[xd, 'open']:.4f}",
                         "일치": "예" if m else "아니오"})
    st["4. 손 검산"] = ok
    L += ["## 4. 손 검산 (판정 종목, 첫 거래와 가운데 거래)", "", "원표는 원래 파일의 시가·고가·저가에 adj_close / close 를 곱해 다시 만든 가격이다.", "",
          C.md_table(pd.DataFrame(rows)), "", f"**{yes(ok)}**", ""]

    # 5 random entries
    rng = np.random.default_rng(3)
    bc, sk = cost_factors(cfg)
    ok = True
    for t in cfg["tickers"]["judged"]:
        f = frames[t]
        dates, _ = C.curve_inputs(f, cfg)
        o = f.loc[dates, "open"].to_numpy(np.float64)
        sg = C.signals(f, cfg)
        kk = np.asarray(f.index >= dates[0])
        pool = np.flatnonzero(sg["below"][kk] & engine.arrays(f)["eligible"][kk])
        tr = pd.read_parquet(out / f"trades_{t}_bear.parquet")
        ix = pd.Index(dates)
        spans = ix.get_indexer(pd.DatetimeIndex(tr["exit_date"])) - ix.get_indexer(pd.DatetimeIndex(tr["entry_date"]))
        below = set(pool.tolist())
        for _ in range(20):
            s, x, r = monkey.random_run(o, pool, spans, len(tr), bc, sk, rng)
            order = np.argsort(s)
            s, x = s[order], x[order]
            ok &= len(r) == len(tr) and all(v in below for v in s) and bool((s[1:] >= x[:-1]).all())
    st["5. 무작위 진입"] = ok
    L += ["## 5. 무작위 진입", "", "판정 종목마다 20번씩 뽑아, 거래 수가 전략과 같고, 모든 신호일이 200일선 아래 eligible 날이며, 다음 신호가 앞 거래의 청산일 이후(당일 허용)인지 보았다. "
          f"**{yes(ok)}**", ""]

    # 6 DSR by hand
    xs = pd.read_parquet(out / "basket_daily.parquet")["bear"].to_numpy(np.float64)
    n = overfit.n_trials(cfg)
    v, _ = overfit.trial_variance(cfg)
    T = len(xs)
    mu = sum(xs) / T
    v1 = sum((a - mu) ** 2 for a in xs) / (T - 1)
    v0 = sum((a - mu) ** 2 for a in xs) / T
    sr = mu / math.sqrt(v1)
    g3 = sum((a - mu) ** 3 for a in xs) / T / v0 ** 1.5
    g4 = sum((a - mu) ** 4 for a in xs) / T / v0 ** 2
    nd = NormalDist()
    e = 0.5772156649015329
    emax = (1 - e) * nd.inv_cdf(1 - 1 / n) + e * nd.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(v) * emax
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    code = pd.read_csv(out / "dsr.csv").iloc[0]
    ok = abs(nd.cdf(z) - code["dsr"]) < 1e-9
    st["6. Deflated Sharpe 손계산"] = ok
    L += ["## 6. Deflated Sharpe 손계산 (판정 바구니)", "", f"T = {T}, 일별 샤프 = {sr:.6f}, 왜도 = {g3:.3f}, 첨도 = {g4:.2f}, N = {n}, V = {v:.6f}, "
          f"E[max z] = {emax:.4f}, SR0 = {sr0:.6f}, z = {z:.3f}, DSR = {nd.cdf(z):.6f}. 코드 값 {code['dsr']:.6f}. **{yes(ok)}**", ""]

    L += ["## 요약", "", C.md_table(pd.DataFrame([{"검사": a, "결과": yes(b)} for a, b in st.items()])), ""]
    (out / "checks.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"{a}: {yes(b)}" for a, b in st.items()))


if __name__ == "__main__":
    main()
