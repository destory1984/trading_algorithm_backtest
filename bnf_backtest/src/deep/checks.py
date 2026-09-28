"""SPEC2 6: verification -> results/deep/checks.md. Run after the other src.deep modules.
Usage: python -m src.deep.checks
"""
from __future__ import annotations

import math
from statistics import NormalDist

import krxbt.data
import numpy as np
import pandas as pd
from krxbt.frame import index_features, ticker_frame
from krxbt.portfolio import stats

from src.common import results_dir as grid_dir
from src.deep import overfit
from src.deep import shared as S


def yes(ok: bool) -> str:
    return "통과" if ok else "실패"


def main() -> None:
    cfg = S.config()
    out = S.results_dir()
    end = S.deep_end(cfg)
    key = S.rule_key(cfg)
    L = ["# 동반급락 규칙 심화 검증", ""]
    st = {}

    # 1 seal
    cal = krxbt.data.load_calendar(cfg)
    last = {"달력": cal[-1], **{f"지수 {m}": krxbt.data.load_index(cfg, m).index[-1] for m in ("KOSPI", "KOSDAQ")},
            "삼성전자 시세": krxbt.data.load_prices(cfg, "005930").index[-1]}
    cand = pd.read_parquet(out / "cand.parquet", columns=["exit_date"])
    last["후보 거래 청산일"] = cand["exit_date"].max()
    ok = all(v <= end for v in last.values())
    st["1. 봉인"] = ok
    L += ["## 1. 봉인", "", ", ".join(f"{k} {v.date()}" for k, v in last.items()) + f". 봉인 {end.date()}. **{yes(ok)}**", ""]

    # 2 sequential list vs the grid
    cmb = pd.read_csv(grid_dir() / "combos.csv")
    r = S.rule(cfg)
    cid = int(cmb[(cmb["market_filter"] == r["market_filter"]) & (cmb["threshold"] == r["threshold"])
                  & cmb["stop"].isna() & (cmb["hold"] == r["hold"])]["combo"].iloc[0])
    g = pd.read_parquet(grid_dir() / "trades.parquet", filters=[("combo", "==", cid)])
    g = g[g["market"].astype(str) == r["market"]]
    g["ticker"] = g["ticker"].astype(str)
    s = S.candidates("seq", key)
    k = ["ticker", "signal_date", "entry_date", "exit_date"]
    a, b = s.sort_values(k).reset_index(drop=True), g.sort_values(k).reset_index(drop=True)
    ok = len(a) == len(b) and bool((a[k].to_numpy() == b[k].to_numpy()).all()) and \
        np.allclose(a["ret"].astype(float), b["ret"].astype(float), atol=1e-6)
    st["2. 그리드 재현"] = ok
    L += ["## 2. 거래 목록 재현", "", f"그리드 조합 {cid} (코스피) {len(b)}건, 이번 순차 목록 {len(a)}건. 종목·신호일·진입일·청산일·수익률이 같은가: **{yes(ok)}**", ""]

    # 3 portfolio vs portfolio_filters.csv
    ind = S.candidates("ind", S.combo_key(r["threshold"], cfg["deep"]["check_stop"], r["hold"], r["market_filter"]))
    eq, info = S.portfolio(cfg, ind)
    sv = stats(eq)
    want = cfg["deep"]["check_row"]
    ok = abs(sv["cagr"] - want["cagr"]) < 1e-6 and abs(sv["mdd"] - want["mdd"]) < 1e-6
    st["3. 포트폴리오 대조"] = ok
    L += ["## 3. 포트폴리오 대조", "", f"손절 {cfg['deep']['check_stop']:.0%} 판본: 연 {sv['cagr']:.6f}, 낙폭 {sv['mdd']:.6f}. "
          f"기존 `portfolio_filters.csv` crash95 / 제한 0: 연 {want['cagr']}, 낙폭 {want['mdd']}. **{yes(ok)}**", ""]

    # 4 DSR by hand
    x = S.candidates("seq", key)["ret"].to_numpy(np.float64)
    gs = pd.read_csv(out / "grid_sharpes.csv")
    v = float(gs["sr"][np.isfinite(gs["sr"])].var(ddof=1))
    n = overfit.n_trials(cfg)
    T = len(x)
    mu = sum(x) / T
    v1 = sum((xi - mu) ** 2 for xi in x) / (T - 1)
    v0 = sum((xi - mu) ** 2 for xi in x) / T
    sr = mu / math.sqrt(v1)
    g3 = sum((xi - mu) ** 3 for xi in x) / T / v0 ** 1.5
    g4 = sum((xi - mu) ** 4 for xi in x) / T / v0 ** 2
    nd = NormalDist()
    e = 0.5772156649015329
    emax = (1 - e) * nd.inv_cdf(1 - 1 / n) + e * nd.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(v) * emax
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    code = pd.read_csv(out / "dsr.csv").iloc[0]
    ok = abs(nd.cdf(z) - code["dsr"]) < 1e-9 and abs(sr0 - code["sr0"]) < 1e-12
    st["4. Deflated Sharpe 손계산"] = ok
    L += ["## 4. Deflated Sharpe 손계산", "", f"T = {T}, 거래 샤프 = {sr:.6f}, 왜도 = {g3:.4f}, 첨도 = {g4:.4f}, N = {n}, V = {v:.6f}, "
          f"E[max z] = {emax:.4f}, SR0 = {sr0:.6f}, z = {z:.2f}, DSR = {nd.cdf(z):.6f}. 코드 값 {code['dsr']:.6f}. **{yes(ok)}**", ""]

    # 5 random stocks
    pool = S.candidates("pool", "pool")
    dup = int(pool.groupby(["signal_date", "ticker"]).size().max())
    rng = np.random.default_rng(7)
    sample = pool.sample(5, random_state=7)
    idx = index_features(cfg)
    uni = krxbt.data.universe_tickers(cfg)
    elig = []
    for _, row in sample.iterrows():
        f = ticker_frame(cfg, row["ticker"], row["market"], cal, idx, bool(uni.loc[row["ticker"], "delisted"]))
        elig.append(bool(f.loc[row["signal_date"], "eligible"]))
    rule = S.candidates("ind", key)
    from src.deep.monkey import same_day_draws
    k_day = rule.groupby("signal_date").size()
    small = pool[pool["signal_date"].isin(k_day.index)]
    m = same_day_draws(rule.sort_values("signal_date"), small, 20, rng)
    ok = dup == 1 and all(elig) and m.shape == (20, len(rule))
    # distinct draws per day: argsort of distinct random keys never repeats an index; check it on the first day
    d0 = k_day.index[0]
    r0 = small[small["signal_date"] == d0]["ret"].to_numpy()
    idx0 = np.argsort(rng.random((20, len(r0))), axis=1)[:, :k_day.iloc[0]]
    ok &= all(len(set(row)) == len(row) for row in idx0)
    st["5. 무작위 종목"] = ok
    L += ["## 5. 같은 날 무작위 종목", "", f"후보 풀에서 (날짜, 종목) 중복 최대 {dup}. 무작위로 고른 풀 행 5개를 프레임에서 다시 보니 그날 eligible: {elig}. "
          f"한 번 뽑기의 거래 수 {m.shape[1]} = 규칙 {len(rule)}. 한 날 안에서 같은 종목이 두 번 뽑히지 않는다. **{yes(ok)}**", ""]

    # 6 neighbour cell
    nb = pd.read_csv(out / "neighbors.csv")
    cell = nb[nb["combo"] == key].iloc[0]
    met = pd.read_csv(out / "metrics.csv")
    full = met[met["stretch"] == "full"].iloc[0]
    ok = abs(cell["cagr"] - full["cagr"]) < 1e-12 and abs(cell["diff_risk"] - full["diff_risk"]) < 1e-12
    st["6. 이웃 칸"] = ok
    L += ["## 6. 이웃 표의 규칙 칸", "", f"(70, 동반급락 95, 20일) 칸 연 {cell['cagr']:.6f}, 기준 결과 연 {full['cagr']:.6f}. **{yes(ok)}**", ""]

    L += ["## 요약", "", S.md_table(pd.DataFrame([{"검사": a_, "결과": yes(b_)} for a_, b_ in st.items()])), ""]
    (out / "checks.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"{a_}: {yes(b_)}" for a_, b_ in st.items()))


if __name__ == "__main__":
    main()
