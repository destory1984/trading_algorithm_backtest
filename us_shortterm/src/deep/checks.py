"""SPEC2 6: verification -> results/deep/checks.md. Run after the other src.deep modules.

1 seal             second-round tables end <= deep_end; first-round loading still ends <= screen_end
2 screen repro     the screen stretch equals the first round's S1 open rows in results/screen/metrics.csv
3 ibs_backtest     SPY trades vs ibs_backtest combo 126 (same rule) up to deep_end
4 DSR by hand      SPY's DSR recomputed step by step without the module's functions
5 random entries   sample runs: trade count equals the rule's, trades never overlap
6 neighbour cell   (0.20, 0.80) in neighbors.csv equals the first round
Usage: python -m src.deep.checks
"""
from __future__ import annotations

import math
from statistics import NormalDist

import krxbt.us
import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from src import common
from src.deep import base, monkey, overfit

ROOT_IBS = common.ROOT.parent / "ibs_backtest" / "results"
COLS = ["trades", "win_rate", "expectancy", "profit_factor", "cagr", "sharpe", "mdd", "exposure", "cagr_zero_cost",
        "breakeven_bp", "risk_w", "risk_cagr", "exp_cagr", "diff_risk", "diff_exp"]


def yes(ok: bool) -> str:
    return "통과" if ok else "실패"


def main() -> None:
    cfg = common.load_config()
    out = common.results_dir("deep")
    end, send = base.deep_end(cfg), common.screen_end(cfg)
    L = ["# 2차 심화 검증", ""]
    status = {}

    # 1 seal
    cal_d, fd = base.load(cfg, deep=True)
    cal_s, fs = base.load(cfg, deep=False)
    with common.sealed_loaders(cfg, end=end):
        idx_last = {n: krxbt.us.load_index(cfg, n).index[-1] for n in ("SPX", "NDX", "DJI")}
    last_d = max([cal_d[-1], *[f.index[-1] for f in fd.values()], *idx_last.values()])
    last_s = max([cal_s[-1], *[f.index[-1] for f in fs.values()]])
    ok = last_d <= end and last_s <= send
    status["1. 봉인"] = ok
    L += ["## 1. 봉인", "", f"2차 표의 가장 늦은 날 {last_d.date()} (봉인 {end.date()}), 1차 표의 가장 늦은 날 {last_s.date()} "
          f"(봉인 {send.date()}). **{yes(ok)}**", ""]

    # 2 screen reproduction
    m1 = pd.read_csv(common.results_dir() / "metrics.csv")
    m1 = m1[(m1["strategy"] == "S1") & (m1["version"] == "open")].set_index("ticker")
    m2 = pd.read_csv(out / "metrics.csv")
    m2 = m2[m2["stretch"] == "screen"].set_index("ticker")
    diff = (m1.loc[m2.index, COLS].astype(float) - m2[COLS].astype(float)).abs().max().max()
    ok = bool(diff < 1e-12)
    status["2. 선별 구간 재현"] = ok
    L += ["## 2. 선별 구간 재현", "", f"1차 `metrics.csv` 의 S1 open 4행과 2차 선별 구간 4행의 지표 {len(COLS)}개 차이 최대 {diff:.1e}. "
          f"**{yes(ok)}**", ""]

    # 3 ibs_backtest
    combo = int(cfg["deep"]["ibs_backtest_combo"])
    p = ROOT_IBS / "grid_trades.parquet"
    if p.exists():
        c = pd.read_csv(ROOT_IBS / "combos.csv").set_index("combo").loc[combo]
        ib = pd.read_parquet(p)
        ib = ib[ib["combo"] == combo]
        ours = base.rule_trades(fd["SPY"], cfg)
        ib = ib[ib["exit_date"] <= end]
        both = min(len(ours), len(ib))
        key = ["signal_date", "entry_date", "exit_date"]
        same_dates = len(ours) == len(ib) and bool((ours[key].to_numpy() == ib[key].to_numpy()).all())
        px = np.allclose(ours[["entry_px", "exit_px"]].to_numpy()[:both], ib[["entry_px", "exit_px"]].to_numpy()[:both],
                         rtol=1e-9) if same_dates else False
        ok = same_dates and px
        L += ["## 3. ibs_backtest 대조 (SPY)", "",
              f"ibs_backtest 조합 {combo} ({c['ticker']}, 진입 {c['entry']}, 청산 {c['exit']}, {c['exec']}, 필터 {c['trend']}) "
              f"{len(ib)}건, 2차 {len(ours)}건. 신호일·진입일·청산일 같음: {'예' if same_dates else '아니오'}, "
              f"진입가·청산가 같음(상대 1e-9): {'예' if px else '아니오'}. **{yes(ok)}**", ""]
        if not ok:
            k1 = set(map(tuple, ours[key].astype(str).to_numpy()))
            k2 = set(map(tuple, ib[key].astype(str).to_numpy()))
            L += [f"2차에만 있는 거래 {len(k1 - k2)}건, ibs_backtest 에만 있는 거래 {len(k2 - k1)}건. "
                  f"예: {sorted(k1 - k2)[:3]} / {sorted(k2 - k1)[:3]}", ""]
    else:
        ok = False
        L += ["## 3. ibs_backtest 대조", "", f"`{p}` 가 없어 대조하지 못했다. **실패**", ""]
    status["3. ibs_backtest 대조"] = ok

    # 4 DSR by hand (SPY)
    _, st = base.all_stretches(cfg)
    x = st[("screen", "SPY")]["daily"].to_numpy()
    n = overfit.n_trials(cfg)
    v = overfit.trial_variance()
    T = len(x)
    mu = sum(x) / T
    var1 = sum((xi - mu) ** 2 for xi in x) / (T - 1)
    var0 = sum((xi - mu) ** 2 for xi in x) / T
    sr = mu / math.sqrt(var1)
    g3 = sum((xi - mu) ** 3 for xi in x) / T / var0 ** 1.5
    g4 = sum((xi - mu) ** 4 for xi in x) / T / var0 ** 2
    nd = NormalDist()
    emax = (1 - 0.5772156649015329) * nd.inv_cdf(1 - 1 / n) + 0.5772156649015329 * nd.inv_cdf(1 - 1 / (n * math.e))
    sr0 = math.sqrt(v) * emax
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    hand = nd.cdf(z)
    code = pd.read_csv(out / "dsr.csv")
    code = code[(code["ticker"] == "SPY") & (code["role"] == "판정")].iloc[0]
    ok = abs(hand - code["dsr"]) < 1e-9 and abs(sr0 - code["sr0"]) < 1e-12
    status["4. Deflated Sharpe 손계산"] = ok
    L += ["## 4. Deflated Sharpe 손계산 (SPY, 선별 구간)", "",
          f"T = {T}, 일별 샤프 = {sr:.6f}, 왜도 = {g3:.4f}, 첨도 = {g4:.4f}, N = {n}, V = {v:.6f}, "
          f"E[max z] = {emax:.4f}, SR0 = √V × E[max z] = {sr0:.6f}, z = {z:.3f}, DSR = Φ(z) = {hand:.6g}. "
          f"코드 값 {code['dsr']:.6g}. **{yes(ok)}**", ""]

    # 5 random entries
    rng = np.random.default_rng(1)
    bc, sk = cost_factors(cfg)
    ok = True
    for t in cfg["screen"]["tickers"]:
        f = fs[t]
        dates, _ = base.curve_inputs(f, cfg)
        o = f.loc[dates, "open"].to_numpy(np.float64)
        tr = base.rule_trades(f, cfg)
        ix = pd.Index(dates)
        spans = ix.get_indexer(pd.DatetimeIndex(tr["exit_date"])) - ix.get_indexer(pd.DatetimeIndex(tr["entry_date"]))
        for _ in range(50):
            s, xx, r = monkey.random_run(o, spans, len(tr), bc, sk, rng)
            ok &= len(r) == len(tr) and bool((s[1:] >= xx[:-1]).all()) and s[0] >= 0 and xx[-1] <= len(o) - 1
    status["5. 무작위 진입"] = ok
    L += ["## 5. 무작위 진입", "", "ETF 마다 50번씩 뽑아 거래 수가 전략과 같고, 다음 무작위 신호가 앞 거래의 청산일 이후이며(청산일 당일 허용), "
          f"모든 거래가 선별 구간 안에 있는지 보았다. **{yes(ok)}**", ""]

    # 6 neighbour cell
    nb = pd.read_csv(out / "neighbors.csv")
    cell = nb[(nb["entry"].round(2) == 0.2) & (nb["exit"].round(2) == 0.8)].set_index("ticker")
    d6 = (cell.loc[m1.index, "diff_risk"] - m1["diff_risk"]).abs().max()
    ok = bool(d6 < 1e-12)
    status["6. 이웃 표 (0.2, 0.8) 칸"] = ok
    L += ["## 6. 이웃 표의 (0.2, 0.8) 칸", "", f"1차 위험 맞춘 보유 대비 차이와의 차이 최대 {d6:.1e}. **{yes(ok)}**", ""]

    L += ["## 요약", "", common.md_table(pd.DataFrame([{"검사": k, "결과": yes(v)} for k, v in status.items()])), ""]
    (out / "checks.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"{k}: {yes(v)}" for k, v in status.items()))


if __name__ == "__main__":
    main()
