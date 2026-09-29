"""SPEC 7.3 (original section 10): integrated comparison, descriptive only (no verdicts).
Series (daily returns, common window compare.start -> end): the judged new strategies (Korea 5-slot equal-weight
portfolios, US runs), earlier folders' curves (bnf deep-dive portfolio, index_timing US T3 15%, asset_alloc A1 dual
momentum and A2 10-month SMA), KOSPI total return and SPY. turtle_kr_backtest is left out: its account was near zero
by 2011, so its later returns carry no information.
  1 ranking: CAGR / MDD / Sharpe over the window; plus each Korean strategy's best-expectancy grid combo (trade level,
    >= 200 trades, market ALL; picked on results, so not judged)
  2 monthly-return correlation matrix
  3 regime winners: month m's regime from the previous month-end of the strategy's own index (KOSPI or SPY): close
    above / below its 120-day MA x 20-day realised vol above / below its median over the window
  4 mixes: equal-weight monthly-rebalanced mixes of the 2 and 3 series with the lowest mean pairwise correlation
  5 robustness: train (2015 -> 2020) vs test (2021 ->) rank correlation of CAGR across series, and of the mean trade
    return across each Korean grid (all combos x markets with >= 30 trades in both halves)
Won and dollar returns are mixed month by month; exchange rates are ignored.
Usage: python -m src.compare   -> results/compare/*.csv, equity_compare.png, correlation.png
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from krxbt.data import load_index

from src import common as C
from src.us import data as D

OUT = "compare"
BASE = C.ROOT.parent


def series(cfg) -> tuple[pd.DataFrame, dict]:
    kr = pd.read_parquet(C.results_dir("kr") / "equity.parquet")
    us = pd.read_parquet(C.results_dir("us") / "equity.parquet")
    capk, capu = cfg["kr"]["capital"], cfg["us"]["capital"]
    s, market = {}, {}
    for k, name in (("A2|main", "A2 RSI2"), ("BO|main", "B-O 오닐"), ("BM|main", "B-M 미너비니"), ("C|main", "C 변동성 돌파")):
        s[name] = C.from_equity(kr[k].dropna(), capk)
        market[name] = "KR"
    for k, name in (("U1|TQQQ|40|0.1", "U1 TQQQ"), ("U1|SOXL|40|0.12", "U1 SOXL"), ("U2|TQQQ|0.15", "U2 TQQQ"),
                    ("U2|SOXL|0.15", "U2 SOXL"), ("U3G|12", "U3-G GEM"), ("U3A|TLT", "U3-A 가속")):
        e = us[k].dropna()
        s[name] = C.from_equity(e, capu)
        market[name] = "US"
    b = pd.read_csv(BASE / "bnf_backtest/results/deep/equity.csv", parse_dates=["date"], index_col="date")["equity"]
    s["bnf 동반급락(기존)"] = b.pct_change().dropna()
    market["bnf 동반급락(기존)"] = "KR"
    it = pd.read_parquet(BASE / "index_timing_backtest/results/daily.parquet")
    s["미국 T3 변동성 목표(기존)"] = it["US|T3|0.15|"].dropna()
    market["미국 T3 변동성 목표(기존)"] = "US"
    aa = pd.read_parquet(BASE / "asset_alloc_backtest/results/daily.parquet")
    s["A1 이중 모멘텀(기존)"] = aa["A1|12|0.0|net"].dropna()
    s["A2 10개월선 5자산(기존)"] = aa["A2|10|0.0|net"].dropna()
    market["A1 이중 모멘텀(기존)"] = market["A2 10개월선 5자산(기존)"] = "US"
    k = load_index(cfg, "KOSPI")["close"]
    s["코스피 총수익"] = k.pct_change().dropna() + (1 + cfg["kr"]["kospi_dividend"]) ** (1 / 252) - 1
    market["코스피 총수익"] = "KR"
    s["SPY"] = D.prices(cfg, "SPY")["close"].pct_change().dropna()
    market["SPY"] = "US"
    start = pd.Timestamp(cfg["compare"]["start"])
    return {n: x[x.index >= start] for n, x in s.items()}, market


def monthly(d: pd.Series) -> pd.Series:
    return (1 + d).groupby(d.index.to_period("M")).prod() - 1


def regimes(cfg, idx_close: pd.Series) -> pd.Series:
    ma = idx_close.rolling(cfg["indicators"]["index_ma_window"]).mean()
    vol = idx_close.pct_change().rolling(cfg["compare"]["vol_window"]).std()
    me = idx_close.groupby(idx_close.index.to_period("M")).tail(1).index
    up = (idx_close > ma).loc[me]
    v = vol.loc[me]
    v = v[v.index >= pd.Timestamp(cfg["compare"]["start"]) - pd.Timedelta(days=40)]
    hi = v > v.median()
    lab = np.where(up.reindex(v.index), "상승", "하락")
    lab = pd.Series([f"{a}·{'고변동' if b else '저변동'}" for a, b in zip(lab, hi)], index=v.index.to_period("M"))
    return lab.shift(1).dropna()        # the regime known at the start of each month


def plots(cfg, s: dict, mret: pd.DataFrame, corr: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 6))
    for n, d in s.items():
        ax.plot((1 + d).cumprod(), label=n, lw=1)
    ax.set_yscale("log")
    ax.set_title(f"누적 수익 ({cfg['compare']['start']} → 끝, 로그 눈금)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "equity_compare.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)   # positive red, negative blue
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im)
    ax.set_title("월간 수익률 상관")
    fig.tight_layout()
    fig.savefig(out / "correlation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir(OUT)
    s, market = series(cfg)
    names = list(s)
    # 1 ranking
    te = pd.Timestamp(cfg["data"]["train_end"])
    rows = []
    for n, d in s.items():
        tr, ts = d[d.index <= te], d[d.index > te]
        rows.append({"series": n, "market": market[n], "first_day": d.index[0].date(), "last_day": d.index[-1].date(),
                     "cagr": C.cagr(d), "mdd": C.mdd(d), "sharpe": C.sharpe(d),
                     "cagr_train": C.cagr(tr) if len(tr) > 20 else np.nan, "cagr_test": C.cagr(ts) if len(ts) > 20 else np.nan})
    rank = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    rank.to_csv(out / "ranking.csv", index=False)
    gs = pd.read_csv(C.results_dir("kr") / "grid_summary.csv")
    best = gs[(gs["market"] == "ALL") & (gs["trades"] >= 200)].sort_values("mean", ascending=False).groupby("strategy").head(1)
    best.to_csv(out / "best_expectancy.csv", index=False)
    # 2 correlation
    mret = pd.DataFrame({n: monthly(d) for n, d in s.items()})
    corr = mret.corr(min_periods=24)
    corr.to_csv(out / "correlation.csv")
    # 3 regimes
    idx = {"KR": load_index(cfg, "KOSPI")["close"], "US": D.prices(cfg, "SPY")["close"]}
    reg = {m: regimes(cfg, c) for m, c in idx.items()}
    rrows = []
    for n in names:
        r = reg[market[n]].reindex(mret.index)
        g = mret[n].groupby(r).mean()
        rrows.append({"series": n, **g.to_dict()})
    rt = pd.DataFrame(rrows)
    rt.to_csv(out / "regimes.csv", index=False)
    # 4 mixes (strategies only, not the two benchmarks)
    strat = [n for n in names if n not in ("코스피 총수익", "SPY")]
    mm = mret[strat].dropna(how="any")
    mixes = []
    for k in (2, 3):
        for combo in itertools.combinations(strat, k):
            cc = mm[list(combo)]
            pairs = [mm[a].corr(mm[b]) for a, b in itertools.combinations(combo, 2)]
            mix = cc.mean(axis=1)
            eq = (1 + mix).cumprod()
            mixes.append({"k": k, "series": " + ".join(combo), "mean_corr": float(np.mean(pairs)),
                          "mix_cagr": float(eq.iloc[-1] ** (12 / len(mix)) - 1),
                          "mix_mdd": float((eq / eq.cummax() - 1).min()),
                          "parts_mdd_mean": float(np.mean([((1 + cc[c]).cumprod() / (1 + cc[c]).cumprod().cummax() - 1).min() for c in combo])),
                          "months": len(mix), "first": str(mm.index[0]), "last": str(mm.index[-1])})
    mx = pd.DataFrame(mixes).sort_values(["k", "mean_corr"]).groupby("k").head(5)
    mx.to_csv(out / "mixes.csv", index=False)
    # 5 robustness
    rk = rank.dropna(subset=["cagr_train", "cagr_test"])
    rho = rk["cagr_train"].rank().corr(rk["cagr_test"].rank())
    g = pd.read_parquet(C.results_dir("kr") / "grid_trades.parquet", columns=["strategy", "combo", "tid", "entry_i", "exit_px", "entry_px", "exit_i", "reason"])
    from src.kr.evaluate import net_ret, slip
    from src.kr.kernel import BRK
    from src.kr.portfolio import Panel
    P = Panel(in_ram=False)
    g = g[g["reason"] != BRK].copy()
    g["strategy"] = g["strategy"].astype(str)
    g["ret"] = 0.0
    for st in ("A2", "B", "C"):
        m = g["strategy"] == st
        g.loc[m, "ret"] = net_ret(cfg, g[m], P.cal, slip(cfg, "C" if st == "C" else "A2"))
    ti = P.cal.searchsorted(te, side="right")
    g["half"] = np.where(g["entry_i"] < ti, "train", "test")
    rob = []
    for st, x in g.groupby("strategy"):
        a = x.groupby(["combo", "half"])["ret"].agg(["mean", "size"]).unstack()
        a = a[(a[("size", "train")] >= 30) & (a[("size", "test")] >= 30)]
        r = a[("mean", "train")].rank().corr(a[("mean", "test")].rank())
        top = a[("mean", "train")].nlargest(10).index
        stay = float((a.loc[top, ("mean", "test")] >= a[("mean", "test")].quantile(0.75)).mean())
        rob.append({"strategy": st, "combos": len(a), "rank_corr": r, "top10_train_in_test_top_quartile": stay,
                    "test_mean_of_train_top10": float(a.loc[top, ("mean", "test")].mean())})
    rb = pd.DataFrame(rob)
    rb["series_cagr_rank_corr"] = rho
    rb.to_csv(out / "robustness.csv", index=False)
    plots(cfg, s, mret, corr, out)
    print(rank.round(3).to_string(index=False))
    print(rt.round(4).to_string(index=False))
    print(mx.round(3).to_string(index=False))
    print(rb.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
