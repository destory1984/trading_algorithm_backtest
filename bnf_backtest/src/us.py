"""The same disparity grid and portfolio on US stocks (config `us`).

Universe: Nasdaq-100 + Dow 30 + watchlist from krxbt.fetch_us. Only today's
members exist in the data, so results are survivorship biased. Stocks and
ETFs (leveraged ETFs from the watchlist) are summarised apart.

Writes results/us/: trades.parquet, grid_summary.csv, portfolio.csv,
portfolio_equity.csv, report.md.

Usage:  python -m src.us
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from krxbt import us
from krxbt import portfolio as kp
from krxbt.portfolio import stats

from .common import load_config, results_dir
from .grid import _metrics, combos
from .portfolio import filter_label
from .report import md_table, pct, stop_label
from .simulate import arrays, simulate


def us_config() -> dict:
    cfg = load_config()
    u = copy.deepcopy(cfg)
    u["costs"] = cfg["us"]["costs"]
    return u


def frames(cfg: dict):
    cal = us.load_calendar(cfg)
    idx = us.index_features(cfg)
    uni = us.load_universe(cfg)
    for t, row in uni.iterrows():
        f = us.ticker_frame(cfg, t, cal, idx)
        if f is not None:
            yield t, ("etf" if row.get("type") == "ETF" else "stock"), f


def run_grid(cfg: dict, fr: list) -> pd.DataFrame:
    cmb = combos(cfg)
    out = []
    for t, kind, f in fr:
        a = arrays(f)
        for cid, c in cmb.iterrows():
            stop = None if pd.isna(c["stop"]) else float(c["stop"])
            tr = simulate(f, cfg, c["threshold"], stop, int(c["hold"]), c["market_filter"], a)
            if len(tr):
                tr.insert(0, "combo", np.int16(cid))
                tr["ticker"], tr["kind"] = t, kind
                out.append(tr)
    return pd.concat(out, ignore_index=True).drop(columns="data_break")


def summarize(cfg: dict, tr: pd.DataFrame) -> pd.DataFrame:
    df = tr[["combo", "kind", "ret", "rebound", "hold_days", "entry_date"]].copy()
    df["win"] = df["ret"] > 0
    df["pos"] = df["ret"].where(df["win"])
    df["neg"] = df["ret"].where(~df["win"])
    df = pd.concat([df, df.assign(kind="ALL")], ignore_index=True)
    s = _metrics(df.groupby(["combo", "kind"]))
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    for name, m in (("train", df["entry_date"] <= te), ("test", df["entry_date"] > te)):
        g = df[m].groupby(["combo", "kind"])["ret"]
        s[f"ev_{name}"], s[f"n_{name}"] = g.mean(), g.count()
    s = s.reset_index().join(combos(cfg), on="combo")
    return s.sort_values("expectancy", ascending=False)


def portfolios(cfg: dict, fr: list, pick: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = us.load_calendar(cfg)
    cal = cal[cal >= pd.Timestamp(cfg["data"]["start"])]
    u = cfg["us"]
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    closes = pd.DataFrame({t: f["close"] for t, _, f in fr}).reindex(cal).ffill()
    rows, curves = [], {}
    for _, c in pick.iterrows():
        stop = None if pd.isna(c["stop"]) else float(c["stop"])
        parts = []
        for t, kind, f in fr:
            if c["kind"] != "ALL" and kind != c["kind"]:
                continue
            x = simulate(f, cfg, c["threshold"], stop, int(c["hold"]), c["market_filter"], independent=True)
            if len(x):
                x["ticker"] = t
                parts.append(x)
        tr = pd.concat(parts, ignore_index=True)
        eq, info = kp.run_portfolio(cal, tr, closes, u["initial_capital"], cfg["stage2"]["max_positions"], "disp")
        name = (f"{'주식' if c['kind'] == 'stock' else '주식+ETF'} ≤{c['threshold']} {stop_label(c['stop'])} "
                f"{int(c['hold'])}일 {filter_label(c['market_filter'])}")
        curves[name] = eq
        rows.append({"조합": name, **stats(eq), "cagr_train": stats(eq[eq.index <= te])["cagr"],
                     "cagr_test": stats(eq[eq.index > te])["cagr"], **info})
    for nm in ("NDX", "SPX"):
        ix = us.load_index(cfg, nm)["close"].reindex(cal).ffill()
        ix = ix / ix.iloc[0] * u["initial_capital"]
        curves[f"{nm} 지수"] = ix
        rows.append({"조합": f"{nm} 지수 보유 (배당 제외)", **stats(ix), "cagr_train": stats(ix[ix.index <= te])["cagr"],
                     "cagr_test": stats(ix[ix.index > te])["cagr"]})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def report(cfg: dict, s: pd.DataFrame, tr: pd.DataFrame, pf: pd.DataFrame, n_stock: int, n_etf: int) -> str:
    u = cfg["us"]
    yr = int(cfg["stage2"]["train_end"][:4])
    out = ["# 미국 주식 이격도 백테스트", "",
           f"종목: 나스닥100 + 다우30 + 관심 종목. 주식 {n_stock}개, ETF {n_etf}개(레버리지 포함). "
           f"기간 {cfg['data']['start']} → {tr['exit_date'].max().date()}. 가격은 배당 반영.",
           f"시장 필터 지수: {u['market_index']}. 비용: 수수료 편도 {u['costs']['buy_fee']:.2%}, 슬리피지 편도 {u['costs']['slippage']:.2%}, 세금 없음.",
           "**지금 지수에 든 종목만 있어서 생존편향이 있다.** 그동안 빠지거나 망한 종목의 급락은 빠져 있다.", ""]

    # 1. threshold x filter, stocks, no stop, 20 days
    base = s[(s["kind"] == "stock") & s["stop"].isna() & (s["hold"] == 20)]
    t1 = base.pivot_table(index="threshold", columns="market_filter", values="expectancy").sort_index(ascending=False)
    n1 = base.pivot_table(index="threshold", columns="market_filter", values="trades").sort_index(ascending=False)
    cols = [c for c in ["none", "uptrend", "crash99", "crash97", "crash95"] if c in t1]
    rows = [{"이격도": f"≤{th}", **{filter_label(c): (f"{pct(t1.loc[th, c])} ({int(n1.loc[th, c]):,})"
                                                   if pd.notna(t1.loc[th, c]) else "") for c in cols}} for th in t1.index]
    out += ["## 이격도 기준 × 시장 필터 (주식만, 손절 없음, 20일 보유)", "",
            "칸 = 거래당 기대값 (거래 수).", "", md_table(pd.DataFrame(rows)), ""]

    # 2. top combos
    for kind, label in (("stock", "주식만"), ("etf", "ETF만")):
        x = s[(s["kind"] == kind) & (s["trades"] >= (u["min_trades"] if kind == "stock" else 30))].head(cfg["report"]["top_n"])
        if x.empty:
            continue
        out += [f"## 기대값 상위 조합 ({label}, 거래 {u['min_trades'] if kind == 'stock' else 30}건 이상)", "", md_table(pd.DataFrame({
            "이격도": x["threshold"].map(lambda v: f"≤{v}"), "손절": x["stop"].map(stop_label), "보유": x["hold"].map(lambda v: f"{int(v)}일"),
            "시장 필터": x["market_filter"].map(filter_label), "거래": x["trades"].map("{:,.0f}".format),
            "승률": x["win_rate"].map(lambda v: f"{v:.0%}"), "기대값": x["expectancy"].map(pct),
            "중앙값": x["median_ret"].map(pct), f"{yr}년까지": x["ev_train"].map(pct), f"{yr + 1}년부터": x["ev_test"].map(pct)})), ""]

    # 3. portfolio
    out += [f"## 포트폴리오 ({u['initial_capital']:,}달러, 최대 {cfg['stage2']['max_positions']}종목)", "",
            "한국과 같은 방식: 새 종목에 전날 평가금액의 1/5, 같은 날 신호가 많으면 이격도 낮은 순. 지수는 가격 지수라 배당이 빠져 있다(연 1 → 2%p 불리).", "",
            md_table(pd.DataFrame({
                "조합": pf["조합"], "연환산": pf["cagr"].map(lambda v: f"{v:+.1%}"), "MDD": pf["mdd"].map(lambda v: f"{v:.0%}"),
                f"{yr}년까지": pf["cagr_train"].map(lambda v: f"{v:+.1%}"), f"{yr + 1}년부터": pf["cagr_test"].map(lambda v: f"{v:+.1%}"),
                "체결 거래": pf.get("trades_taken", pd.Series(dtype=float)).map(lambda v: "" if pd.isna(v) else f"{int(v):,}"),
                "보유 중인 날": pf.get("invested_share", pd.Series(dtype=float)).map(lambda v: "" if pd.isna(v) else f"{v:.0%}")})), ""]
    return "\n".join(out)


def main() -> None:
    cfg = us_config()
    rd = results_dir() / "us"
    rd.mkdir(exist_ok=True)
    fr = list(frames(cfg))
    n_etf = sum(k == "etf" for _, k, _ in fr)
    print(f"{len(fr)} tickers ({n_etf} ETF)")
    tr = run_grid(cfg, fr)
    tr.to_parquet(rd / "trades.parquet", index=False)
    s = summarize(cfg, tr)
    s.to_csv(rd / "grid_summary.csv", index=False, encoding="utf-8-sig")
    print(f"trades {len(tr):,}")
    u = cfg["us"]
    top = s[(s["kind"] == "stock") & (s["trades"] >= u["min_trades"])].head(u["portfolio_top_n"])
    pick = pd.concat([top, top.assign(kind="ALL")], ignore_index=True)
    pf, curves = portfolios(cfg, fr, pick)
    pf.to_csv(rd / "portfolio.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(rd / "portfolio_equity.csv")
    (rd / "report.md").write_text(report(cfg, s, tr, pf, len(fr) - n_etf, n_etf), encoding="utf-8")
    print((rd / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
