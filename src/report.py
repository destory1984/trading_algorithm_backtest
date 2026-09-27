"""Stage 6: heatmaps, stage-2 analyses and results/report.md.

Needs results/trades.parquet, grid_summary.csv (python -m src.grid),
portfolio_*.csv (python -m src.portfolio) and checks.md (python -m src.checks).

Usage:  python -m src.report
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .common import load_calendar, load_config, load_universe, results_dir  # noqa: E402
from .grid import _metrics, combos, with_all_market  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# Korean convention: gain red, loss blue, neutral gray midpoint
RED, BLUE, MID = "#c8323c", "#2f5fb3", "#eeeeee"
CMAP = LinearSegmentedColormap.from_list("krx", [BLUE, MID, RED])


def stop_label(s) -> str:
    return "없음" if pd.isna(s) else f"{s:.0%}"


def filt_label(v) -> str:
    if v == "none":
        return "없음"
    if v == "uptrend":
        return "상승장만"
    return f"동반급락≤{v[5:]}" if v[5:] else "동반급락"


def pct(x, d=2) -> str:
    return "" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def combo_name(r) -> str:
    return (f"{r['market']} / 이격도≤{r['threshold']} / 손절 {stop_label(r['stop'])} / "
            f"보유 {int(r['hold'])}일 / 시장필터 {filt_label(r['market_filter'])}")


# ---------------------------------------------------------------- heatmaps
def heatmaps(cfg: dict, s: pd.DataFrame) -> list[str]:
    rd, g = results_dir(), cfg["grid"]
    min_n = cfg["stage2"]["min_trades"]
    lim = s.loc[s["trades"] >= min_n, "expectancy"].abs().quantile(0.98) * 100 or 1.0
    files = []
    for m in g["markets"]:
        for rf in g["market_filter"]:
            fig, axes = plt.subplots(1, len(g["stops"]), figsize=(4.2 * len(g["stops"]), 4.6), sharey=True)
            for ax, st in zip(axes, g["stops"]):
                sub = s[(s["market"] == m) & (s["market_filter"] == rf)
                        & ((s["stop"].isna()) if st is None else np.isclose(s["stop"], st))]
                ev = sub.pivot(index="threshold", columns="hold", values="expectancy").reindex(
                    index=g["thresholds"], columns=g["holds"]) * 100
                n = sub.pivot(index="threshold", columns="hold", values="trades").reindex(
                    index=g["thresholds"], columns=g["holds"])
                ax.imshow(ev.values, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
                for i in range(ev.shape[0]):
                    for j in range(ev.shape[1]):
                        v, k = ev.values[i, j], n.values[i, j]
                        if np.isnan(v):
                            continue
                        star = "*" if k < min_n else ""
                        ax.text(j, i, f"{v:+.2f}%{star}\n(n={int(k):,})", ha="center", va="center", fontsize=8,
                                color="white" if abs(v) > 0.55 * lim else "#1a1a1a")
                ax.set_xticks(range(len(g["holds"])), [f"{h}일" for h in g["holds"]])
                ax.set_yticks(range(len(g["thresholds"])), [f"≤{t}" for t in g["thresholds"]])
                ax.set_title(f"손절 {stop_label(st)}", fontsize=10)
                ax.set_xlabel("최대 보유일")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            axes[0].set_ylabel("이격도 기준값")
            fig.suptitle(f"{m} · 시장필터 {filt_label(rf)} · 거래당 기대값 (비용 포함, 빨강=이익 파랑=손실, "
                         f"* 는 거래 {min_n}건 미만)", fontsize=11)
            fig.tight_layout()
            name = f"heatmap_{m}_{rf}.png"
            fig.savefig(rd / name, dpi=130)
            plt.close(fig)
            files.append(name)
    return files


def equity_chart(cfg: dict) -> str | None:
    rd = results_dir()
    p = rd / "portfolio_equity.csv"
    if not p.exists():
        return None
    eq = pd.read_csv(p, index_col=0, parse_dates=True) / cfg["stage2"]["initial_capital"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#c8323c", "#e07b39", "#7a4fb3", "#2a8c6a", "#b3478a"]
    for i, c in enumerate([c for c in eq.columns if c != "KOSPI"]):
        ax.plot(eq.index, eq[c], lw=1.6, color=colors[i % len(colors)], label=c)
    ax.plot(eq.index, eq["KOSPI"], lw=2, color="#555555", label="KOSPI 지수 보유")
    ax.axhline(1, color="#bbbbbb", lw=0.8)
    ax.set_ylabel("자본 (초기 = 1)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.set_title("상위 조합 포트폴리오 (1억 원, 최대 5종목 균등)", fontsize=11)
    fig.tight_layout()
    fig.savefig(rd / "portfolio_equity.png", dpi=130)
    plt.close(fig)
    return "portfolio_equity.png"


# ---------------------------------------------------------------- analyses
def neighbours_ev(s: pd.DataFrame, r, cfg: dict) -> float:
    g = cfg["grid"]
    axes = {"threshold": g["thresholds"], "hold": g["holds"],
            "stop": [np.nan if x is None else x for x in g["stops"]]}
    base = s[(s["market"] == r["market"]) & (s["market_filter"] == r["market_filter"])]
    vals = []
    for ax, lst in axes.items():
        cur = r[ax]
        idx = next(i for i, v in enumerate(lst) if (pd.isna(v) and pd.isna(cur)) or v == cur)
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(lst):
                want = {k: r[k] for k in axes} | {ax: lst[j]}
                m = base
                for k, v in want.items():
                    m = m[m[k].isna()] if pd.isna(v) else m[np.isclose(m[k], v)]
                vals += m["expectancy"].tolist()
    return float(np.mean(vals)) if vals else np.nan


def period_metrics(tr: pd.DataFrame, a: str | None, b: str | None) -> pd.DataFrame:
    t = tr
    if a:
        t = t[t["entry_date"] >= pd.Timestamp(a)]
    if b:
        t = t[t["entry_date"] <= pd.Timestamp(b)]
    t = t[["combo", "market", "ret", "rebound", "hold_days"]].copy()
    t["win"] = t["ret"] > 0
    t["pos"] = t["ret"].where(t["win"])
    t["neg"] = t["ret"].where(~t["win"])
    return _metrics(with_all_market(t).groupby(["combo", "market"]))


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    s = pd.read_csv(rd / "grid_summary.csv")
    tr = pd.read_parquet(rd / "trades.parquet")
    cmb = combos(cfg)
    st2, min_n = cfg["stage2"], cfg["stage2"]["min_trades"]
    ranked = s[s["trades"] >= min_n].sort_values("expectancy", ascending=False)
    cal = load_calendar(cfg)
    uni = load_universe(cfg)
    usable = uni[uni["excluded"].isna()]
    last_snap = usable["snapshot"].max()
    n_delisted = (usable.groupby("ticker")["snapshot"].max() < last_snap).sum()
    src = ", ".join(sorted(uni["source"].unique()))

    out = ["# BNF 25일선 이격도 매매 백테스트 결과", ""]
    out += [
        "## 데이터와 가정", "",
        f"- 기간: {cfg['data']['start']} → {cal[-1].date()} (신호 기준). 거래일 {int((cal >= pd.Timestamp(cfg['data']['start'])).sum()):,}일.",
        f"- 종목: 코스피·코스닥 {usable['ticker'].nunique():,}개. 그중 기간 중 상장폐지 {n_delisted}개 포함. 목록 출처: {src}.",
        f"- 전체 거래 기록 {len(tr):,}건 ({len(cmb)}개 조합 × 종목, 시장 구분은 같은 거래를 나눠 본 것).",
        f"- 비용: 매수·매도 수수료 각 {cfg['costs']['buy_fee']:.3%}, 매도세 {cfg['costs']['sell_tax']:.2%}, "
        f"슬리피지 편도 {cfg['costs']['slippage']:.1%}. 왕복 약 "
        f"{(1 - (1 - cfg['costs']['slippage']) * (1 - cfg['costs']['sell_fee'] - cfg['costs']['sell_tax']) / ((1 + cfg['costs']['slippage']) * (1 + cfg['costs']['buy_fee']))):.2%}.",
        "- 진입은 신호 다음 날 시가. 보유일은 진입일을 1일째로 센다.",
        f"- 순위표는 거래가 {min_n}건 이상인 조합만 넣었다.",
        "- 기대값 = 승률 × 평균 이익 + (1 − 승률) × 평균 손실. 거래당 평균 수익률과 같은 값이다.",
        "",
    ]
    if "fdr_current" in src and "krx" not in src:
        out += ["- 상장폐지 종목 목록은 KRX KIND 공시에서 받았다. KIND 목록에 없는 폐지 종목(주로 이름 없는 합병·이전)은 빠졌을 수 있다.", ""]

    # top 10
    top = ranked.head(cfg["report"]["top_n"]).copy()
    t10 = pd.DataFrame({
        "순위": range(1, len(top) + 1),
        "조합": [combo_name(r) for _, r in top.iterrows()],
        "거래 수": top["trades"].map("{:,}".format),
        "연평균 거래": top["trades_per_year"].round(0).astype(int),
        "반등 확률": top["rebound_rate"].map(lambda x: f"{x:.1%}"),
        "승률": top["win_rate"].map(lambda x: f"{x:.1%}"),
        "기대값": top["expectancy"].map(pct),
        "중앙값": top["median_ret"].map(pct),
        "손익비": top["payoff"].round(2),
        "평균 보유일": top["avg_hold"].round(1),
        "최악": top["worst"].map(lambda x: pct(x, 1)),
        "주변 조합 기대값": [pct(neighbours_ev(s, r, cfg)) for _, r in top.iterrows()],
    })
    out += [f"## 기대값 상위 {len(top)}개 조합", "", md_table(t10), "",
            "주변 조합 기대값은 이격도·보유일·손절 가운데 하나만 한 칸 옮긴 조합들의 평균이다. "
            "이 값이 순위 값과 크게 다르면 그 조합은 우연히 좋았을 가능성이 크다.", ""]

    # rebound by threshold
    rb = s[(s["market"] == "ALL") & (s["market_filter"] == "none") & (s["stop"].isna())]
    rbt = rb.pivot(index="threshold", columns="hold", values="rebound_rate").sort_index(ascending=False)
    evt = rb.pivot(index="threshold", columns="hold", values="expectancy").sort_index(ascending=False)
    nt = rb.pivot(index="threshold", columns="hold", values="signals").sort_index(ascending=False).iloc[:, 0]
    tbl = pd.DataFrame({"이격도 기준": [f"≤{t}" for t in rbt.index],
                        "신호 수": nt.map("{:,}".format).values,
                        "연평균 신호": (nt / (s["signals"] / s["signals_per_year"]).dropna().iloc[0]).round(0).astype(int).values})
    for h in rbt.columns:
        tbl[f"반등 {h}일"] = rbt[h].map(lambda x: f"{x:.1%}").values
    for h in evt.columns:
        tbl[f"기대값 {h}일"] = evt[h].map(pct).values
    out += ["## 이격도 기준값별 반등 확률 (합산, 손절 없음, 시장필터 없음)", "",
            "반등 = 최대 보유일 안에 종가가 25일선 이상으로 올라온 비율.", "", md_table(tbl), ""]

    # market filters
    piv = s.pivot_table(index=["market", "threshold", "stop", "hold"], columns="market_filter",
                        values=["expectancy", "trades", "worst"])
    rows = []
    for mf in [f for f in cfg["grid"]["market_filter"] if f != "none"]:
        ok = piv[(piv[("trades", "none")] >= min_n) & (piv[("trades", mf)] >= min_n)]
        for m in cfg["grid"]["markets"]:
            b = ok.xs(m, level="market")
            rows.append({"필터": filt_label(mf), "시장": m, "비교한 조합": len(b),
                         "필터 쪽이 나은 조합": int((b[("expectancy", mf)] > b[("expectancy", "none")]).sum()),
                         "평균 기대값 없음": pct(b[("expectancy", "none")].mean()),
                         "평균 기대값 필터": pct(b[("expectancy", mf)].mean()),
                         "거래 수 비율 (필터/없음)": f"{(b[('trades', mf)].sum() / b[('trades', 'none')].sum()):.0%}"})
    out += ["## 시장 필터 비교", "",
            f"- 상승장만: 신호일 지수 종가가 지수 120일선 위일 때만 산다.",
            "- 동반급락≤N: 신호일 지수 이격도(25일선)가 N 이하일 때만 산다. 종목 혼자 빠진 날은 거른다.", "",
            f"필터 없음과 필터 적용 모두 거래 {min_n}건 이상인 조합끼리 짝지어 비교했다.", "",
            md_table(pd.DataFrame(rows)), ""]
    fx = s[(s["stop"].isna()) & (s["hold"] == 10) & (s["market"] == "ALL")]
    fxp = fx.pivot(index="threshold", columns="market_filter", values=["expectancy", "trades"]).sort_index(ascending=False)
    det = pd.DataFrame({"이격도 기준": [f"≤{t}" for t in fxp.index]})
    for mf in cfg["grid"]["market_filter"]:
        det[f"기대값 {filt_label(mf)}"] = fxp[("expectancy", mf)].map(pct).values
    for mf in cfg["grid"]["market_filter"]:
        det[f"거래 {filt_label(mf)}"] = fxp[("trades", mf)].map("{:,.0f}".format).values
    out += ["합산 · 손절 없음 · 보유 10일 기준 세부:", "", md_table(det), ""]
    cr = ranked[ranked["market_filter"].str.startswith("crash")].head(cfg["report"]["top_n"])
    out += [f"동반급락 필터 조합 가운데 기대값 상위 {len(cr)}개:", "", md_table(pd.DataFrame({
        "조합": [combo_name(r) for _, r in cr.iterrows()],
        "거래 수": cr["trades"].map("{:,}".format), "연평균 거래": cr["trades_per_year"].round(0).astype(int),
        "승률": cr["win_rate"].map(lambda x: f"{x:.1%}"), "기대값": cr["expectancy"].map(pct),
        "중앙값": cr["median_ret"].map(pct), "최악": cr["worst"].map(lambda x: pct(x, 1))})), ""]

    # yearly
    top5 = ranked.head(st2["top_n"])
    ycols = sorted(c for c in s.columns if c.startswith("ev_") and c[3:].isdigit())
    yt = pd.DataFrame({"연도": [c[3:] for c in ycols]})
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        yt[f"#{i}"] = [pct(r[c]) for c in ycols]
    out += ["## 연도별 기대값 (상위 5개 조합, 진입 연도 기준)", "",
            "\n".join(f"- #{i}: {combo_name(r)}" for i, (_, r) in enumerate(top5.iterrows(), 1)), "",
            md_table(yt), ""]
    ct = []
    for label in st2["crash_periods"]:
        row = {"구간": label}
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            n = r.get(f"n_{label}")
            row[f"#{i}"] = f"{pct(r.get(f'ev_{label}'))} ({0 if pd.isna(n) else int(n)}건)"
        ct.append(row)
    out += ["급락 구간만 따로:", "", md_table(pd.DataFrame(ct)), ""]
    ex = []
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        t = tr[tr["combo"] == r["combo"]]
        if r["market"] != "ALL":
            t = t[t["market"] == r["market"]]
        m = pd.Series(False, index=t.index)
        for a, b in st2["crash_periods"].values():
            m |= t["entry_date"].between(pd.Timestamp(a), pd.Timestamp(b))
        x = t[~m]["ret"]
        ex.append({"조합": f"#{i}", "전체 거래": f"{len(t):,}", "급락 구간 거래 비중": f"{m.mean():.0%}",
                   "전체 기대값": pct(t["ret"].mean()), "급락 구간 뺀 기대값": pct(x.mean()),
                   "급락 구간 뺀 중앙값": pct(x.median()), "급락 구간 뺀 승률": f"{(x > 0).mean():.1%}"})
    out += ["급락 구간(위 표의 네 구간)에 진입한 거래를 모두 뺀 성과:", "", md_table(pd.DataFrame(ex)), ""]

    # co-crash split
    thr = cfg["indicators"]["index_disparity_crash"]
    rows = []
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        t = tr[tr["combo"] == r["combo"]]
        if r["market"] != "ALL":
            t = t[t["market"] == r["market"]]
        for lab, m in ((f"시장 동반 급락 (지수 이격도 ≤{thr})", t["idx_disp"] <= thr),
                       (f"개별 급락 (지수 이격도 >{thr})", t["idx_disp"] > thr)):
            x = t[m]["ret"]
            rows.append({"조합": f"#{i}", "구분": lab, "거래 수": f"{len(x):,}",
                         "승률": f"{(x > 0).mean():.1%}" if len(x) else "",
                         "기대값": pct(x.mean()), "중앙값": pct(x.median()), "최악": pct(x.min(), 1)})
    out += ["## 시장 동반 급락과 개별 급락", "",
            "신호일의 해당 시장 지수 이격도(25일선)로 나눴다. 개별 급락은 그 종목만의 악재일 가능성이 크다.", "",
            md_table(pd.DataFrame(rows)), ""]

    # portfolio
    pp = rd / "portfolio_summary.csv"
    if pp.exists():
        ps = pd.read_csv(pp)
        lab = {(int(r["combo"]), r["market"]): f"#{i}" for i, (_, r) in enumerate(top5.iterrows(), 1)}
        pt = pd.DataFrame({
            "조합": [lab.get((int(c), m), "?") if c >= 0 else "KOSPI 지수" for c, m in zip(ps["combo"], ps["market"])],
            "시장": ps["market"],
            "누적 수익률": ps["total_return"].map(lambda x: f"{x:+.1%}"),
            "연환산": ps["cagr"].map(lambda x: f"{x:+.1%}"),
            "MDD": ps["mdd"].map(lambda x: f"{x:.1%}"),
            "체결한 거래": ps.get("trades_taken", pd.Series(dtype=float)).map(lambda x: "" if pd.isna(x) else f"{int(x):,}"),
            "신호 거래": ps.get("trades_available", pd.Series(dtype=float)).map(lambda x: "" if pd.isna(x) else f"{int(x):,}"),
            "보유 중인 날 비율": ps.get("invested_share", pd.Series(dtype=float)).map(lambda x: "" if pd.isna(x) else f"{x:.0%}"),
            "체결 거래 평균": ps.get("taken_mean_ret", pd.Series(dtype=float)).map(pct),
            "전체 신호 거래 평균": ps.get("all_mean_ret", pd.Series(dtype=float)).map(pct),
        })
        out += ["## 포트폴리오 시뮬레이션 (상위 5개 조합)", "",
                f"초기 자본 {st2['initial_capital'] / 1e8:.0f}억 원, 동시 보유 최대 {st2['max_positions']}종목, "
                "새 종목에 전날 평가금액의 1/5 을 넣는다. 같은 날 신호가 빈 자리보다 많으면 이격도가 낮은 종목부터 산다. "
                "청산한 자리는 다음 날부터 쓴다.", "", md_table(pt), "", "![자본 곡선](portfolio_equity.png)", ""]

    pf = rd / "portfolio_filters.csv"
    if pf.exists() and st2.get("compare_combo"):
        cc = st2["compare_combo"]
        f = pd.read_csv(pf)
        out += [f"같은 조합({cc['market']} / 이격도≤{cc['threshold']} / 손절 {stop_label(cc.get('stop'))} / "
                f"보유 {cc['hold']}일)을 시장 필터만 바꿔 돌린 결과:", "", md_table(pd.DataFrame({
                    "시장 필터": [filt_label(x) if x != "KOSPI buy&hold" else "KOSPI 지수 보유" for x in f["filter"]],
                    "연환산": f["cagr"].map(lambda x: f"{x:+.1%}"),
                    "MDD": f["mdd"].map(lambda x: f"{x:.1%}"),
                    f"연환산 ~{st2['train_end'][:4]}": f["cagr_train"].map(lambda x: f"{x:+.1%}"),
                    f"연환산 {int(st2['train_end'][:4]) + 1}~": f["cagr_test"].map(lambda x: f"{x:+.1%}"),
                    "체결한 거래": f.get("trades_taken", pd.Series(dtype=float)).map(lambda x: "" if pd.isna(x) else f"{int(x):,}"),
                    "보유 중인 날 비율": f.get("invested_share", pd.Series(dtype=float)).map(lambda x: "" if pd.isna(x) else f"{x:.0%}"),
                })), ""]

    # train / test
    te = st2["train_end"]
    a = period_metrics(tr, None, te)
    b = period_metrics(tr, str((pd.Timestamp(te) + pd.Timedelta(days=1)).date()), None)
    j = a[["expectancy", "trades"]].join(b[["expectancy", "trades"]], lsuffix="_train", rsuffix="_test").dropna()
    j = j[(j["trades_train"] >= min_n) & (j["trades_test"] >= min_n)]
    j["rank_train"] = j["expectancy_train"].rank(ascending=False).astype(int)
    j["rank_test"] = j["expectancy_test"].rank(ascending=False).astype(int)
    rho = j["rank_train"].corr(j["rank_test"])  # Spearman = Pearson on ranks (no scipy needed)
    jt = j.sort_values("rank_train").head(10).reset_index().join(cmb, on="combo")
    out += ["## 학습 · 검증 기간 비교", "",
            f"학습 {cfg['data']['start']} → {te}, 검증 그다음 날 → 최근. 두 기간 모두 거래 {min_n}건 이상인 조합 {len(j)}개.",
            f"두 기간 기대값 순위의 스피어만 상관계수: {rho:.2f} (1 이면 순위가 그대로, 0 이면 무관).", "",
            md_table(pd.DataFrame({
                "학습 순위": jt["rank_train"], "검증 순위": jt["rank_test"],
                "조합": [combo_name(r) for _, r in jt.iterrows()],
                "학습 기대값": jt["expectancy_train"].map(pct), "검증 기대값": jt["expectancy_test"].map(pct),
                "학습 거래": jt["trades_train"].map("{:,.0f}".format), "검증 거래": jt["trades_test"].map("{:,.0f}".format)})), ""]

    # heatmaps
    files = heatmaps(cfg, s)
    equity_chart(cfg)
    out += ["## 히트맵 (이격도 기준값 × 최대 보유일, 기대값)", ""]
    out += [f"![{f}]({f})" for f in files] + [""]

    ck = rd / "checks.md"
    if ck.exists():
        out += [ck.read_text(encoding="utf-8").replace("\n## ", "\n### ").replace("# 검증 체크리스트 결과", "## 검증 체크리스트"), ""]
    out += ["## 읽을 때 주의할 점", "",
            "- 과거 성과가 미래 수익을 보장하지 않는다.",
            f"- {len(cmb) * len(cfg['grid']['markets'])}개 조합 중 1등을 고르면 운이 좋았던 조합을 고를 위험이 크다. 주변 조합 기대값과 학습·검증 비교를 같이 본다.",
            "- 소형주는 호가가 비어 있어 실제 체결이 슬리피지 가정(편도 0.1%)보다 나쁠 수 있다.",
            "- 거래대금은 종가 × 거래량으로 어림했다.", ""]
    (rd / "report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"report.md written, {len(files)} heatmaps")


if __name__ == "__main__":
    main()
