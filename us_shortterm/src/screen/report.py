"""results/screen/report.md from metrics.csv (and daily.parquet for the correlation table). Run after run.py.

Pass rule per setting (SPEC 10), fixed before the run in config `pass`:
  1+2  at least min_etfs ETFs where the strategy's CAGR (with costs) beats the risk-matched hold AND the ETF has
       >= min_trades trades
  3    mean breakeven cost over the 4 ETFs >= min_breakeven_bp (one way)
The strict reading of 1+2 (every beating ETF must have >= min_trades) is computed too; the report says if the two
readings ever disagree.
Usage: python -m src.screen.report
"""
from __future__ import annotations

import pandas as pd

from src import common
from src.screen.run import label, settings


def judge(met: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = cfg["pass"]
    rows = []
    for (lab, v), g in met.groupby(["label", "version"], sort=False):
        beat = g["cagr"] > g["risk_cagr"]
        enough = g["trades"] >= p["min_trades"]
        n12 = int((beat & enough).sum())
        c12 = n12 >= p["min_etfs"]
        c12_strict = int(beat.sum()) >= p["min_etfs"] and bool(enough[beat].all())
        be = float(g["breakeven_bp"].mean())
        c3 = be >= p["min_breakeven_bp"]
        rows.append({"label": lab, "version": v, "strategy": g["strategy"].iloc[0], "name": g["name"].iloc[0],
                     "beat": int(beat.sum()), "beat_50": n12, "c12": c12, "c12_strict": c12_strict,
                     "mean_be": be, "c3": c3, "pass": c12 and c3, "pass_strict": c12_strict and c3})
    j = pd.DataFrame(rows)
    j["note"] = ""
    for s, g in j.groupby("strategy"):
        if set(g["version"]) == {"close", "open"}:
            cl, op = g[g["version"] == "close"].iloc[0], g[g["version"] == "open"].iloc[0]
            if cl["pass"] and not op["pass"]:
                j.loc[cl.name, "note"] = "종가 체결 필요"
    return j


def verdict(r) -> str:
    if r["pass"]:
        return "통과" + (f" ({r['note']})" if r["note"] else "")
    failed = [k for k, ok in (("1·2", r["c12"]), ("3", r["c3"])) if not ok]
    return "탈락 (기준 " + ", ".join(failed) + ")"


def setting_name(r) -> str:
    return f"{r['strategy']} {r['name']} {r['version']}"


def main() -> None:
    cfg = common.load_config()
    out = common.results_dir()
    met = pd.read_csv(out / "metrics.csv", parse_dates=["first_day", "last_day"])
    tick = list(cfg["screen"]["tickers"])
    order = [(label(m), v) for m, v in settings()]
    met["_o"] = [order.index((a, b)) for a, b in zip(met["label"], met["version"])]
    met["_t"] = met["ticker"].map(tick.index)
    met = met.sort_values(["_o", "_t"])
    j = judge(met, cfg)
    j["_o"] = [order.index((a, b)) for a, b in zip(j["label"], j["version"])]
    j = j.sort_values("_o").reset_index(drop=True)
    p = cfg["pass"]
    end = common.screen_end(cfg)
    L = ["# 미국 단기 전략 1차 선별 보고서", "",
         f"선별 기간 {met['first_day'].min().date()} → {met['last_day'].max().date()}, ETF {', '.join(tick)}. "
         f"비용 편도 {(cfg['us']['costs']['buy_fee'] + cfg['us']['costs']['slippage']) * 100:.2f}% "
         f"(수수료 {cfg['us']['costs']['buy_fee'] * 100:.2f}% + 슬리피지 {cfg['us']['costs']['slippage'] * 100:.2f}%). "
         "수치는 모두 `metrics.csv` 에서 채웠다.", ""]

    # 1 summary
    rows = []
    for _, r in j.iterrows():
        g = met[(met["label"] == r["label"]) & (met["version"] == r["version"])].set_index("ticker")
        row = {"설정": setting_name(r)}
        for t in tick:
            row[t] = f"{common.pct(g.loc[t, 'cagr'])} / {common.pct(g.loc[t, 'diff_risk'])}"
        row["통과"] = verdict(r)
        rows.append(row)
    L += ["## 1. 요약", "", "각 칸은 \"비용 포함 연 수익률 / 위험을 맞춘 보유 대비 차이\"다.", "",
          common.md_table(pd.DataFrame(rows)), ""]
    rows = []
    for _, r in j.iterrows():
        g = met[(met["label"] == r["label"]) & (met["version"] == r["version"])].set_index("ticker")
        rows.append({"설정": setting_name(r), "위험 맞춘 보유를 이긴 ETF": f"{r['beat']}/4",
                     f"그중 거래 {p['min_trades']}건 이상": r["beat_50"],
                     "거래 수 " + "/".join(tick): "/".join(str(int(g.loc[t, "trades"])) for t in tick),
                     "평균 손익분기 비용(bp)": common.num(r["mean_be"], 1),
                     "기준 1·2": "예" if r["c12"] else "아니오", "기준 3": "예" if r["c3"] else "아니오"})
    L += [f"기준: (1·2) 4개 ETF 중 {p['min_etfs']}개 이상에서 연 수익률이 위험을 맞춘 보유보다 높고 그 ETF 의 거래가 각각 "
          f"{p['min_trades']}건 이상, (3) 4개 ETF 평균 손익분기 비용이 편도 {p['min_breakeven_bp']}bp 이상.", "",
          common.md_table(pd.DataFrame(rows)), ""]
    differ = j[j["pass"] != j["pass_strict"]]
    L += [("기준 1·2 를 \"이긴 ETF 모두가 거래 50건 이상\"으로 엄격하게 읽어도 판정이 같다." if differ.empty else
           "기준 1·2 를 엄격하게 읽으면 판정이 달라지는 설정: " + ", ".join(map(setting_name, [r for _, r in differ.iterrows()])) + "."), ""]

    # 2 details
    passed = j[j["pass"]]
    L += ["## 2. 통과 설정 상세", ""]
    cols = [("trades", "거래", lambda v: str(int(v))), ("win_rate", "승률", common.share),
            ("expectancy", "거래당 기대값", lambda v: common.pct(v, 2)), ("profit_factor", "손익비", common.num),
            ("cagr", "연 수익률", common.pct), ("sharpe", "샤프", common.num), ("mdd", "최대 낙폭", common.pct),
            ("exposure", "노출", common.share), ("cagr_zero_cost", "비용 0 연 수익률", common.pct),
            ("breakeven_bp", "손익분기 비용(bp)", lambda v: common.num(v, 1)),
            ("risk_cagr", "위험 맞춘 보유", common.pct), ("diff_risk", "차이", common.pct),
            ("exp_cagr", "노출 맞춘 보유", common.pct), ("diff_exp", "차이 ", common.pct)]
    if passed.empty:
        L += ["통과한 설정이 없다.", ""]
    for _, r in passed.iterrows():
        g = met[(met["label"] == r["label"]) & (met["version"] == r["version"])]
        t = pd.DataFrame([{"ETF": x["ticker"], **{name: fmt(x[k]) for k, name, fmt in cols}} for _, x in g.iterrows()])
        L += [f"### {setting_name(r)}", "", common.md_table(t), ""]

    # 3 correlation
    L += ["## 3. 통과 설정끼리의 일별 수익률 상관 (SPY)", ""]
    if len(passed) < 2:
        L += [f"통과 설정이 {len(passed)}개라 상관표를 만들지 않는다.", ""]
    else:
        daily = pd.read_parquet(out / "daily.parquet")
        keys = {setting_name(r): f"{r['label']}|{r['version']}|SPY" for _, r in passed.iterrows()}
        c = daily[list(keys.values())].corr()
        c.index = c.columns = list(keys)
        c = c.map(lambda v: f"{v:.2f}").reset_index().rename(columns={"index": ""})
        L += [common.md_table(c), ""]

    # 4 close vs open
    rows = []
    for s in ["S2", "S3", "S4", "S5", "S6", "S7"]:
        g = met[met["strategy"] == s]
        row = {"전략": f"{s} {g['name'].iloc[0]}"}
        for t in tick:
            cl = g[(g["version"] == "close") & (g["ticker"] == t)]["cagr"].iloc[0]
            op = g[(g["version"] == "open") & (g["ticker"] == t)]["cagr"].iloc[0]
            row[t] = f"{common.pct(cl)} / {common.pct(op)} / {common.pct(cl - op)}"
        rows.append(row)
    L += ["## 4. close 판본과 open 판본의 연 수익률 차이", "", "각 칸은 \"close / open / close - open\"이다.", "",
          common.md_table(pd.DataFrame(rows)), ""]

    # 5 S9
    g = met[met["strategy"] == "S9"].set_index("ticker")
    t9 = pd.DataFrame([{"ETF": t, "비용 0 연 수익률": common.pct(g.loc[t, "cagr_zero_cost"]),
                        "비용 포함 연 수익률": common.pct(g.loc[t, "cagr"]), "보유 연 수익률": common.pct(g.loc[t, "hold_cagr"]),
                        "거래": int(g.loc[t, "trades"]), "손익분기 비용(bp)": common.num(g.loc[t, "breakeven_bp"], 1)} for t in tick])
    L += ["## 5. S9 밤사이 보유의 비용 0 연 수익률", "", "종가에 사서 다음 날 시가에 파는 수익만 모은 기준선이다.", "",
          common.md_table(t9), ""]

    # 6 count and seal
    last = met["last_day"].max()
    L += ["## 6. 돌린 설정 수와 봉인 확인", "",
          f"- 돌린 설정: 전략 × 판본 {len(j)}개 × ETF {len(tick)}개 = **{len(met)}개**. 각각 한 번씩만 돌렸다.",
          f"- 가장 늦은 데이터 날짜: {last.date()}. {end.date()} 뒤 데이터는 쓰지 않았다(불러온 직후 잘라내고 단언문으로 확인, "
          "`checks.md` 1장).", ""]

    # 7 conclusion
    if passed.empty:
        near = j.sort_values(["beat_50", "mean_be"], ascending=False).iloc[0]
        concl = (f"18개 설정 중 통과한 설정은 없다. 가장 가까운 설정은 {setting_name(near)} 로, 위험을 맞춘 보유를 "
                 f"{near['beat']}/4 개 ETF 에서 이겼고 평균 손익분기 비용은 {near['mean_be']:.1f}bp 다.")
    else:
        concl = f"18개 설정 중 {len(passed)}개가 통과했다: " + ", ".join(
            setting_name(r) + (f"({r['note']})" if r["note"] else "") for _, r in passed.iterrows()) + "."
    L += ["## 7. 결론", "", concl, ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    j.drop(columns="_o").to_csv(out / "verdicts.csv", index=False)
    print(concl)


if __name__ == "__main__":
    main()
