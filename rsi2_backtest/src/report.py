"""results/report.md (the spec's nine sections) and results/readme_results.md, from the csv files of the other stages.

Every number is read from a csv (or the saved daily parquet) here; none is typed in. Costs, trial counts, years, rule
text and capitals come from config.yaml. The decision rule is computed from the rows; a missing row fails its
criterion.

Usage:  python -m src.report            writes results/report.md and results/readme_results.md
        python -m src.report --readme   also replaces the block between the result markers in README.md
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from .common import ROOT, load_config, md_table, num, one, pct, results_dir, share
from .grid import EXIT, TREND, label
from .walkforward import WF_VARIANTS

START, END = "<!-- results:start -->", "<!-- results:end -->"
PART = {"IS": "학습", "OOS": "표본 밖", "ALL": "전체"}
SOURCE = {"reported": "원본 보고치", "rerun": "원본 코드 재실행", "ours": "우리 코드"}
VARIANT = {"original": "원본 (신호일 양봉 포함)", "no_signal_day": "신호일 뺌"}
PRICE = {"csv": "원본 CSV", "us": "us_data QQQ"}
VARNAME = {"all": "모든 조합", "open": "open 조합만", "close": "close 조합만"}
MIXNAME = {"rsi2": "RSI(2) 후보", "ibs": "IBS 원본 규칙", "or": "둘 중 하나 (OR)", "and": "둘 다 (AND)", "half": "자본 반반"}
KIND = {"wf": "걸어가며 검증", "candidate": "후보 조합"}


# ---------- text derived from config (F9) ----------

def cost_us(cfg: dict) -> str:
    c = cfg["us"]["costs"]
    return f"편도 {c['buy_fee'] + c['slippage']:.2%}"


def cost_us_detail(cfg: dict) -> str:
    c = cfg["us"]["costs"]
    return f"{cost_us(cfg)}(수수료 {c['buy_fee']:.2%} + 슬리피지 {c['slippage']:.2%})"


def cost_orig(cfg: dict) -> str:
    return f"편도 {cfg['original']['slippage'] * 1e4:.0f}bp"


def rule_text(cfg: dict) -> str:
    o = cfg["original"]
    return f"RSI({cfg['rsi']['period']}) < {o['rsi_thr']}, 종가 > {o['sma']}일선, {EXIT['green2']} 청산"


def ibs_text(cfg: dict) -> str:
    b = cfg["ibs"]
    return f"IBS < {b['entry']} 진입, > {b['exit']} 청산, {b['exec']} 체결"


def grid_dims(cfg: dict) -> tuple[int, int, str]:
    """(combos per trend filter, all combos, 'ticker 4 x threshold 3 x ...') from the grid lists."""
    g = cfg["grid"]
    parts = [("종목", g["tickers"]), ("문턱", g["rsi_thr"]), ("청산", g["exit"]), ("체결", g["exec"])]
    per = math.prod(len(v) for _, v in parts)
    return per, per * len(g["trend"]), " × ".join(f"{n} {len(v)}" for n, v in parts)


def n_trials(cfg: dict) -> int:
    return sum(cfg["overfit"]["trials"].values())


def top_share(cfg: dict) -> str:
    return f"상위 {cfg['overfit']['monkey_top']:.0%}"


def krw(k) -> str:
    k = int(k)
    if k % 100_000_000 == 0:
        return f"{k // 100_000_000}억 원"
    if k % 10_000_000 == 0:
        return f"{k // 10_000_000}천만 원"
    return f"{k:,}원"


def start_year(cfg: dict) -> str:
    return str(cfg["data"]["start"])[:4]


def sma_txt(cfg: dict) -> str:
    return f"{cfg['original']['sma']}일선"


# ---------- formatting ----------

def intf(v) -> str:
    return "" if pd.isna(v) else f"{int(v)}"


def bp(v) -> str:
    return "" if pd.isna(v) else f"{v:.1f}"


def flag(v) -> bool:
    return str(v) == "True"


def flag_series(s: pd.Series) -> pd.Series:
    return s.astype(str).eq("True")


def trade_cols(r) -> dict:
    return {"거래": intf(r["trades"]), "승률": share(r["win_rate"]), "PF": num(r["profit_factor"]),
            "거래당": pct(r["expectancy"], 2), "연 수익률": pct(r["cagr"]), "일별 최대 낙폭": pct(r["mdd"])}


def trade_table(df: pd.DataFrame, lead: dict, tail: dict | None = None) -> str:
    body = pd.DataFrame([trade_cols(r) for _, r in df.iterrows()], index=df.index)
    return md_table(pd.concat([pd.DataFrame(lead, index=df.index), body, pd.DataFrame(tail or {}, index=df.index)], axis=1))


def us_grid(g: pd.DataFrame) -> pd.DataFrame:
    u = g[g["costs"] == "us"].copy()
    u["조합"] = [label(r) for _, r in u.iterrows()]
    return u


def top10(cfg: dict, u: pd.DataFrame) -> pd.DataFrame:
    m = cfg["walkforward"]["min_train_trades"]
    return u[u["trades"] >= m].sort_values(["sharpe", "combo"], ascending=[False, True]).head(10)


def primary(ov: pd.DataFrame, var: str) -> pd.Series | None:
    """The pre-registered DSR row (N = all trials, V from this grid) of one variant; None when missing."""
    return one(ov, variant=var, case="primary")


# ---------- results ----------

def trend_pairs(g: pd.DataFrame) -> dict:
    u = g[g["costs"] == "us"]
    p = u.pivot_table(index=["ticker", "rsi_thr", "exit", "exec"], columns="trend", values="sharpe")
    p = p.reindex(columns=list(TREND))
    mean = u.groupby("trend")["sharpe"].mean()
    return {"n": len(p), "above_none": int((p["above"] > p["none"]).sum()), "below_none": int((p["below"] > p["none"]).sum()),
            "below_above": int((p["below"] > p["above"]).sum()), **{f"mean_{k}": float(mean.get(k, np.nan)) for k in TREND}}


def decision(cfg: dict, wfs: pd.DataFrame, ov: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    dmin, top = cfg["overfit"]["dsr_min"], 1 - cfg["overfit"]["monkey_top"]
    rows = []
    for var in WF_VARIANTS:
        s, h, kd, ke = (one(wfs, variant=var, name=n) for n in ("wf", "hold", "kdd", "kexp"))
        o = primary(ov, var)
        if any(x is None for x in (s, h, kd, ke, o)):
            rows.append({"variant": var, "c1": False, "c2": False, "c3": False, "c4": False, "note": "행 없음"})
            continue
        rows.append({"variant": var, "c1": bool(s["sharpe"] > h["sharpe"]),
                     "c2": bool(s["cagr"] > kd["cagr"] and s["cagr"] > ke["cagr"]),
                     "c3": bool(o["dsr"] >= dmin), "c4": bool(o["monkey_pctile"] >= top), "note": "",
                     "sharpe": s["sharpe"], "hold_sharpe": h["sharpe"], "cagr": s["cagr"], "kdd_cagr": kd["cagr"],
                     "kexp_cagr": ke["cagr"], "dsr": o["dsr"], "pctile": o["monkey_pctile"], "label": o["label"]})
    d = pd.DataFrame(rows)
    d["pass"] = d[["c1", "c2", "c3", "c4"]].all(axis=1)
    passed = d[d["pass"]]
    if len(passed):
        line = "신호 알림 후보로 올린다: " + ", ".join(f"{VARNAME[v]} ({l})" for v, l in zip(passed["variant"], passed["label"])) + "."
    else:
        counts = ", ".join(f"{k} {int(d[c].sum())}" for k, c in (("샤프 > 보유", "c1"), ("연 수익률 > 위험·노출 맞춘 보유", "c2"),
                                                                  (f"DSR ≥ {cfg['overfit']['dsr_min']}", "c3"),
                                                                  (f"무작위 {top_share(cfg)}", "c4")))
        line = f"신호 알림 후보로 올리지 않는다: 기준 넷을 모두 만족한 판본 0/{len(d)} ({counts})."
    return d, line


def sec1(cfg, s0, meta, st, em) -> list[str]:
    rows = []
    for p in PART:
        for k, name in SOURCE.items():
            r = one(s0, source=k, part=p)
            if r is None:
                rows.append({"구간": PART[p], "출처": name, "거래": "행 없음"})
                continue
            rows.append({"구간": PART[p], "출처": name, **trade_cols(r), "노출": share(r["exposure"])})
    o = cfg["original"]
    n_orig, n_ours, same = int(st["orig_entry"].notna().sum()), int(st["our_entry"].notna().sum()), int(st["same_dates"].sum())
    return ["## 1. 원본 재현", "",
            f"원본 저장소(커밋 {meta['commit']})에 들어 있는 `{o['csv']}` 하나로 원본 보고치, 원본 코드 재실행, 우리 코드를 나란히 놓았다. "
            f"규칙은 {rule_text(cfg)}(신호 다음 날 시가 체결)이다. "
            f"표본 {meta['first']} → {meta['last']} ({int(meta['bars']):,}거래일), 학습 → {meta['is_last']}, 표본 밖 {meta['oos_first']} →. "
            f"비용 {cost_orig(cfg)}, 전액 투입. 노출은 원본 정의(일별 수익률이 0 이 아닌 날의 비율)다.", "",
            md_table(pd.DataFrame(rows)), "",
            f"- 원본 코드 재실행 수치는 저장소에 들어 있던 `metricas.json` 과 {'같고' if flag(meta['json_same']) else '다르고'}, "
            f"`trades.csv` 도 {'같다' if flag(meta['trades_csv_same']) else '다르다'}.",
            f"- 거래 목록: 원본 {n_orig}건, 우리 {n_ours}건, 진입일·청산일이 같은 거래 {same}건, 수익률 최대 차이 {st['ret_diff'].max():.1e} "
            "(원본 trades.csv 는 소수 다섯째 자리로 반올림돼 있다).",
            f"- RSI(2): {int(meta['rsi_days']):,}일 모두 원본 코드 값과 비교했다. 소수 넷째 자리에서 다른 날 {meta['rsi_mismatch_4dp']}일, "
            f"최대 차이 {float(meta['rsi_max_diff']):.1e}.", "",
            f"### 엔진 대조 (단계 1, {cfg['data']['start']} 이후 신호, 원본 규칙, {cost_orig(cfg)})", "",
            md_table(pd.DataFrame({"가격": em["source"].map(PRICE), "판본": em["variant"].map(VARIANT),
                                   "반복문 거래": em["loop_trades"], "엔진 거래": em["engine_trades"], "다른 거래": em["mismatched"],
                                   "수익률 최대 차이": em["max_ret_diff"].map(lambda v: f"{v:.1e}")})), ""]


def sec2(cfg, rp, qk) -> list[str]:
    a, b = one(rp, variant="original", part="ALL"), one(rp, variant="no_signal_day", part="ALL")
    qa = one(qk, source="us", variant="original")
    out = ["## 2. 청산 조건 특이점 (신호일 양봉 포함 여부)", "",
           "원본의 청산 조건 \"전날과 그 전날이 모두 양봉\"에서 그 전날이 신호일일 수 있다. 신호일과 진입일이 모두 양봉이면 진입 다음 날 시가에 판다. "
           "엔진도 청산 배열을 진입일부터 보므로 같게 동작한다(1장 엔진 대조). \"신호일 뺌\"은 진입일의 청산 조건을 무시해 두 양봉이 모두 진입일 이후여야 판다. "
           "\"끝에 열린 거래\"는 구간 마지막 날에 아직 팔지 않은 거래 수이고, 마지막 날 종가에서 매도 비용을 빼고 평가했다.", "",
           "### 원본 CSV (단계 0 반복문)", "",
           trade_table(rp, {"구간": rp["part"].map(PART), "판본": rp["variant"].map(VARIANT)},
                       {"노출": rp["exposure"].map(share), "신호일 양봉으로 판 거래": rp["signal_day_exits"].map(intf),
                        "끝에 열린 거래": rp["open_at_end"].map(intf)}), "",
           f"### 엔진 ({cfg['data']['start']} 이후 신호, {cost_orig(cfg)})", "",
           trade_table(qk, {"가격": qk["source"].map(PRICE), "판본": qk["variant"].map(VARIANT),
                            "기간": qk["first"].astype(str) + " → " + qk["last"].astype(str)},
                       {"샤프": qk["sharpe"].map(num), "신호일 양봉으로 판 거래": qk["signal_day_exits"].map(intf),
                        "끝에 열린 거래": qk["open_at_end"].map(intf)}), ""]
    qcsv = one(qk, source="csv", variant="original")
    if qa is not None and pd.notna(qa["shared_csv_us"]) and qcsv is not None:
        n_shared, n_total = int(qa["shared_csv_us"]), int(qcsv["trades"])
        same_txt = "거래가 전부 같았다" if n_shared == n_total else f"{n_total}건 중 {n_shared}건만 같고 나머지는 달랐다"
        ratio_txt = (f"약 {abs(1 - qa['csv_us_ratio']) * 100:.3f}%" if pd.notna(qa.get("csv_us_ratio")) else "알 수 없는 비율")
        out += [f"- 두 가격 파일이 같은 진입일·청산일로 낸 거래: {n_shared}/{n_total}건 "
                f"(원본 CSV 와 us_data 는 배당 수정 계수가 달라 겹치는 구간의 종가 비율이 거의 일정하게 {ratio_txt} 다르지만, 이 기간에는 {same_txt}).", ""]
    if a is None or b is None:
        return out + ["- 원본 CSV 전체 구간 행이 없어 차이를 적지 못했다.", ""]
    return out + [f"- 원본 CSV 전체 구간에서 신호일 양봉을 센 청산은 {int(a['signal_day_exits'])}건(거래의 {share(a['signal_day_exits'] / a['trades'])})이다. "
                  f"신호일을 빼면 거래 {int(a['trades'])} → {int(b['trades'])}건, 승률 {share(a['win_rate'])} → {share(b['win_rate'])}, "
                  f"PF {num(a['profit_factor'])} → {num(b['profit_factor'])}, 연 수익률 {pct(a['cagr'])} → {pct(b['cagr'])} 이다.", ""]


def sec3(cfg, g) -> list[str]:
    u = g[g["costs"] == "us"]
    order = list(TREND)
    sma = sma_txt(cfg)
    per, _, dims = grid_dims(cfg)
    agg = u.groupby("trend").agg(n=("combo", "size"), trades=("trades", "mean"), win=("win_rate", "mean"),
                                 pf=("profit_factor", "median"), exp=("expectancy", "mean"), cagr=("cagr", "mean"),
                                 mdd=("mdd", "mean"), sharpe=("sharpe", "mean"), expo=("exposure", "mean")).reindex(order)
    t = pd.DataFrame({"추세 필터": [TREND[k] for k in order], "조합": agg["n"].map(intf).to_numpy(),
                      "거래(평균)": agg["trades"].map(lambda v: f"{v:.0f}").to_numpy(), "승률": agg["win"].map(share).to_numpy(),
                      "PF(중앙값)": agg["pf"].map(num).to_numpy(), "거래당": agg["exp"].map(lambda v: pct(v, 2)).to_numpy(),
                      "연 수익률": agg["cagr"].map(pct).to_numpy(), "낙폭": agg["mdd"].map(pct).to_numpy(),
                      "샤프": agg["sharpe"].map(num).to_numpy(), "노출": agg["expo"].map(share).to_numpy()})
    bt = u.pivot_table(index="ticker", columns="trend", values="sharpe", aggfunc="mean").reindex(index=cfg["grid"]["tickers"], columns=order)
    tt = pd.DataFrame({"종목": bt.index, **{TREND[k]: bt[k].map(num).to_numpy() for k in order}})
    p = trend_pairs(g)
    n = p["n"]
    an, ba = p["above_none"], p["below_above"]
    if an == 0:
        bnf_dir = f"bnf 와 같다({n}쌍 중 {sma} 위가 나은 쌍이 없다)"
    elif an < n / 2:
        bnf_dir = f"bnf 와 같은 방향이지만, {n}쌍 중 {an}쌍은 {sma} 위가 나아 bnf 처럼 한쪽으로 쏠리지는 않았다"
    else:
        bnf_dir = f"bnf 와 반대 방향이다({n}쌍 중 {an}쌍에서 {sma} 위가 필터 없음보다 낫다)"
    if ba > n / 2:
        c3_dir = f"2번 C3 와 같은 방향이다({n}쌍 중 {ba}쌍에서 {sma} 아래가 위보다 낫다)"
    else:
        c3_dir = f"2번 C3 와 반대 방향이다({sma} 아래가 위보다 나은 쌍이 {n}쌍 중 {ba}쌍이다)"
    rev = [t for t in bt.index if bt.loc[t, "below"] > bt.loc[t, "above"]]
    rev_line = (f" 종목별 평균 샤프로는 {', '.join(rev)} 에서 {sma} 아래가 위보다 높다." if rev
                else f" 종목별 평균 샤프로도 {sma} 아래가 위보다 높은 종목은 없다.")
    return ["## 3. 추세 필터 3종 비교 (이 작업의 핵심)", "",
            f"비용 {cost_us_detail(cfg)}, 필터별 {per}개 조합({dims})의 평균이다.", "",
            md_table(t), "", "### 종목별 평균 샤프", "", md_table(tt), "",
            f"같은 종목·문턱·청산·체결에서 필터만 바꾼 {n}쌍을 비교했다. {sma} 위가 필터 없음보다 샤프가 높은 쌍 {p['above_none']}쌍, "
            f"{sma} 아래가 필터 없음보다 높은 쌍 {p['below_none']}쌍, {sma} 아래가 위보다 높은 쌍 {p['below_above']}쌍이다. "
            f"{cfg['refs']['bnf']} RSI(2) 는 {bnf_dir}. {cfg['refs']['ibs_c3']} RSI(2) 는 {c3_dir}.{rev_line} "
            f"다만 2번 C3 는 지수의 {sma}과 신호 다음 날 수익, 여기는 종목 자신의 {sma}과 전략 샤프를 본 것이다.", ""]


def sec4(cfg, g) -> list[str]:
    u = us_grid(g)
    top = top10(cfg, u)
    m = cfg["walkforward"]["min_train_trades"]
    t = trade_table(top, {"조합": top["조합"]},
                    {"PF(최대 거래 뺌)": top["pf_ex_best"].map(num), "샤프": top["sharpe"].map(num), "노출": top["exposure"].map(share),
                     "손익분기(편도bp)": top["breakeven_bp"].map(bp),
                     "끝에 열린 거래": top["open_at_end"].map(lambda v: "있음" if flag(v) else "")})
    ag = u.groupby(["exit", "exec"]).agg(n=("combo", "size"), trades=("trades", "mean"), win=("win_rate", "mean"),
                                        pf=("profit_factor", "median"), exp=("expectancy", "mean"), cagr=("cagr", "mean"),
                                        mdd=("mdd", "mean"), sharpe=("sharpe", "mean"), be=("breakeven_bp", "median")).reset_index()
    ag["o1"] = ag["exit"].map({k: i for i, k in enumerate(EXIT)})
    ag["o2"] = ag["exec"].map({"open": 0, "close": 1})
    ag = ag.sort_values(["o1", "o2"])
    at = pd.DataFrame({"청산": ag["exit"].map(EXIT), "체결": ag["exec"], "조합": ag["n"], "거래(평균)": ag["trades"].map(lambda v: f"{v:.0f}"),
                       "승률": ag["win"].map(share), "PF(중앙값)": ag["pf"].map(num), "거래당": ag["exp"].map(lambda v: pct(v, 2)),
                       "연 수익률": ag["cagr"].map(pct), "낙폭": ag["mdd"].map(pct), "샤프": ag["sharpe"].map(num),
                       "손익분기 중앙값(bp)": ag["be"].map(bp)})
    cc = g.groupby(["costs", "exec"]).agg(exp=("expectancy", "mean"), cagr=("cagr", "mean"), sharpe=("sharpe", "mean"),
                                          pos=("cagr", lambda s: int((s > 0).sum()))).reset_index()
    ct = pd.DataFrame({"비용": cc["costs"].map({"us": cost_us(cfg), "orig": f"원본 {cost_orig(cfg)}"}), "체결": cc["exec"],
                       "거래당(평균)": cc["exp"].map(lambda v: pct(v, 2)), "연 수익률(평균)": cc["cagr"].map(pct),
                       "샤프(평균)": cc["sharpe"].map(num), "연 수익률 > 0 인 조합": cc["pos"]})
    n_open = int(flag_series(u["open_at_end"]).sum())
    return ["## 4. 그리드 상위 10개와 청산·체결별 평균", "",
            f"{len(u)}개 조합, {cfg['data']['start']} → {cfg['data']['end']}, 비용 {cost_us(cfg)}. 샤프 순, 거래 {m}건 미만 조합은 뺐다({int((u['trades'] < m).sum())}개). "
            "손익분기 비용은 비용을 내고 난 누적 수익이 0 이 되는 편도 비용이다. "
            f"표본 끝에 열린 거래가 있는 조합은 {len(u)}개 중 {n_open}개이고, 그 거래는 마지막 날 종가에서 매도 비용을 빼고 평가했다(\"끝에 열린 거래\" 열).", "",
            t, "", "### 청산 × 체결별 평균", "", md_table(at), "", f"### 비용별 평균 (원본 비용 {cost_orig(cfg)} 결과 포함)", "", md_table(ct), ""]


def sec5(cfg, g) -> list[str]:
    u = us_grid(g)
    n = len(u)
    both = (u["cagr"] > u["kdd_cagr"]) & (u["cagr"] > u["kexp_cagr"])
    top = top10(cfg, u)
    t = pd.DataFrame({"조합": top["조합"], "연 수익률": top["cagr"].map(pct), "보유": top["hold_cagr"].map(pct),
                      "낙폭 맞춘 보유": top["kdd_cagr"].map(pct), "k": top["kdd"].map(num) + top["kdd_capped"].map(lambda v: "*" if flag(v) else ""),
                      "노출 맞춘 보유": top["kexp_cagr"].map(pct), "w": top["kexp"].map(num),
                      "샤프 전략/보유": top["sharpe"].map(num) + " / " + top["hold_sharpe"].map(num),
                      "낙폭 전략/보유": top["mdd"].map(pct) + " / " + top["hold_mdd"].map(pct)})
    by = u.assign(both=both).groupby("trend")["both"].agg(["sum", "size"]).reindex(list(TREND))
    return ["## 5. 비교 기준 3종 대비", "",
            "같은 종목 매수 보유, 보유를 전략의 일별 최대 낙폭에 맞춰 줄인 것(비중 k, 나머지 현금 0%), 보유를 전략의 노출에 맞춰 줄인 것(비중 w = 노출). "
            "맞춘 두 곡선에는 매매 비용이 없고, 보유에는 처음 살 때와 끝에 팔 때 비용이 있다. "
            f"`*` 는 전략 낙폭이 보유보다 깊어 k 가 상한 1 에 걸린 조합이다({n}개 중 {int(flag_series(u['kdd_capped']).sum())}개).", "",
            f"- 연 수익률이 보유보다 높은 조합 {int((u['cagr'] > u['hold_cagr']).sum())}/{n}, 낙폭 맞춘 보유보다 높은 조합 "
            f"{int((u['cagr'] > u['kdd_cagr']).sum())}/{n}, 노출 맞춘 보유보다 높은 조합 {int((u['cagr'] > u['kexp_cagr']).sum())}/{n}, "
            f"둘 다 높은 조합 {int(both.sum())}/{n}.",
            f"- 샤프가 보유보다 높은 조합 {int((u['sharpe'] > u['hold_sharpe']).sum())}/{n}.",
            "- 필터별로 두 맞춘 보유를 모두 이긴 조합: " + ", ".join(f"{TREND[k]} {int(r['sum'])}/{int(r['size'])}" for k, r in by.iterrows()) + ".", "",
            "### 그리드 상위 10개", "", md_table(t), ""]


def sec6(cfg, wfs, wfp) -> list[str]:
    fy, m = cfg["walkforward"]["first_test_year"], cfg["walkforward"]["min_train_trades"]
    w = wfs.copy()
    w["구분"] = w["variant"].map(VARNAME).fillna("")
    t = pd.DataFrame({"구분": w["구분"], "방식": w["label"], "연 수익률": w["cagr"].map(pct), "낙폭": w["mdd"].map(pct),
                      "샤프": w["sharpe"].map(num), "거래": w["trades"].map(intf), "노출·비중": w["exposure"].map(share),
                      "상한": w["capped"].map(lambda v: "*" if flag(v) else "")})
    out = ["## 6. 걸어가며 검증", "",
           f"{fy}년부터 해마다 그 전까지의 일별 수익률(비용 {cost_us(cfg)})만 보고 샤프 1등 조합(그 전 거래 {m}건 이상)을 골라 그 해에 쓴다. "
           f"학습 기간은 {start_year(cfg)}년부터 늘어난다. 한 계좌·한 자리로 이어 붙였다. 보유 비교는 해마다 고른 종목을 그 해 들고 있는 것이다. "
           f"\"뒤늦게 고른 1등\"은 전체 기간 샤프 1등 조합을 {fy}년부터 쓴 것으로, 미래를 본 값이라 비교용이다.", "",
           md_table(t), "",
           f"원본의 걸어가며 검증은 규칙({rule_text(cfg)})을 고정한 채 3년 학습 / 1년 검증 창을 한 해씩 옮긴 것이다. "
           "여기서는 해마다 과거만 보고 조합을 새로 고르므로, 원본은 고정 규칙이 해마다 버티는지를, 여기는 고르는 방법까지 포함한 성과를 잰다.", ""]
    for var in WF_VARIANTS:
        p = wfp[wfp["variant"] == var]
        s = one(wfs, variant=var, name="wf")
        head = f"### 해마다 고른 조합 ({VARNAME[var]})"
        if s is not None:
            head += f": 서로 다른 조합 {intf(s['distinct_picks'])}개, 바뀐 횟수 {intf(s['changes'])}번"
        out += [head, "", md_table(pd.DataFrame({"해": p["year"], "고른 조합": p["label"], "학습 샤프": p["train_sharpe"].map(num),
                                                 "학습 거래": p["train_trades"], "그 해 샤프": p["test_sharpe"].map(num),
                                                 "그 해 수익률": p["test_return"].map(pct),
                                                 "그 해 순위(백분위)": p["test_pctile"].map(lambda v: f"{v:.0%}")})), ""]
    return out


def sec7(cfg, mx, cands) -> list[str]:
    c = one(cands, variant="open")
    if c is None:
        return ["## 7. IBS 와 섞은 결과", "", "- open 판본 후보가 없어 계산하지 못했다.", ""]
    t = pd.DataFrame({"종목": mx["ticker"] + mx["candidate"].map(lambda v: " (후보 종목)" if flag(v) else ""),
                      "방식": mx["name"].map(MIXNAME), "연 수익률": mx["cagr"].map(pct), "낙폭": mx["mdd"].map(pct),
                      "샤프": mx["sharpe"].map(num), "거래": mx["trades"], "노출": mx["exposure"].map(share)})
    per = mx.drop_duplicates("ticker")
    pt = pd.DataFrame({"종목": per["ticker"], "일별 수익 상관": per["corr"].map(num),
                       "RSI(2) 진입일": per["entries_rsi2"], "IBS 진입일": per["entries_ibs"], "같은 날": per["entries_both"],
                       "겹침(RSI 기준)": per["overlap_rsi2"].map(share), "겹침(IBS 기준)": per["overlap_ibs"].map(share)})
    r, i, h = (one(mx, ticker=c["ticker"], name=k) for k in ("rsi2", "ibs", "half"))
    if r is None or i is None or h is None:
        verdict = "- 후보 종목 행이 없어 판정하지 못했다."
    else:
        low = r["corr"] < cfg["mix"]["low_corr"]
        better = h["sharpe"] > max(r["sharpe"], i["sharpe"])
        verdict = (f"- 후보 종목 {c['ticker']}: 상관 {num(r['corr'])} ({'낮다' if low else '낮지 않다'}, 기준 {cfg['mix']['low_corr']}), "
                   f"반반 샤프 {num(h['sharpe'])} vs RSI(2) {num(r['sharpe'])}, IBS {num(i['sharpe'])} ({'각각보다 높다' if better else '각각보다 높지 않다'}). "
                   + ("알림을 둘 다 받는 것이 낫다는 근거가 된다." if low and better else "알림을 둘 다 받을 근거는 되지 않는다."))
    return ["## 7. IBS 와 섞은 결과", "",
            f"1번(ibs_backtest)은 판단 기준을 통과하지 못해 1번 원본 규칙({ibs_text(cfg)})을 썼다. "
            f"RSI(2) 쪽은 걸어가며 검증(open 조합만)이 마지막 해에 고른 조합 `{c['label']}` 이다. "
            f"{cfg['data']['start']} 이후, 비용 {cost_us(cfg)}. OR 은 먼저 난 신호의 규칙으로 팔고, AND 는 두 청산 중 먼저 오는 날 판다.", "",
            md_table(pt), "", md_table(t), "", verdict, ""]


def dsr_rows(cfg, ov) -> pd.DataFrame:
    """Primary and sensitivity DSR rows of the open and close variants, in that order."""
    dmin, own = cfg["overfit"]["dsr_min"], cfg["overfit"]["trials"]["rsi2_backtest"]
    rows = []
    for var in ("open", "close"):
        p = primary(ov, var)
        if p is not None:
            rows.append({"var": var, "case": "primary", "n": p["n_trials"], "v_src": f"이 그리드 {own}개 (미리 정함)",
                         "var_sr": p["var_sr"], "sr0": p["sr0"], "dsr": p["dsr"]})
        for _, r in ov[ov["variant"].astype(str).str.startswith(var + "|")].iterrows():
            rows.append({"var": var, "case": r["case"], "n": r["n_trials"], "v_src": r["v_source"],
                         "var_sr": r["var_sr"], "sr0": r["sr0"], "dsr": r["dsr"]})
    d = pd.DataFrame(rows)
    if len(d):
        d["ok"] = d["dsr"] >= dmin
    return d


def nonzero_days(daily: pd.DataFrame | None, combo) -> int | None:
    if daily is None or pd.isna(combo) or str(int(combo)) not in daily.columns:
        return None
    return int((daily[str(int(combo))] != 0).sum())


def sec8(cfg, ov, at, daily, cands) -> list[str]:
    v = ov[ov["variant"].isin(list(WF_VARIANTS)) & (ov["case"] == "primary")]
    dmin = cfg["overfit"]["dsr_min"]
    t = pd.DataFrame({"판본": v["variant"].map(VARNAME), "후보": v["label"], "샤프(연)": v["sharpe"].map(num),
                      "일별 샤프": v["sr_daily"].map(lambda x: num(x, 4)), "일수": v["T"].map(intf), "왜도": v["skew"].map(num),
                      "첨도": v["kurt"].map(num), "시행 수": v["n_trials"].map(intf), "시행 샤프 분산": v["var_sr"].map(lambda x: f"{x:.2e}"),
                      "SR0(일별)": v["sr0"].map(lambda x: num(x, 4)), "DSR": v["dsr"].map(lambda x: num(x, 3)),
                      "PF": v["profit_factor"].map(num), "무작위 PF 백분위": v["monkey_pctile"].map(lambda x: num(x, 3)),
                      "무작위 PF 중앙값": v["monkey_median_pf"].map(num)})
    s0 = one(ov, variant="stage0")
    s0_line = ("- 원본 규칙·원본 CSV 에서 같은 방식으로 무작위 진입 {runs:,}번: 우리 백분위 {a}, 원본 metricas.json {b}.".format(
        runs=int(s0["monkey_runs"]), a=num(s0["monkey_pctile"], 4), b=num(s0["orig_monkey_pct_pf"], 4)) if s0 is not None else "- stage0 행 없음")

    ds = dsr_rows(cfg, ov)
    sens_lines = []
    if len(ds):
        st = pd.DataFrame({"판본": ds["var"].map(VARNAME), "시행 수 N": ds["n"].map(lambda x: f"{int(x):,}"),
                           "분산 V 를 잡은 시행": ds["v_src"], "V": ds["var_sr"].map(lambda x: f"{x:.2e}"),
                           "SR0(일별)": ds["sr0"].map(lambda x: num(x, 4)), "DSR": ds["dsr"].map(lambda x: num(x, 3)),
                           f"DSR ≥ {dmin}": ds["ok"].map({True: "예", False: "아니오"})})
        flips, bullets = False, []
        for var in ("open", "close"):
            d = ds[ds["var"] == var]
            p = d[d["case"] == "primary"]
            if not len(p):
                continue
            p_ok = bool(p["ok"].iloc[0])
            other = d[(d["case"] != "primary") & (d["ok"] != p_ok)]
            head = f"{VARNAME[var]} 판본: 미리 정한 값의 DSR 은 {num(p['dsr'].iloc[0], 3)}({'합격' if p_ok else '불합격'})"
            if len(other):
                flips = True
                bullets.append(f"- {head}이지만, " + ", ".join(f"N = {int(r['n']):,}, V = {r['v_src']} 로 잡으면 {num(r['dsr'], 3)}"
                                                           for _, r in other.iterrows())
                               + f" 로 {dmin} {'미만' if p_ok else '이상'}이 된다.")
            else:
                bullets.append(f"- {head}이고, N 과 V 를 달리 잡은 {len(d) - 1}가지에서도 판정이 같다.")
        dep = ("- 따라서 이 기준의 판정은 N 과 V 를 어떻게 잡느냐에 달려 있다. 판단에는 미리 정한 값을 그대로 쓴다."
               if flips else "- N 과 V 를 바꿔도 이 기준의 판정은 같다.")
        all_c, open_c = one(cands, variant="all"), one(cands, variant="open")
        same_combo = (all_c is not None and open_c is not None and all_c["combo"] == open_c["combo"]
                     and all_c["label"] == open_c["label"])
        omit_note = (f"- \"모든 조합\" 판본은 이 표에 없다. 걸어가며 검증이 마지막 해에 고른 조합이 open 판본과 같기 때문이다"
                    f"(둘 다 조합 {int(all_c['combo'])}번 `{all_c['label']}`)." if same_combo else
                    "- \"모든 조합\" 판본의 후보가 open 판본과 다른데도 이 표에 빠져 있다(확인 필요).")
        sens_lines = ["### Deflated Sharpe 민감도", "",
                      f"판단에는 계획을 쓸 때, 결과를 보기 전에 정한 값(N = 세 폴더 시행 수 합 {n_trials(cfg):,}, "
                      f"V = 이 그리드 {cfg['overfit']['trials']['rsi2_backtest']}개 조합의 일별 샤프 분산)을 쓴다. 결과를 본 뒤 바꾸면 결과에 맞춘 선택이 되기 때문이다. "
                      "나머지 줄은 N 과 V 를 달리 잡았을 때의 값이다.", "",
                      md_table(st), "", omit_note, *bullets, dep]
    else:
        sens_lines = ["- Deflated Sharpe 행이 없다."]
    mom = 0
    for _, r in v[v["variant"].isin(["open", "close"])].iterrows():
        nz = nonzero_days(daily, r["combo"])
        if nz is not None:
            mom += 1
            sens_lines.append(f"- {VARNAME[r['variant']]} 후보: 왜도 {num(r['skew'], 1)}, 첨도 {num(r['kurt'], 0)} 이 {int(r['T']):,}일 중 "
                              f"수익률이 0 이 아닌 {nz}일({nz / r['T']:.1%})에서 나왔다.")
    if mom:
        sens_lines.append("- 포지션이 없는 날이 대부분이라 왜도·첨도가 큰 수익 몇 개에 크게 흔들리고, DSR 도 그만큼 불안정하다.")
    sens_lines.append("")

    a = at.copy()
    a["구분"] = a["kind"].map(KIND) + " · " + a["variant"].map(VARNAME)
    tt = pd.DataFrame({"구분": a["구분"], "자본": a["capital_krw"].map(krw),
                       "세전": a["pre_cagr"].map(pct), "세후": a["post_cagr"].map(pct),
                       "세금 합계(원)": a["tax_total_krw"].map(lambda x: f"{x:,.0f}"), "세금 낸 해": a["years_taxed"],
                       "보유 세전": a["hold_pre_cagr"].map(pct), "보유 세후": a["hold_post_cagr"].map(pct)})
    low = at.loc[at["min_cash"].idxmin()]
    if low["min_cash"] < 0:
        cash_line = (f"- 세금 낼 때 포지션이 열려 있으면 현금이 음수가 된다. 가장 낮은 현금: {KIND.get(low['kind'], low['kind'])}"
                     f"({VARNAME.get(low['variant'], low['variant'])} 판본), 자본 {krw(low['capital_krw'])}, {low['min_cash_date']}, "
                     f"{low['min_cash']:,.0f} 달러(그날 평가금액의 {low['min_cash_share']:.1%}).")
    else:
        cash_line = "- 세금을 낼 때 현금이 음수가 된 계좌는 없다."
    tc = cfg["tax"]
    ft = cfg["overfit"]["trials"]
    return ["## 8. Deflated Sharpe, 무작위 진입, 세후", "",
            f"Deflated Sharpe 는 Bailey·López de Prado(2014) 식을 직접 짰다. 시행 수 = 1번 {ft['ibs_backtest']} + "
            f"2번 {ft['ibs_lev_backtest']} + 3번 {ft['rsi2_backtest']} = {n_trials(cfg):,}. 시행 샤프 분산은 이 그리드 {ft['rsi2_backtest']}개 조합의 일별 샤프로 잡았다. "
            f"후보는 판본마다 걸어가며 검증이 마지막 해에 고른 조합이고, {cfg['data']['start']} → {cfg['data']['end']} 일별 수익률(비용 {cost_us(cfg)})로 쟀다. "
            f"무작위 진입은 같은 거래 수, 같은 보유일 분포, 같은 추세 필터가 허락하는 날에서 {cfg['overfit']['monkey_runs']:,}번 뽑았다. "
            "무작위 진입 백분위는 이 전략 하나가 무작위 진입보다 나은지만 보고, 여러 조합 중 이 데이터에서 가장 좋아 보인 조합을 골랐다는 점은 바로잡지 않는다"
            "(그 보정은 시행 수를 나누는 Deflated Sharpe 가 한다).", "",
            md_table(t), "", s0_line, "", *sens_lines,
            f"### 세후 (양도세 {tc['rate']:.0%}, 해마다 공제 {tc['deduction_krw']:,}원, 환율 {tc['fx']:,}원 고정, Y 년 세금은 Y+1 년 {tc['pay_month']}월 첫 거래일, "
            "마지막 해는 마지막 날)", "",
            "보유는 마지막 날 한 번에 팔아 그날 세금을 낸다. 걸어가며 검증의 보유는 해마다 고른 종목을 이어 붙인 것이라, 종목을 바꿀 때의 세금은 넣지 않았다(단순화).", "",
            md_table(tt), "", cash_line, ""]


def sec9(cfg, dec, line) -> list[str]:
    yes = {True: "만족", False: "불만족"}
    t = pd.DataFrame({"판본": dec["variant"].map(VARNAME), "샤프 > 보유": dec["c1"].map(yes), "연 수익률 > 위험·노출 맞춘 보유": dec["c2"].map(yes),
                      f"DSR ≥ {cfg['overfit']['dsr_min']}": dec["c3"].map(yes),
                      f"무작위 PF {top_share(cfg)}": dec["c4"].map(yes), "비고": dec["note"]})
    out = ["## 9. 한 줄 결론", "",
           f"미리 정한 판단 기준 넷을 걸어가며 검증(비용 포함)과 그 후보에 댔다. DSR 은 8장의 미리 정한 값(N {n_trials(cfg):,})이다.", "",
           md_table(t), ""]
    if "c1" in dec.columns and "c2" in dec.columns and not dec["c1"].any() and not dec["c2"].any():
        out += [f"샤프 > 보유(C1), 연 수익률 > 위험·노출 맞춘 보유(C2) 는 {len(dec)}개 판본 모두 불만족이다. "
                "네 기준을 다 만족해야 후보로 올리므로, DSR(C3) 을 8장 민감도의 어느 값으로 바꿔도 이 결론은 바뀌지 않는다.", ""]
    out += [f"**{line}**", ""]
    return out


def readme_block(cfg, s0, st, rp, g, wfs, ov, mx, at, dec, line) -> str:
    rep, ours = one(s0, source="reported", part="ALL"), one(s0, source="ours", part="ALL")
    a, b = one(rp, variant="original", part="ALL"), one(rp, variant="no_signal_day", part="ALL")
    p = trend_pairs(g)
    u = us_grid(g)
    top = top10(cfg, u)
    sma = sma_txt(cfg)
    out = []
    if rep is not None and ours is not None:
        out += ["| 원본 CSV 전체 구간 | 거래 | 승률 | PF | 연 수익률 | 일별 최대 낙폭 | 노출 |", "|---|---|---|---|---|---|---|",
                *[f"| {n} | {intf(r['trades'])} | {share(r['win_rate'])} | {num(r['profit_factor'])} | {pct(r['cagr'])} | {pct(r['mdd'])} | {share(r['exposure'])} |"
                  for n, r in (("원본 보고치", rep), ("우리 코드", ours))], ""]
    out.append(f"- 원본 코드를 같은 CSV 로 다시 돌린 거래 {int(st['orig_entry'].notna().sum())}건과 우리 코드 거래의 진입일·청산일이 {int(st['same_dates'].sum())}건 같다.")
    if a is not None and b is not None:
        out.append(f"- 청산 특이점: 신호일 양봉을 센 청산이 {int(a['signal_day_exits'])}건이다. 신호일을 빼면 연 수익률이 {pct(a['cagr'])} 에서 {pct(b['cagr'])} 가 된다.")
    out.append(f"- 추세 필터({len(u)}개 조합, 비용 {cost_us(cfg)}): 평균 샤프가 {sma} 위 {num(p['mean_above'])}, 필터 없음 {num(p['mean_none'])}, "
               f"{sma} 아래 {num(p['mean_below'])} 다. 조건이 같은 {p['n']}쌍 중 {sma} 위가 필터 없음보다 나은 쌍은 {p['above_none']}쌍이다.")
    if len(top):
        r = top.iloc[0]
        out.append(f"- 그리드 샤프 1등: {r['조합']}, 샤프 {num(r['sharpe'])}, 연 {pct(r['cagr'])}, 같은 종목 보유 샤프 {num(r['hold_sharpe'])}.")
    both = int(((u["cagr"] > u["kdd_cagr"]) & (u["cagr"] > u["kexp_cagr"])).sum())
    out.append(f"- {len(u)}개 조합 중 낙폭을 맞춘 보유와 노출을 맞춘 보유를 둘 다 연 수익률에서 이긴 조합은 {both}개다.")
    out += ["", f"| 걸어가며 검증 {cfg['walkforward']['first_test_year']}년부터 | 연 수익률 | 샤프 | 같은 종목 보유 샤프 | 후보 조합 DSR | 후보 조합 무작위 PF 백분위 |",
            "|---|---|---|---|---|---|"]
    for r in dec.itertuples():
        s, h, o = one(wfs, variant=r.variant, name="wf"), one(wfs, variant=r.variant, name="hold"), primary(ov, r.variant)
        if s is None or h is None or o is None:
            out.append(f"| {VARNAME[r.variant]} | 행 없음 | | | | |")
            continue
        out.append(f"| {VARNAME[r.variant]} | {pct(s['cagr'])} | {num(s['sharpe'])} | {num(h['sharpe'])} | {num(o['dsr'], 3)} | {num(o['monkey_pctile'], 3)} |")
    out.append(f"\n뒤 두 열(DSR, 무작위 PF 백분위)은 걸어가며 검증이 아니라 그 후보 조합을 {cfg['data']['start']} → {cfg['data']['end']} 전체 구간에서 잰 값이다.")
    if "c1" in dec.columns and "c2" in dec.columns and not dec["c1"].any() and not dec["c2"].any():
        out.append(f"- 샤프 > 보유(C1), 연 수익률 > 위험·노출 맞춘 보유(C2) 는 {len(dec)}개 판본 모두 불만족이다. "
                   "따라서 DSR(C3) 을 어떤 값으로 잡아도 전체 결론(네 기준을 다 만족해야 후보로 올림)은 바뀌지 않는다.")
    out.append("")
    ds = dsr_rows(cfg, ov)
    if len(ds):
        d = ds[ds["var"] == "open"]
        p, o = d[d["case"] == "primary"], d[d["case"] != "primary"]
        if len(p) and len(o):
            flip = bool((o["ok"] != p["ok"].iloc[0]).any())
            out.append(f"- DSR 은 결과를 보기 전에 정한 값(N {n_trials(cfg):,}, V 이 그리드 {cfg['overfit']['trials']['rsi2_backtest']}개)이다. "
                       f"open 판본은 이 값으로 {num(p['dsr'].iloc[0], 3)} 이고, N·V 를 달리 잡은 {len(o)}가지는 "
                       + ", ".join(num(x, 3) for x in o["dsr"])
                       + (f" 이다. 이 기준({cfg['overfit']['dsr_min']} 이상)의 판정은 그 선택에 달려 있다." if flip
                          else " 이다. 어느 쪽이든 이 기준의 판정은 같다."))
    c = mx[flag_series(mx["candidate"])]
    rr, ii, hh = (one(c, name=k) for k in ("rsi2", "ibs", "half"))
    if rr is not None and ii is not None and hh is not None:
        out.append(f"- IBS 와 섞기({rr['ticker']}): 일별 수익 상관 {num(rr['corr'])}, 자본 반반 샤프 {num(hh['sharpe'])} (RSI(2) {num(rr['sharpe'])}, IBS {num(ii['sharpe'])}).")
    base = cfg["tax"]["base_capital_krw"]
    wf1 = one(at, kind="wf", variant="open", capital_krw=base)
    if wf1 is not None:
        out.append(f"- 세후({krw(base)}, open 판본 걸어가며 검증): 연 {pct(wf1['pre_cagr'])} → {pct(wf1['post_cagr'])}, "
                   f"같은 종목 보유 {pct(wf1['hold_pre_cagr'])} → {pct(wf1['hold_post_cagr'])}.")
    out += ["", f"**{line}**"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    rd = results_dir()
    rc = lambda n: pd.read_csv(rd / n)  # noqa: E731
    s0, st, rp, qk, em, g = rc("stage0.csv"), rc("stage0_trades.csv"), rc("replicate.csv"), rc("quirk.csv"), rc("engine_match.csv"), rc("grid.csv")
    wfs, wfp, cands, mx, ov, at = (rc("walkforward_summary.csv"), rc("walkforward.csv"), rc("candidates.csv"), rc("mix.csv"),
                                   rc("overfit.csv"), rc("aftertax.csv"))
    daily_path = rd / "grid_daily.parquet"
    daily = pd.read_parquet(daily_path) if daily_path.exists() else None
    m = pd.read_csv(rd / "stage0_meta.csv", dtype=str)
    meta = dict(zip(m["key"], m["value"]))
    dec, line = decision(cfg, wfs, ov)
    out = ["# RSI(2) 평균회귀 백테스트 결과", "",
           f"원본: tradeitsimplesolutions/TIS-RSI2-Research (QQQ). 엔진 구간 {cfg['data']['start']} → {cfg['data']['end']}. "
           "샤프는 무위험 수익률을 빼지 않은 일별 수익률 기준. 수치는 모두 `results/` 의 csv 와 일별 수익률 parquet 에서 코드로 읽었다.", ""]
    out += sec1(cfg, s0, meta, st, em) + sec2(cfg, rp, qk) + sec3(cfg, g) + sec4(cfg, g) + sec5(cfg, g)
    out += sec6(cfg, wfs, wfp) + sec7(cfg, mx, cands) + sec8(cfg, ov, at, daily, cands) + sec9(cfg, dec, line)
    text = "\n".join(out) + "\n"
    (rd / "report.md").write_text(text, encoding="utf-8")
    block = readme_block(cfg, s0, st, rp, g, wfs, ov, mx, at, dec, line)
    (rd / "readme_results.md").write_text(block, encoding="utf-8")
    if a.readme:
        path = ROOT / "README.md"
        doc = path.read_text(encoding="utf-8")
        i, j = doc.index(START) + len(START), doc.index(END)
        path.write_text(doc[:i] + "\n" + block + doc[j:], encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
