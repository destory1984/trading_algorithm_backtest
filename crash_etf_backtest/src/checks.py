"""SPEC 5: the eight checks. Run after run / monkey / overfit. Usage: python -m src.checks -> results/checks.md"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import common as C
from src import overfit as O
from src.run import find_trades, book, basket, run_variant

KNOWN_MONTHS = {"1997-10": "아시아 위기", "1998-02": "아시아 위기 반등", "1998-09": "러시아·LTCM, 말레이시아 자본 통제",
                "2008-09": "금융위기", "2008-10": "금융위기", "2016-06": "브렉시트 투표", "2017-05": "브라질 대통령 녹취 파문",
                "2020-03": "코로나"}


def reference(fr: pd.DataFrame, thr: float, cfg: dict, start: str, min_dv: float, cost: float) -> pd.DataFrame:
    """Plain day-by-day loop written separately from run.find_trades."""
    r = cfg["rule"]
    o, c, tr = fr["open"].to_numpy(), fr["close"].to_numpy(), fr["tradable"].to_numpy()
    ma = c.copy() * np.nan
    for i in range(r["ma"] - 1, len(c)):
        ma[i] = c[i - r["ma"] + 1:i + 1].sum() / r["ma"]
    dv = fr["dv"].to_numpy()
    first = int(fr.index.searchsorted(pd.Timestamp(start)))
    rows, holding, want_buy, sell_open, e = [], False, False, False, -1

    def close_out(i: int, price: float, kind: str) -> None:
        rows.append({"entry_day": fr.index[e], "exit_day": fr.index[i], "kind": kind,
                     "ret": price * (1 - cost) / (o[e] * (1 + cost)) - 1})

    for i in range(len(c)):
        sold_at_close = False
        if holding and sell_open and tr[i]:
            close_out(i, o[i], "ma")
            holding = sell_open = False
        if not holding and want_buy:
            want_buy = False
            if tr[i] and i >= first:
                holding, e = True, i
        if holding and not sell_open:
            if i - e + 1 >= r["max_hold"]:
                if tr[i]:
                    close_out(i, c[i], "max")
                    holding, sold_at_close = False, True
            elif c[i] >= ma[i]:
                sell_open = True
        if not holding and not sold_at_close and i >= first - 1:
            if not np.isnan(ma[i]) and c[i] / ma[i] <= thr and dv[i] >= min_dv:
                want_buy = True
    if holding:
        close_out(len(c) - 1, c[-1], "forced")
    return pd.DataFrame(rows)


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    r, cost, start, floor = cfg["rule"], cfg["cost"], cfg["data"]["start"], cfg["rule"]["min_dollar_volume"]
    trades = pd.read_parquet(out / "trades.parquet")
    daily = pd.read_parquet(out / "daily.parquet")
    sl = pd.read_parquet(out / "sleeves.parquet")
    t95 = trades[trades["variant"] == "95"]
    tk = C.tickers(cfg)
    md, ok_all = ["# 검증", ""], []

    def head(n: int, title: str, ok: bool) -> None:
        ok_all.append(ok)
        md.extend([f"## {n}. {title}: {'통과' if ok else '실패'}", ""])

    # 1 seal
    last = pd.Timestamp(cfg["data"]["end"])
    file_last = max(C.raw(cfg, t).index[-1] for t in tk)
    ok = file_last <= last and daily.index[-1] <= last and trades["exit_day"].max() <= last
    head(1, "봉인", ok)
    md += [f"- 불러온 가격의 마지막 날 {file_last.date()}, 곡선 마지막 날 {daily.index[-1].date()}, "
           f"거래 마지막 청산일 {trades['exit_day'].max().date()} (봉인선 {last.date()})", ""]

    # 2 no future data
    rng = np.random.default_rng(cfg["checks"]["seed"])
    k, changed, compared = cfg["checks"]["perturb_days"], 0, 0
    for t in tk:
        raw = C.raw(cfg, t)
        p = raw.copy()
        f = rng.uniform(0.8, 1.2, k)
        for col in ["open", "high", "low", "close", "adj_close"]:
            p.iloc[-k:, p.columns.get_loc(col)] = p[col].to_numpy()[-k:] * f
        cut = raw.index[-k]
        for th in [r["main"], *r["neighbors"]]:
            a, _, _ = book(C.frame(cfg, t), find_trades(C.frame(cfg, t), th, cfg, start, floor), cost)
            fp = C.frame(cfg, t, p)
            b, _, _ = book(fp, find_trades(fp, th, cfg, start, floor), cost)
            a, b = a[a["exit_day"] < cut].reset_index(drop=True), b[b["exit_day"] < cut].reset_index(drop=True)
            compared += len(a)
            changed += 0 if a.equals(b) else 1
    head(2, "미래 데이터 금지", changed == 0)
    md += [f"- 종목마다 마지막 {k}거래일 가격에 0.8 → 1.2 무작위 배수를 곱해 다시 돌렸다. 그 전에 끝난 거래 "
           f"{compared}건(3개 판본) 중 달라진 묶음 {changed}개.", ""]

    # 3 hand check from the raw files
    rows, bad = [], 0
    for t in cfg["checks"]["hand_tickers"]:
        raw = C.raw(cfg, t)
        adj = raw["adj_close"]
        ao = raw["open"] * raw["adj_close"] / raw["close"]
        tt = t95[t95["ticker"] == t]
        for kind in ["ma", "max"]:
            g = tt[tt["kind"] == kind]
            tr = g.iloc[len(g) // 2]
            s, e = raw.index.get_loc(tr["signal_day"]), raw.index.get_loc(tr["entry_day"])
            disp = adj.iloc[s] / adj.iloc[s - r["ma"] + 1:s + 1].mean()
            x, price = None, None
            for j in range(e, len(raw)):
                if j - e + 1 >= r["max_hold"]:
                    x, price = j, adj.iloc[j]
                    break
                if adj.iloc[j] >= adj.iloc[j - r["ma"] + 1:j + 1].mean():
                    x, price = j + 1, ao.iloc[j + 1]
                    break
            ret = price * (1 - cost) / (ao.iloc[e] * (1 + cost)) - 1
            good = (e == s + 1 and disp <= r["main"] and raw.index[x] == tr["exit_day"] and abs(ret - tr["ret"]) < 1e-12
                    and abs(disp - tr["disp"]) < 1e-12 and raw["volume"].iloc[e] > 0 and raw["volume"].iloc[x] > 0)
            bad += not good
            rows.append({"종목": t, "청산": "25일선 회복" if kind == "ma" else "20일째", "신호일": tr["signal_day"].date(),
                         "이격도": f"{disp:.4f}", "진입일": tr["entry_day"].date(), "진입가": f"{ao.iloc[e]:.4f}",
                         "청산일": raw.index[x].date(), "청산가": f"{price:.4f}", "수익(다시 계산)": f"{ret:+.6f}",
                         "수익(기록)": f"{tr['ret']:+.6f}", "일치": "O" if good else "X"})
    head(3, "손 검산", bad == 0)
    md += ["원래 가격 파일에서 이격도, 진입가(시가 × adj ÷ 종가), 청산일, 청산가, 비용을 다시 계산했다.", "",
           C.md_table(pd.DataFrame(rows)), ""]

    # 4 reference loop
    rows, bad = [], 0
    for t in cfg["checks"]["ref_tickers"]:
        fr = C.frame(cfg, t)
        for th in [r["main"], *r["neighbors"]]:
            a, _, _ = book(fr, find_trades(fr, th, cfg, start, floor), cost)
            b = reference(fr, th, cfg, start, floor, cost)
            same = (len(a) == len(b) and (a["entry_day"].to_numpy() == b["entry_day"].to_numpy()).all()
                    and (a["exit_day"].to_numpy() == b["exit_day"].to_numpy()).all()
                    and (a["kind"].to_numpy() == b["kind"].to_numpy()).all()
                    and np.abs(a["ret"].to_numpy() - b["ret"].to_numpy()).max() < 1e-12)
            bad += not same
            rows.append({"종목": t, "이격도": round(th * 100), "본 코드 거래": len(a), "반복문 거래": len(b), "일치": "O" if same else "X"})
    head(4, "단순 반복문 대조", bad == 0)
    md += [C.md_table(pd.DataFrame(rows)), ""]

    # 5 money
    worst, overlap, novol = 0.0, 0, 0
    for t in tk:
        tt = t95[t95["ticker"] == t].sort_values("entry_day")
        worst = max(worst, abs(float((1 + sl[t]).prod()) - float((1 + tt["ret"]).prod())))
        overlap += int((tt["entry_day"].to_numpy()[1:] < tt["exit_day"].to_numpy()[:-1]).sum())
        vol = C.raw(cfg, t)["volume"]
        nf = tt[tt["kind"] != "forced"]
        novol += int((vol.reindex(tt["entry_day"]).to_numpy() <= 0).sum() + (vol.reindex(nf["exit_day"]).to_numpy() <= 0).sum())
    bdiff = float((basket(sl) - daily["95"]).abs().max())
    eq_diff = abs(float((1 + daily["95"]).prod()) - float((1 + sl).prod().mean()))
    res = run_variant(cfg, r["main"])
    rerun = float((res["basket"] - daily["95"]).abs().max())
    ok = worst < 1e-9 and overlap == 0 and novol == 0 and bdiff < 1e-12 and eq_diff < 1e-9 and rerun == 0.0
    head(5, "돈 계산", ok)
    md += [f"- 칸의 마지막 자산과 그 칸 거래 수익률 곱의 최대 차이 {worst:.1e} (19칸)",
           f"- 바구니 마지막 자산과 칸 평균의 차이 {eq_diff:.1e}, 일별 최대 차이 {bdiff:.1e}, 다시 돌린 곡선과의 차이 {rerun:.1e}",
           f"- 같은 칸에서 겹친 거래 {overlap}건, 거래량 0 인 날 체결 {novol}건, 끝까지 열려 강제로 닫은 거래 {(t95['kind'] == 'forced').sum()}건", ""]

    # 6 random entries
    rc = pd.read_csv(out / "random_check.csv").iloc[0]
    ok = rc["overlap"] == 0 and rc["not_buyable"] == 0 and rc["short"] == 0 and rc["placed"] == rc["runs"] * rc["rule_trades"]
    head(6, "무작위 진입", ok)
    md += [f"- {int(rc['runs'])}번 × 규칙 거래 {int(rc['rule_trades'])}건 = 놓은 거래 {int(rc['placed'])}건. 살 수 없는 날 진입 "
           f"{int(rc['not_buyable'])}건, 겹침 {int(rc['overlap'])}건, 못 놓은 거래 {int(rc['short'])}건.", ""]

    # 7 DSR by hand
    d = pd.read_csv(out / "dsr.csv").set_index("variant").loc[95]
    x = t95.groupby("entry_day")["ret"].mean().to_numpy()
    sr = x.mean() / x.std(ddof=1)
    z3 = (1 - O.EULER) * 0.430727 + O.EULER * 1.162084          # Φ⁻¹(2/3), Φ⁻¹(1 - 1/(3e)) from a normal table
    sr0 = math.sqrt(d["var_sr"]) * z3
    z = (sr - sr0) * math.sqrt(len(x) - 1) / math.sqrt(1 - d["skew"] * sr + (d["kurt"] - 1) / 4 * sr ** 2)
    hand = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    ok = abs(hand - d["dsr"]) < 1e-4
    head(7, "Deflated Sharpe 손계산", ok)
    md += [f"- 진입일 {len(x)}개, 진입일 샤프 {sr:.5f}, 왜도 {d['skew']:.3f}, 첨도 {d['kurt']:.3f}, V {d['var_sr']:.6f}",
           f"- 기대 최대 z(N = 3) = 0.4228 × 0.4307 + 0.5772 × 1.1621 = {z3:.4f}, SR0 = √V × z = {sr0:.5f}",
           f"- z = {z:.4f}, DSR = {hand:.5f} (코드 {d['dsr']:.5f})", ""]

    # 8 big moves
    mv = pd.read_csv(out / "big_moves.csv", parse_dates=["date"])
    mv["month"] = mv["date"].dt.strftime("%Y-%m")
    mv["event"] = mv["month"].map(KNOWN_MONTHS)
    mv["split_like"] = (mv["ret"] - mv["raw_ret"]).abs() > 1e-3
    unknown = mv[mv["event"].isna()]
    ok = not mv["split_like"].any()
    head(8, "큰 움직임", ok)
    md += [f"- 하루 ±{cfg['checks']['big_move']:.0%} 넘게 움직인 날 {len(mv)}개. 원래 종가와 배당 반영 종가의 움직임이 다른 날(분할·병합 오류 의심) "
           f"{int(mv['split_like'].sum())}개.",
           f"- 알려진 사건의 달에 든 날 {len(mv) - len(unknown)}개: " + ", ".join(f"{m} {e} {int((mv['month'] == m).sum())}개" for m, e in KNOWN_MONTHS.items() if (mv["month"] == m).any()),
           "- 사건을 확인하지 못한 날: " + (", ".join(f"{a} {b.date()} {c:+.1%}(거래량 {int(v):,}주)" for a, b, c, v in zip(unknown["ticker"], unknown["date"], unknown["ret"], unknown["volume"])) or "없음"), ""]

    md[1:1] = ["", f"{len(ok_all)}개 중 {sum(ok_all)}개 통과.", ""]
    (out / "checks.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
