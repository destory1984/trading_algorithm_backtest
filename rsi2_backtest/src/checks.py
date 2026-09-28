"""Verification checks. Writes results/checks.md (full run only).

Every check returns (passed, markdown lines) and must be able to fail: a missing input row is a failure, never a pass.
Tasks add checks above main(); modules created by later tasks are imported inside the check that needs them.

Usage:  python -m src.checks                 all checks, writes results/checks.md
        python -m src.checks --only a b      only these, prints only
Exit code 1 when a check fails.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Callable

import krxbt
import numpy as np
import pandas as pd

from .common import ROOT, load_config, one, results_dir

CHECKS: dict[str, Callable[[dict], tuple[bool, list[str]]]] = {}


def check(fn):
    CHECKS[fn.__name__] = fn
    return fn


@check
def engine_pinned(cfg: dict) -> tuple[bool, list[str]]:
    """The engine is the repo's krx_backtest_core, version 0.2.1, unchanged since planning."""
    core = ROOT.parent / "krx_backtest_core"
    here = Path(krxbt.__file__).resolve().parent.parent == core.resolve()
    ver = importlib.metadata.version("krxbt")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(core)], cwd=ROOT).returncode != 0
    last = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(core)], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    ok = here and ver == "0.2.1" and not dirty and last == cfg["checks"]["engine_commit"]
    return ok, [f"- krxbt 위치가 모음 저장소 krx_backtest_core: {here}", f"- 버전: {ver}",
                f"- 작업 트리 변경: {dirty}", f"- 마지막 변경 커밋: {last} (기대 {cfg['checks']['engine_commit']})"]


@check
def config_fixed(cfg: dict) -> tuple[bool, list[str]]:
    """Grid values as in the spec (216 combos), trials 768 + 48 + 216, rule constants, tax rules."""
    g = cfg["grid"]
    want = {"tickers": ["SPY", "QQQ", "IWM", "DIA"], "rsi_thr": [5, 10, 15], "trend": ["above", "none", "below"],
            "exit": ["green2", "sma5", "rsi70"], "exec": ["open", "close"], "costs": ["us", "orig"]}
    n = len(g["tickers"]) * len(g["rsi_thr"]) * len(g["trend"]) * len(g["exit"]) * len(g["exec"])
    tr = cfg["overfit"]["trials"]
    o, t = cfg["original"], cfg["tax"]
    ok = (g == want and n == 216 and tr == {"ibs_backtest": 768, "ibs_lev_backtest": 48, "rsi2_backtest": 216}
          and sum(tr.values()) == 1032 and tr["rsi2_backtest"] == n
          and cfg["rsi"] == {"period": 2, "exit_above": 70} and cfg["sma_exit"] == 5
          and (o["rsi_thr"], o["sma"], o["slippage"], o["is_frac"]) == (15, 200, 0.0005, 0.70)
          and cfg["us"]["costs"] == {"buy_fee": 0.0007, "sell_fee": 0.0007, "sell_tax": 0.0, "slippage": 0.0005}
          and cfg["walkforward"] == {"first_test_year": 2011, "min_train_trades": 30}
          and (cfg["ibs"]["entry"], cfg["ibs"]["exit"], cfg["ibs"]["exec"]) == (0.20, 0.80, "open")
          and (cfg["overfit"]["dsr_min"], cfg["overfit"]["monkey_runs"], cfg["overfit"]["monkey_top"]) == (0.95, 2000, 0.05)
          and (t["rate"], t["deduction_krw"], t["fx"], t["pay_month"]) == (0.22, 2500000, 1350, 5))
    return ok, [f"- grid: {g}", f"- 조합 수: {n}", f"- 시행 횟수: {tr} = {sum(tr.values())}",
                f"- 원본 규칙: RSI < {o['rsi_thr']}, SMA {o['sma']}, 편도 {o['slippage']}, 학습 {o['is_frac']}",
                f"- 세금: {t['rate']}, 공제 {t['deduction_krw']:,}원, 환율 {t['fx']}, {t['pay_month']}월 납부"]


@check
def data_files(cfg: dict) -> tuple[bool, list[str]]:
    """The four ETFs are in the us_data watchlist, start on checks.first_day, end on data.end, and every day is valid."""
    from krxbt import us as kus
    wl = (kus.us_dir(cfg) / "watchlist.txt").read_text(encoding="utf-8").split()
    cal, idx = kus.load_calendar(cfg), kus.index_features(cfg)
    end, first = pd.Timestamp(cfg["data"]["end"]), pd.Timestamp(cfg["checks"]["first_day"])
    lines = ["| 종목 | watchlist | 첫날 | 마지막 날 | 거래일 | 값 없는 날 |", "|---|---|---|---|---|---|"]
    ok = True
    for t in cfg["grid"]["tickers"]:
        f = kus.ticker_frame(cfg, t, cal, idx)
        if f is None:
            ok = False
            lines.append(f"| {t} | {t in wl} | 없음 | | | |")
            continue
        f = f[f.index <= end]
        bad = int((~f["valid"]).sum())
        ok &= t in wl and f.index[0] == first and f.index[-1] == end and bad == 0
        lines.append(f"| {t} | {t in wl} | {f.index[0].date()} | {f.index[-1].date()} | {len(f):,} | {bad} |")
    return ok, lines


PARTS_ORDER = ("ALL", "IS", "OOS")


@check
def stage0_reported(cfg: dict) -> tuple[bool, list[str]]:
    """Our loop on the original CSV against the spec's table (the original README): trades exact, every other number
    within half a unit of the reported digit; sample and split dates as the original's report."""
    rp = pd.read_csv(results_dir() / "replicate.csv")
    o = cfg["original"]
    rep, dig = o["reported"], o["reported_digits"]
    lines = ["| 구간 | 항목 | 보고치 | 우리 | 차이 |", "|---|---|---|---|---|"]
    ok = True
    for part, want in rep.items():
        r = one(rp, variant="original", part=part)
        if r is None:
            ok = False
            lines.append(f"| {part} | 행 없음 | | | |")
            continue
        ok &= int(r["trades"]) == want["trades"]
        lines.append(f"| {part} | trades | {want['trades']} | {int(r['trades'])} | {int(r['trades']) - want['trades']} |")
        for k, d in dig.items():
            diff = float(r[k]) - want[k]
            ok &= abs(diff) <= 0.5 * 10 ** -d + 1e-12
            lines.append(f"| {part} | {k} | {want[k]} | {float(r[k]):.{d + 2}f} | {diff:+.{d + 2}f} |")
    a, i, s = (one(rp, variant="original", part=p) for p in PARTS_ORDER)
    dates_ok = (a is not None and i is not None and s is not None and a["first"] == o["first_day"]
                and a["last"] == o["last_day"] and int(a["days"]) == o["bars"] and i["last"] == o["is_last"]
                and s["first"] == o["oos_first"])
    ok &= dates_ok
    return ok, lines + ["", f"- 표본 {o['first_day']} → {o['last_day']} ({o['bars']}일), 학습 끝 {o['is_last']}, "
                            f"표본 밖 시작 {o['oos_first']}: {'같음' if dates_ok else '다름'}"]


def stage0_meta() -> dict[str, str]:
    m = pd.read_csv(results_dir() / "stage0_meta.csv", dtype=str)
    return dict(zip(m["key"], m["value"]))


@check
def stage0_rerun(cfg: dict) -> tuple[bool, list[str]]:
    """The original re-run on its own CSV gives the shipped metricas.json fase1 numbers and the same trades.csv;
    the ref repo is at the planned commit; the re-run really happened (a new `generado` date)."""
    import json
    from .stage0 import head_commit, rerun_text, same_fase1, shipped_text
    sj = json.loads(shipped_text(cfg, "resultados/metricas.json"))
    rj = json.loads(rerun_text(cfg, "resultados/metricas.json"))
    s_tr, r_tr = shipped_text(cfg, "resultados/trades.csv"), rerun_text(cfg, "resultados/trades.csv")
    head = head_commit(cfg)
    fresh = rj["meta"]["generado"] != sj["meta"]["generado"]
    same = same_fase1(sj, rj)
    rows = r_tr.strip().count("\n")
    ok = head == cfg["original"]["commit"] and fresh and same and s_tr == r_tr and rows == cfg["original"]["reported"]["ALL"]["trades"]
    return ok, [f"- 원본 커밋 {head} (기대 {cfg['original']['commit']})",
                f"- metricas.json 생성일: 저장소 {sj['meta']['generado']}, 재실행 {rj['meta']['generado']} (달라야 재실행한 것)",
                f"- fase1 IS/OOS/TODO 수치 같음: {same}", f"- trades.csv 같음: {s_tr == r_tr}, 거래 {rows}건"]


@check
def stage0_ours(cfg: dict) -> tuple[bool, list[str]]:
    """Our loop equals the original re-run: 287 trades with the same entry and exit dates, returns within the
    trades.csv rounding (5 decimals), every fase1 metric within 1e-9, same sample and split dates."""
    rd = results_dir()
    s0 = pd.read_csv(rd / "stage0.csv")
    st = pd.read_csv(rd / "stage0_trades.csv")
    meta = stage0_meta()
    cols = ["trades", "win_rate", "profit_factor", "pf_ex_best", "expectancy", "cagr", "mdd", "exposure"]
    lines = ["| 구간 | 최대 차이 (우리 - 재실행) |", "|---|---|"]
    ok = True
    for p in ("IS", "OOS", "ALL"):
        a, b = one(s0, source="ours", part=p), one(s0, source="rerun", part=p)
        if a is None or b is None:
            ok = False
            lines.append(f"| {p} | 행 없음 |")
            continue
        worst = max(abs(float(a[c]) - float(b[c])) for c in cols)
        ok &= worst < 1e-9
        lines.append(f"| {p} | {worst:.1e} |")
    n_orig, n_ours = int(st["orig_entry"].notna().sum()), int(st["our_entry"].notna().sum())
    same, retd = int(st["same_dates"].sum()), float(st["ret_diff"].max())
    want = cfg["original"]["reported"]["ALL"]["trades"]
    ok &= n_orig == n_ours == same == want and retd <= 5e-6 + 1e-12
    rp = pd.read_csv(rd / "replicate.csv")
    a, i, s = one(rp, variant="original", part="ALL"), one(rp, variant="original", part="IS"), one(rp, variant="original", part="OOS")
    dates = (a is not None and i is not None and s is not None and a["first"] == meta["first"] and a["last"] == meta["last"]
             and int(a["days"]) == int(meta["bars"]) and i["last"] == meta["is_last"] and s["first"] == meta["oos_first"])
    ok &= dates
    return ok, lines + ["", f"- 거래: 원본 {n_orig}, 우리 {n_ours}, 진입·청산일 같음 {same}, 수익률 최대 차이 {retd:.1e}",
                        f"- 표본·분할 날짜가 원본 meta 와 같음: {dates}"]


@check
def rsi_digits(cfg: dict) -> tuple[bool, list[str]]:
    """RSI(2) equals the original code's on every day of its sample to 4 decimals; SMA200 and SMA5 to 1e-9."""
    from .replicate import load_csv, with_indicators
    oi = pd.read_csv(results_dir() / "stage0" / "orig_indicators.csv", index_col=0, parse_dates=True)
    ind = with_indicators(cfg, load_csv(cfg)).reindex(oi.index)
    a, b = ind["rsi"].to_numpy(np.float64), oi["rsi2"].to_numpy(np.float64)
    mism = int((np.round(a, 4) != np.round(b, 4)).sum())
    d = float(np.nanmax(np.abs(a - b)))
    sd = max(float(np.nanmax(np.abs(ind["ma"].to_numpy() - oi["sma"].to_numpy()))),
             float(np.nanmax(np.abs(ind["sma5"].to_numpy() - oi["ma5"].to_numpy()))))
    ok = (len(oi) == cfg["original"]["bars"] and mism == 0 and bool(np.isfinite(a).all()) and sd < 1e-9
          and int(stage0_meta()["rsi_mismatch_4dp"]) == mism)
    return ok, [f"- 비교한 날 {len(oi):,} (기대 {cfg['original']['bars']:,}), 소수 넷째 자리에서 다른 날 {mism}, 최대 차이 {d:.1e}",
                f"- 200일선·5일선 최대 차이 {sd:.1e}"]


@check
def green_raw(cfg: dict) -> tuple[bool, list[str]]:
    """green is raw close > raw open: rebuilt here from krxbt.us.load_prices (unadjusted) and compared with the green
    column the rule uses; raw open == raw close is never green. The count of days where the adjusted comparison differs
    is information only (0 on the current data).
    Raw prices are used so that a future adjustment-factor change cannot flip flat days."""
    from krxbt import us as kus
    from .strategy import frames
    lines = ["| 종목 | 시가 = 종가인 날 | 원가격 양봉과 다른 날 | 조정가로 보면 다른 날 |", "|---|---|---|---|"]
    ok = True
    for t, f in frames(cfg, cfg["grid"]["tickers"]).items():
        px = kus.load_prices(cfg, t)
        px = px[~px.index.duplicated(keep="last")].reindex(f.index)
        have = bool(px[["open", "close"]].notna().all().all())
        raw = (px["close"] > px["open"]).to_numpy()
        flat = (px["close"] == px["open"]).to_numpy()
        adj = (f["close"] > f["open"]).to_numpy()
        g = f["green"].to_numpy(bool)
        off = int((g != raw).sum())
        ok &= have and off == 0 and not g[flat].any()
        lines.append(f"| {t} | {int(flat.sum())} | {off} | {int((adj != raw).sum())} |")
    return ok, lines


@check
def sma_warmup(cfg: dict) -> tuple[bool, list[str]]:
    """No above/below signal before the SMA200 exists; the no-filter version signals on RSI alone, also before it
    (checked on the whole original CSV, before the sample trim, and on the engine arrays)."""
    from .indicators import signals
    from .replicate import load_csv, with_indicators
    from .strategy import arrays, frames, rule
    ind = with_indicators(cfg, load_csv(cfg))
    arr = {k: ind[k].to_numpy(np.float64) for k in ("rsi", "close", "ma", "sma5")}
    arr["green"] = ind["green"].to_numpy(bool)
    elig = np.ones(len(ind), bool)
    thr, rx = cfg["original"]["rsi_thr"], cfg["rsi"]["exit_above"]
    no_ma = ~np.isfinite(arr["ma"])
    lines, ok = [], True
    for trend in ("above", "below"):
        c, _ = signals(arr, thr, trend, "green2", elig, rx)
        n = int((c & no_ma).sum())
        ok &= n == 0
        lines.append(f"- 원본 CSV, {trend}: 200일선 전 신호 {n}")
    c, _ = signals(arr, thr, "none", "green2", elig, rx)
    rsi_only = np.isfinite(arr["rsi"]) & (np.nan_to_num(arr["rsi"], nan=np.inf) < thr)
    pre = int((c & no_ma).sum())
    ok &= np.array_equal(c, rsi_only) and pre > 0
    lines.append(f"- 원본 CSV, none: RSI 조건과 같음 {np.array_equal(c, rsi_only)}, 200일선 전 신호 {pre} (0 보다 커야 예외가 실제로 적용된 것)")
    for t, f in frames(cfg, cfg["grid"]["tickers"]).items():
        # eligible = valid only: the frame's own eligible already waits for the SMA200, which would hide a missing
        # trend-filter guard
        a = {**arrays(f), "eligible": arrays(f)["valid"]}
        pre_ma = ~np.isfinite(a["ma"])
        for trend in ("above", "below"):
            cc, _ = rule(a, cfg, thr, trend, "green2")
            n = int((cc & pre_ma).sum())
            ok &= n == 0
            lines.append(f"- {t} 엔진 배열(eligible = valid), {trend}: 200일선 전 신호 {n}")
        cc, _ = rule(a, cfg, thr, "none", "green2")
        n = int((cc & pre_ma).sum())
        ok &= n > 0
        lines.append(f"- {t} 엔진 배열(eligible = valid), none: 200일선 전 신호 {n} (0 보다 커야 함)")
    return ok, lines


@check
def engine_match(cfg: dict) -> tuple[bool, list[str]]:
    """Engine == plain loop from data.start on both price files and both quirk variants: same dates, returns to 1e-12."""
    from .replicate import VARIANTS
    em = pd.read_csv(results_dir() / "engine_match.csv")
    need = {(s, v) for s in ("csv", "us") for v in VARIANTS}
    have = set(zip(em["source"], em["variant"]))
    ok = (need == have and len(em) == len(need) and bool((em["mismatched"] == 0).all())
          and float(em["max_ret_diff"].max()) < 1e-12 and bool((em["loop_trades"] > 0).all()))
    lines = ["| 가격 | 판본 | 반복문 | 엔진 | 다른 거래 | 수익률 최대 차이 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.source} | {r.variant} | {r.loop_trades} | {r.engine_trades} | {r.mismatched} | {r.max_ret_diff:.1e} |"
              for r in em.itertuples()]
    return ok, lines


@check
def quirk(cfg: dict) -> tuple[bool, list[str]]:
    """The engine counts the signal day's candle: every trade that exits on the day after entry by green2 has a green
    signal day and a green entry day; the no_signal_day variant never exits before the second day after entry; the
    stage-0 loop shows the same split."""
    from .common import with_costs
    from .strategy import arrays, frames, rule, run_rule
    o = cfg["original"]
    f = frames(cfg, ["QQQ"])["QQQ"]
    a = arrays(f)
    ix = pd.DatetimeIndex(a["dates"])
    ocfg = with_costs(cfg, "orig")
    cand, target = rule(a, cfg, o["rsi_thr"], "above", "green2")
    t0, t1 = run_rule(a, ocfg, cand, target, "open", 0), run_rule(a, ocfg, cand, target, "open", 1)
    q0 = t0[(t0["hold_days"] == 2) & (t0["reason"] == "target")]
    s, e = ix.get_indexer(q0["signal_date"]), ix.get_indexer(q0["entry_date"])
    used = int((a["green"][s] & a["green"][e]).sum())
    early1 = int(((t1["hold_days"] <= 2) & (t1["reason"] == "target")).sum())
    differ = not t0[["entry_date", "exit_date"]].reset_index(drop=True).equals(t1[["entry_date", "exit_date"]].reset_index(drop=True))
    rp = pd.read_csv(results_dir() / "replicate.csv")
    r0, r1 = one(rp, variant="original", part="ALL"), one(rp, variant="no_signal_day", part="ALL")
    loop_ok = r0 is not None and r1 is not None and int(r0["signal_day_exits"]) > 0 and int(r1["signal_day_exits"]) == 0
    ok = len(q0) > 0 and used == len(q0) and early1 == 0 and differ and loop_ok
    return ok, [f"- us_data QQQ 원본 판본: 진입 다음 날 판 거래 {len(q0)}건, 그중 신호일·진입일이 모두 양봉 {used}건",
                f"- 신호일 뺀 판본: 진입 다음 날 판 거래 {early1}건 (0 이어야 함), 두 판본 거래 목록이 다름: {differ}",
                f"- 단계 0 반복문 전체 구간: 원본 {'' if r0 is None else int(r0['signal_day_exits'])}건, "
                f"신호일 뺌 {'' if r1 is None else int(r1['signal_day_exits'])}건"]


def grid_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    from .strategy import arrays, frames
    rd = results_dir()
    g = pd.read_csv(rd / "grid.csv")
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    fr = frames(cfg, cfg["grid"]["tickers"])
    return g, tr, fr, {t: arrays(f) for t, f in fr.items()}


@check
def grid_vs_loop(cfg: dict) -> tuple[bool, list[str]]:
    """Every one of the 432 grid rows (216 combos x 2 cost sets) equals the plain loop: same dates, forced flags, returns
    to 1e-12; no close-execution trade leaves on its entry day by the exit rule."""
    from .common import cost_set
    from .replicate import compare_trades, loop_trades
    from .strategy import rule
    g, tr, _, ar = grid_inputs(cfg)
    bad, worst = 0, 0.0
    for r in g.itertuples():
        a = ar[r.ticker]
        cand, target = rule(a, cfg, r.rsi_thr, r.trend, r.exit)
        ref = loop_trades(a["dates"], a["open"], a["close"], cand, target, r.exec, cost_set(cfg, r.costs), 0)
        eng = tr[(tr["combo"] == r.combo) & (tr["costs"] == r.costs)].reset_index(drop=True)
        b, d = compare_trades(ref, eng)
        bad += b
        worst = max(worst, d)
    same_day = int(((tr["exit_date"] == tr["entry_date"]) & (tr["reason"] == "target")).sum())
    pairs = set(zip(g["combo"], g["costs"]))
    ok = len(g) == 432 and len(pairs) == 432 and bad == 0 and worst < 1e-12 and same_day == 0
    return ok, [f"- 행 {len(g)} (기대 432), 반복문과 다른 거래 {bad}, 수익률 최대 차이 {worst:.1e}",
                f"- 진입일에 청산 규칙으로 판 거래 {same_day} (0 이어야 함)"]


@check
def entry_exit_days(cfg: dict) -> tuple[bool, list[str]]:
    """open: entry = the trading day after the signal, exit = the trading day after the first day (from the entry day)
    the exit condition holds. close: entry = signal day, exit = the first day after entry the condition holds.
    Only target and forced exits exist."""
    from .strategy import rule
    g, tr, _, ar = grid_inputs(cfg)
    g, tr = g[g["costs"] == "us"], tr[tr["costs"] == "us"]
    bad_entry = bad_exit = bad_first = n = 0
    bad_reason = int((~tr["reason"].isin(["target", "forced"])).sum())
    for r in g.itertuples():
        a = ar[r.ticker]
        ix = pd.DatetimeIndex(a["dates"])
        _, target = rule(a, cfg, r.rsi_thr, r.trend, r.exit)
        x = tr[tr["combo"] == r.combo]
        n += len(x)
        s, e = ix.get_indexer(x["signal_date"]), ix.get_indexer(x["entry_date"])
        bad_entry += int((e != (s + 1 if r.exec == "open" else s)).sum())
        t = x[x["reason"] == "target"]
        e, xx = ix.get_indexer(t["entry_date"]), ix.get_indexer(t["exit_date"])
        cond = xx - 1 if r.exec == "open" else xx
        first = e if r.exec == "open" else e + 1
        bad_exit += int((~target[cond]).sum() + (cond < first).sum())
        cs = np.concatenate([[0], np.cumsum(target)])
        bad_first += int(((cs[cond] - cs[first]) > 0).sum())
    ok = n > 0 and bad_entry == bad_exit == bad_first == bad_reason == 0 and len(g) == 216
    return ok, [f"- 조합 {len(g)} (기대 216, us 비용만), 거래 {n:,}건", f"- 진입일이 규칙과 다름 {bad_entry}",
                f"- 청산 조건이 성립하지 않은 날에 판 거래 {bad_exit}",
                f"- 더 이른 청산 조건을 건너뛴 거래 {bad_first}", f"- target/forced 가 아닌 청산 {bad_reason}"]


@check
def forced_at_end(cfg: dict) -> tuple[bool, list[str]]:
    """A trade open at the end is the combo's last trade, closed at data.end's close, and flagged in grid.csv."""
    g, tr, fr, _ = grid_inputs(cfg)
    bad = n_forced = 0
    for r in g.itertuples():
        x = tr[(tr["combo"] == r.combo) & (tr["costs"] == r.costs)]
        fz = x[x["reason"] == "forced"]
        n_forced += len(fz)
        f = fr[r.ticker]
        bad += int(len(fz) > 1)
        if len(fz):
            bad += int(fz.index[-1] != x.index[-1])
            bad += int((fz["exit_date"] != f.index[-1]).any()) + int((fz["exit_px"] != f["close"].iloc[-1]).any())
        bad += int(bool(r.open_at_end) != (len(fz) > 0))
    ok = bad == 0 and len(g) == 432
    return ok, [f"- 행 {len(g)} (기대 432)",
                f"- 끝에 열린 거래 {n_forced}건 ({int(g['open_at_end'].sum())}개 행에 표시), 규칙과 다른 곳 {bad}"]


@check
def equity_product(cfg: dict) -> tuple[bool, list[str]]:
    """On exit days and flat days, equity = capital x prod(1 + trade returns so far), to the cent, for the 216 us-cost
    combos; run_portfolio skipped no trade."""
    rd = results_dir()
    cap = cfg["portfolio"]["initial_capital"]
    g = pd.read_csv(rd / "grid.csv")
    g = g[g["costs"] == "us"]
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"]
    daily = pd.read_parquet(rd / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    eq = (1 + daily).cumprod() * cap
    worst = 0.0
    for cid in g["combo"]:
        e = eq[cid]
        x = tr[tr["combo"] == cid].sort_values("exit_date")
        if not len(x):
            worst = max(worst, float((e - cap).abs().max()))
            continue
        book = cap * np.cumprod(1 + x["ret"].to_numpy())
        worst = max(worst, float(np.abs(e.loc[x["exit_date"]].to_numpy() - book).max()))
        held = np.zeros(len(e), bool)
        for en, ex in zip(e.index.get_indexer(x["entry_date"]), e.index.get_indexer(x["exit_date"])):
            held[en:ex] = True
        booked = pd.Series(book, x["exit_date"].to_numpy()).reindex(e.index).ffill().fillna(cap)
        worst = max(worst, float(np.abs(e[~held] - booked[~held]).max()))
    skipped = int((g["trades"] != g["taken"]).sum())
    return worst < 0.01 and skipped == 0 and len(g) == 216, [
        f"- 216개 조합, 청산일·포지션 없는 날 최대 차이 {worst:.6f} 달러", f"- 건너뛴 거래가 있는 조합 {skipped}"]


@check
def matched_holds(cfg: dict) -> tuple[bool, list[str]]:
    """kdd's drawdown equals the strategy's (or k = 1 and it is shallower: capped); hold CAGR equals the one computed
    from the first and last close with that row's costs.
    kexp is not checked against exposure directly (both come from the same stats() call in grid.py -- that would
    compare a value with itself). Instead: (a) kexp_cagr is recomputed here from cumprod(1 + exposure x r1) on the
    combo's own calendar and adjusted close -- the same formula as benchmarks.mix_curve/bench_row, but written
    independently, not called -- and compared with the saved column; (b) exposure itself is recomputed from the
    trades' entry/exit dates as the share of calendar days spanned by a position (entry day through exit day,
    inclusive). That is a different definition from common.in_position (which grid.py uses): a close-execution trade
    is bought at the signal day's close, so that day's return does not depend on the position yet and in_position
    excludes it, while the entry-to-exit span includes it. The two must therefore differ by exactly (close-execution
    trades in the combo) / (calendar days) -- no more, no less; open-execution rows must match exactly. A simpler
    "share of days with nonzero return" is not used for this: a day where the ticker's close does not move gives a
    zero return whether or not a position is held that day, so it would undercount exposure on flat-price days.
    """
    from krxbt.costs import cost_factors
    from .common import with_costs
    from .strategy import calendar
    g, tr, fr, _ = grid_inputs(cfg)
    capped = g["kdd_capped"].astype(str).eq("True")
    free, capd = g[~capped], g[capped]
    d_mdd = float((free["kdd_mdd"] - free["mdd"]).abs().max()) if len(free) else 0.0
    cap_ok = bool((capd["kdd"] == 1.0).all() and (capd["kdd_mdd"] >= capd["mdd"]).all())
    cal_cache: dict[str, tuple] = {}
    hold_worst = kexp_worst = 0.0
    exp_bad = 0
    exp_gap_worst = 0.0
    for r in g.itertuples():
        if r.ticker not in cal_cache:
            f = fr[r.ticker]
            cal = calendar(cfg, f)
            r1 = f["close"].reindex(cal).ffill().pct_change().fillna(0.0).to_numpy()
            ix = {d: i for i, d in enumerate(cal)}
            years = (cal[-1] - cal[0]).days / 365.25
            cal_cache[r.ticker] = (f, cal, r1, ix, years)
        f, cal, r1, ix, years = cal_cache[r.ticker]

        bc, sk = cost_factors(with_costs(cfg, r.costs))
        want_hold = (f["close"][cal[-1]] / f["close"][cal[0]] * sk / bc) ** (1 / years) - 1
        hold_worst = max(hold_worst, abs(r.hold_cagr - want_hold))

        eq2 = np.cumprod(1 + r.exposure * r1)
        want_kexp = float(eq2[-1] ** (1 / years) - 1)
        kexp_worst = max(kexp_worst, abs(r.kexp_cagr - want_kexp))

        x = tr[(tr["combo"] == r.combo) & (tr["costs"] == r.costs)]
        held = np.zeros(len(cal), bool)
        for e, xd in zip(x["entry_date"], x["exit_date"]):
            i0, i1 = ix.get(e), ix.get(xd)
            if i0 is None or i1 is None:
                continue
            held[i0:i1 + 1] = True
        span_exposure = float(held.mean())
        expected_gap = (len(x) / len(cal)) if r.exec == "close" else 0.0
        gap = span_exposure - r.exposure
        if gap < -1e-9 or gap > expected_gap + 1e-9:
            exp_bad += 1
        exp_gap_worst = max(exp_gap_worst, abs(gap - min(max(gap, 0.0), expected_gap)))
    ok = (len(g) == 432 and d_mdd < 1e-9 and cap_ok and hold_worst < 1e-12 and kexp_worst < 1e-9
          and exp_bad == 0)
    return ok, [f"- 행 {len(g)} (기대 432)",
                f"- 낙폭 맞춤: 상한에 안 걸린 {len(free)}행 최대 낙폭 차이 {d_mdd:.1e}, 상한(k = 1)에 걸린 {len(capd)}행 (규칙 맞음 {cap_ok})",
                f"- 보유 연 수익률 직접 계산과 최대 차이 {hold_worst:.1e}",
                f"- kexp 연 수익률을 exposure x 일별수익률로 독립 재계산, 저장값과 최대 차이 {kexp_worst:.1e}",
                f"- exposure 를 거래 진입일~청산일 구간(다른 정의, 종가 체결의 진입일 포함)으로 재계산: "
                f"기대한 범위를 벗어난 행 {exp_bad} (0 이어야 함), 범위 안에서 최대 벗어난 정도 {exp_gap_worst:.1e}"]


@check
def wf_no_peek(cfg: dict) -> tuple[bool, list[str]]:
    """The pick for year Y does not change when every daily return from Y on is replaced by noise and every trade
    entered from Y on is dropped; and it equals walkforward.csv's pick (variant all)."""
    from .walkforward import pick_years
    rd = results_dir()
    wf = cfg["walkforward"]
    g = pd.read_csv(rd / "grid.csv")
    g = g[g["costs"] == "us"].set_index("combo")
    daily = pd.read_parquet(rd / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"]
    wp = pd.read_csv(rd / "walkforward.csv")
    wp = wp[wp["variant"] == "all"]
    years = sorted(wp["year"].unique())
    if not years:
        return False, ["- walkforward.csv 에 all 판본 행이 없다"]
    rng = np.random.default_rng(1)
    lines = ["| 해 | 원래 고른 조합 | 미래를 바꾼 뒤 | walkforward.csv |", "|---|---|---|---|"]
    ok = True
    for Y in (years[0], years[len(years) // 2], years[-1]):
        start = pd.Timestamp(f"{Y}-01-01")
        base = pick_years(g, daily, tr, [Y], wf["min_train_trades"], g.index)
        d2 = daily.copy()
        fut = d2.index >= start
        d2.loc[fut] = rng.normal(0.0, 0.05, size=(int(fut.sum()), d2.shape[1]))
        alt = pick_years(g, d2, tr[tr["entry_date"] < start], [Y], wf["min_train_trades"], g.index)
        saved = one(wp, year=Y)
        b = int(base["combo"].iloc[0]) if len(base) else None
        a2 = int(alt["combo"].iloc[0]) if len(alt) else None
        s = int(saved["combo"]) if saved is not None else None
        ok &= b is not None and b == a2 == s and abs(float(base["train_sharpe"].iloc[0]) - float(alt["train_sharpe"].iloc[0])) < 1e-12
        lines.append(f"| {Y} | {b} | {a2} | {s} |")
    return ok, lines


@check
def mix_rules(cfg: dict) -> tuple[bool, list[str]]:
    """On the candidate's ticker: every OR trade taken is one of the two rules' trades, positions never overlap and a
    same-day tie goes to the earlier exit; the half portfolio's saved daily returns equal a curve built purely from
    the saved full-capital {t}_rsi2 / {t}_ibs return columns, each sleeve compounded from half the capital and summed
    (no equity()/run_portfolio call repeated here); the AND entries/exits, derived independently from the RSI and IBS
    rule arrays with their own loop (both signals true on the day, exit at the earlier of the two rules' independent
    exits), match and_trades' output; mix.csv has the candidate ticker's five rows. ibs.exec (not a hard-coded "open")
    gives the IBS rule's execution mode."""
    from .common import with_costs
    from .mix import NAMES, and_trades, or_trades
    from .strategy import arrays, equity, frames, ibs_rule, rule, run_rule
    rd = results_dir()
    c = one(pd.read_csv(rd / "candidates.csv"), variant="open")
    if c is None:
        return False, ["- candidates.csv 에 open 판본 후보가 없다"]
    t = c["ticker"]
    f = frames(cfg, [t])[t]
    a = arrays(f)
    ix = pd.DatetimeIndex(a["dates"])
    ucfg = with_costs(cfg, "us")
    ibs_exec = cfg["ibs"]["exec"]
    cr, trr = rule(a, cfg, c["rsi_thr"], c["trend"], c["exit"])
    ci, ti = ibs_rule(a, cfg)
    ind_r = run_rule(a, ucfg, cr, trr, c["exec"], independent=True)
    ind_i = run_rule(a, ucfg, ci, ti, ibs_exec, independent=True)

    # OR: every ledgered trade is one of the two rules' independent trades, no overlap, and same-day ties resolve to
    # the earlier exit (rank_by="rank" of or_trades).
    o = or_trades(ind_r, ind_i)
    eq, _ = equity(cfg, f, o, t, rank_by="rank")
    led = eq.attrs["ledger"]
    keys = set(zip(o["entry_date"], o["exit_date"], o["ret"]))
    bad_src = sum((e, x, r) not in keys for e, x, r in zip(led["entry_date"], led["exit_date"], led["ret"]))
    overlap = int((led["entry_date"].iloc[1:].to_numpy() <= led["exit_date"].iloc[:-1].to_numpy()).sum())
    first = o.sort_values("rank", kind="mergesort").groupby("entry_date").head(1)
    firsts = set(zip(first["entry_date"], first["exit_date"]))
    bad_tie = sum((e, x) not in firsts for e, x in zip(led["entry_date"], led["exit_date"]))

    # half: mix.main writes the saved {t}_half column from two half-capital equity() runs summed. Here we rebuild the
    # half curve a different way -- purely arithmetically from the saved full-capital {t}_rsi2 / {t}_ibs daily-return
    # columns, each compounded from half the capital and added -- with no equity()/run_portfolio call at all, so the
    # check cannot pass by repeating mix.main's own computation.
    cap = cfg["portfolio"]["initial_capital"]
    saved_daily = pd.read_parquet(rd / "mix_daily.parquet")
    d_rsi2, d_ibs = saved_daily[f"{t}_rsi2"], saved_daily[f"{t}_ibs"]
    half_from_cols = cap / 2 * (1 + d_rsi2).cumprod() + cap / 2 * (1 + d_ibs).cumprod()
    prev = half_from_cols.shift(1)
    prev.iloc[0] = cap
    d_half_built = half_from_cols / prev - 1
    d_half_saved = saved_daily[f"{t}_half"]
    d_half = float((d_half_built - d_half_saved).abs().max())

    # AND: derive entries/exits directly from the RSI and IBS rule arrays with an independent loop (both signals
    # true on the day; exit at the earlier of the two rules' independent exits for that signal day), compared with
    # and_trades' own output.
    exit_r = dict(zip(ind_r["signal_date"], ind_r["exit_date"]))
    exit_i = dict(zip(ind_i["signal_date"], ind_i["exit_date"]))
    both = cr & ci
    loop_pairs = set()
    for k in range(len(both)):
        if not both[k]:
            continue
        d0 = ix[k]
        xr, xi = exit_r.get(d0), exit_i.get(d0)
        if xr is None or xi is None:
            continue
        loop_pairs.add((d0, min(xr, xi)))
    an = and_trades(ind_r, ind_i)
    saved_pairs = set(zip(an["signal_date"], an["exit_date"]))
    bad_and = len(saved_pairs.symmetric_difference(loop_pairs))

    mx = pd.read_csv(rd / "mix.csv")
    rows_ok = all(one(mx, ticker=t, name=k) is not None for k in NAMES)
    ok = (len(led) > 0 and len(loop_pairs) > 0 and len(an) > 0 and bad_src == overlap == bad_tie == bad_and == 0
          and d_half < 1e-9 and rows_ok)
    return ok, [f"- 후보 {c['label']}, 종목 {t}",
                f"- OR: 들어간 거래 {len(led)}, 두 규칙에 없는 거래 {bad_src}, 겹친 포지션 {overlap}, 같은 날 더 늦게 파는 쪽을 고른 경우 {bad_tie}",
                f"- 반반: 저장된 두 규칙 일별수익률에서 독립적으로 쌓은 곡선과 저장된 half 열의 최대 차이 {d_half:.1e}",
                f"- AND: 규칙 배열에서 독립적으로 구한 후보 {len(loop_pairs)}, and_trades 결과 {len(an)}, 서로 다른 곳 {bad_and}",
                f"- mix.csv 후보 종목 다섯 행 있음: {rows_ok}"]


@check
def dsr_formula(cfg: dict) -> tuple[bool, list[str]]:
    """The paper's worked example (N = 100 trials, V = 0.5 annual, T = 1,250 days, annual SR 2.5, skew -3, kurt 10,
    250 days a year) gives DSR 0.9004 (swapping the Euler weights gives 0.912 and fails); more trials lower the DSR;
    overfit.csv has one primary row per candidate with N = 1,032, and per open/close candidate the three sensitivity
    rows: n216_v216 (N 216, same V as the primary, so a higher DSR), n1032_v1032 (V from 1,032 trial Sharpes, or NaN
    with a note), n1032_vqqq54 (V from 54 QQQ combos). Printed only (cannot fail): DSR = 0.5 when SR0 = SR."""
    from statistics import NormalDist

    from .overfit import EULER, deflated_sharpe, dsr_from_moments, expected_max_z
    sr0, dsr = dsr_from_moments(2.5 / np.sqrt(250), 1250, -3.0, 10.0, 100, 0.5 / 250)
    ex = abs(dsr - 0.9004) < 5e-4 and abs(sr0 * np.sqrt(250) - 1.7894) < 5e-3
    # the tolerance must reject the swapped Euler weights (computed here by hand, not through overfit.py)
    nd = NormalDist()
    zs = EULER * nd.inv_cdf(1 - 1 / 100) + (1 - EULER) * nd.inv_cdf(1 - 1 / (100 * np.e))
    sr_d, sr0s = 2.5 / np.sqrt(250), np.sqrt(0.5 / 250) * zs
    dsr_sw = nd.cdf((sr_d - sr0s) * np.sqrt(1249) / np.sqrt(1 + 3.0 * sr_d + 9.0 / 4 * sr_d ** 2))
    swap_rejected = abs(dsr_sw - 0.9004) >= 5e-4
    rng = np.random.default_rng(7)
    x = rng.normal(0.001, 0.01, 2000)
    sr = x.mean() / x.std(ddof=1)
    half = deflated_sharpe(x, 1032, (sr / expected_max_z(1032)) ** 2)["dsr"]
    mono = deflated_sharpe(x, 1000, 1e-4)["dsr"] < deflated_sharpe(x, 10, 1e-4)["dsr"]
    ov = pd.read_csv(results_dir() / "overfit.csv")
    rows = [one(ov, variant=v, primary=True) for v in ("all", "open", "close")]
    rows_ok = all(r is not None and int(r["n_trials"]) == 1032 for r in rows)
    lines = []
    sens_ok = True
    for v, p in zip(("open", "close"), rows[1:]):
        a, b, q = (one(ov, variant=f"{v}|{k}", primary=False) for k in ("n216_v216", "n1032_v1032", "n1032_vqqq54"))
        if p is None or a is None or b is None or q is None:
            sens_ok = False
            lines.append(f"- {v}: 민감도 행 없음")
            continue
        ok_a = int(a["n_trials"]) == 216 and float(a["var_sr"]) == float(p["var_sr"]) and float(a["dsr"]) > float(p["dsr"])
        ok_b = int(b["n_trials"]) == 1032 and ((int(b["v_trials"]) == 1032 and np.isfinite(float(b["dsr"])))
                                               or (not np.isfinite(float(b["dsr"])) and isinstance(b["note"], str)))
        ok_q = int(q["n_trials"]) == 1032 and int(q["v_trials"]) == 54 and np.isfinite(float(q["dsr"]))
        sens_ok &= ok_a and ok_b and ok_q
        lines.append(f"- {v}: 주 DSR {float(p['dsr']):.4f} | N 216 {float(a['dsr']):.4f} | V 1,032개 "
                     f"{float(b['dsr']):.4f} ({int(b['v_trials'])}개) | V QQQ 54개 {float(q['dsr']):.4f}: "
                     f"{ok_a and ok_b and ok_q}")
    ok = ex and swap_rejected and mono and rows_ok and sens_ok
    return ok, [f"- 논문 예: SR0(연) {sr0 * np.sqrt(250):.4f} (기대 1.7894), DSR {dsr:.4f} (기대 0.9004)",
                f"- 오일러 가중치를 바꾸면 DSR {dsr_sw:.4f}, 허용 오차 밖: {swap_rejected}",
                f"- 시행이 많으면 DSR 이 작아짐: {mono}",
                f"- overfit.csv 세 후보 주 행(primary), 시행 수 1,032: {rows_ok}", *lines,
                f"- (참고, 판정에 넣지 않음) SR0 = SR 일 때 DSR {half:.12f}"]


@check
def monkey_mapping(cfg: dict) -> tuple[bool, list[str]]:
    """The random-entry pricing reproduces each candidate's own non-forced trades (and stage 0's original trades on the
    original CSV) from their signal day and holding bars, so random trades are priced like real ones; every real
    signal day is in the random-entry pool; 2,000 runs; the stage0 percentile is within 0.01 of the original's own."""
    from krxbt.costs import cost_factors
    from .common import with_costs
    from .overfit import pool_days, trade_returns
    from .replicate import load_csv, sample, with_indicators
    from .strategy import arrays, frames
    rd = results_dir()
    ov = pd.read_csv(rd / "overfit.csv")
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"]
    cands = pd.read_csv(rd / "candidates.csv")
    bc, sk = cost_factors(with_costs(cfg, "us"))
    lines, ok = [], True
    for v in ("all", "open", "close", "stage0"):
        r = one(ov, variant=v, primary=True)
        if r is None:
            ok = False
            lines.append(f"- {v}: 행 없음")
            continue
        ok &= int(r["monkey_runs"]) == cfg["overfit"]["monkey_runs"]
        if v == "stage0":
            smp = sample(with_indicators(cfg, load_csv(cfg)))
            rt = pd.read_csv(rd / "replicate_trades.csv")
            rt = rt[(rt["variant"] == "original") & (rt["part"] == "ALL") & ~rt["forced"].astype(bool)]
            obc, osk = cost_factors(with_costs(cfg, "orig"))
            got = trade_returns(smp["open"].to_numpy(np.float64), smp["close"].to_numpy(np.float64), rt["s"].to_numpy(),
                                (rt["x"] - rt["e"]).to_numpy(), "open", obc, osk)
            d = float(np.abs(got - rt["ret"].to_numpy()).max()) if len(rt) else np.inf
            gap = abs(float(r["monkey_pctile"]) - float(r["orig_monkey_pct_pf"]))
            ok &= d < 1e-12 and gap <= 0.01
            lines.append(f"- stage0: 거래 {len(rt)}건 다시 값 매김 최대 차이 {d:.1e}, 백분위 {float(r['monkey_pctile']):.4f} "
                         f"(원본 {float(r['orig_monkey_pct_pf']):.4f}, 차이 {gap:.4f} <= 0.01)")
            continue
        a = arrays(frames(cfg, [r["ticker"]])[r["ticker"]])
        ix = pd.DatetimeIndex(a["dates"])
        x = tr[(tr["combo"] == int(r["combo"])) & (tr["reason"] != "forced")]
        s, e, xx = (ix.get_indexer(x[k]) for k in ("signal_date", "entry_date", "exit_date"))
        got = trade_returns(a["open"], a["close"], s, xx - e, r["exec"], bc, sk)
        d = float(np.abs(got - x["ret"].to_numpy()).max()) if len(x) else np.inf
        c = one(cands, variant=v)
        outside = int((~np.isin(s, pool_days(a, cfg, c["trend"]))).sum()) if c is not None else -1
        ok &= d < 1e-12 and outside == 0
        lines.append(f"- {v} {r['label']}: 거래 {len(x)}건 다시 값 매김 최대 차이 {d:.1e}, 무작위 후보일 밖 신호일 "
                     f"{outside}, 백분위 {float(r['monkey_pctile']):.3f}")
    return ok, lines


@check
def tax_rules(cfg: dict) -> tuple[bool, list[str]]:
    """Unit cases: loss year 0, deduction per year, 22% of the excess, KRW at the fixed rate, payment on the first
    trading day of May of the next year (last day for the final year)."""
    from .tax import hold_after_tax, tax_due_usd, tax_pay_days
    tc = cfg["tax"]
    fx = tc["fx"]
    cases = [(-1000.0, 0.0), (0.0, 0.0), (2_500_000 / fx, 0.0), (3_500_000 / fx, 220_000 / fx), (10_000_000 / fx, 1_650_000 / fx)]
    lines, ok = ["| 연간 순이익(원) | 세금(원) | 기대(원) |", "|---|---|---|"], True
    for g, want in cases:
        got = tax_due_usd(g, tc)
        ok &= abs(got - want) < 1e-9
        lines.append(f"| {g * fx:,.0f} | {got * fx:,.0f} | {want * fx:,.0f} |")
    two = tax_due_usd(3_500_000 / fx, tc) + tax_due_usd(3_500_000 / fx, tc)
    ok &= abs(two * fx - 440_000) < 1e-6
    eq = pd.Series([100.0, 150.0, 200.0], pd.date_range("2020-01-01", periods=3))
    h = hold_after_tax(eq, 100.0, tc)
    ok &= h.iloc[:2].tolist() == [100.0, 150.0] and abs(h.iloc[-1] - (200.0 - tax_due_usd(100.0, tc))) < 1e-12
    cal = pd.bdate_range("2020-01-02", "2021-07-30")
    pay = tax_pay_days(cal, tc["pay_month"])
    ok &= pay == {2020: pd.Timestamp("2021-05-03"), 2021: pd.Timestamp("2021-07-30")}
    return ok, lines + [f"- 두 해 각각 350만 원 이익: 세금 합 {two * fx:,.0f}원 (기대 440,000원)",
                        f"- 납부일 예: {', '.join(f'{y}년 → {d.date()}' for y, d in pay.items())}"]


def wf_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .strategy import frames
    rd = results_dir()
    wft = pd.read_parquet(rd / "walkforward_trades.parquet")
    weq = pd.read_csv(rd / "walkforward_equity.csv", index_col=0, parse_dates=True)
    closes = pd.DataFrame({t: f["close"] for t, f in frames(cfg, cfg["grid"]["tickers"]).items()})
    return wft, weq, closes


@check
def account_engine(cfg: dict) -> tuple[bool, list[str]]:
    """Weight 1, no interest, no tax: the account equals run_portfolio(slots=1) and walkforward_equity.csv to 1e-6 USD,
    for the three walk-forward variants (trades on several tickers)."""
    from krxbt import portfolio as kp
    from .account import account
    wft, weq, closes = wf_inputs(cfg)
    cap = cfg["portfolio"]["initial_capital"]
    cal = weq.index
    lines, worst, ok = [], 0.0, True
    for var in ("all", "open", "close"):
        tr = wft[wft["variant"] == var].reset_index(drop=True)
        if tr.empty:
            ok = False
            lines.append(f"- {var}: 거래 없음")
            continue
        a, info = account(cal, tr, closes, cap, np.ones(len(tr)), np.ones(len(cal)))
        b, _ = kp.run_portfolio(cal, tr, closes.reindex(cal).ffill(), cap, slots=1, rank_by="signal_date")
        d = max(float((a - b).abs().max()), float((a - weq[f"wf_{var}"]).abs().max()))
        worst = max(worst, d)
        ok &= info["trades_taken"] == len(tr)
        lines.append(f"- {var}: 거래 {len(tr)}, 종목 {tr['ticker'].nunique()}개, 최대 차이 {d:.2e} 달러")
    return ok and worst < 1e-6, lines


@check
def tax_account(cfg: dict) -> tuple[bool, list[str]]:
    """Walk-forward open variant at the base capital: every year's tax = tax_due_usd(that year's realized gain); loss
    years pay 0; each year paid on the rule's day; pre- and after-tax curves equal before the first payment; on the
    first payment day without an entry, the gap equals the tax exactly (weight 1, so an entry day's gap would need
    the general tax x (1 - w + w x close/entry_px) form; we pick a no-entry payment day instead so the plain identity
    applies). `indep_realized` independently re-derives each trade's dollar gain from the no-tax run's own equity
    curve (`pre`) -- alloc_j = pre-tax equity the day before entry (or `cap` on cal[0]) x that day's cash factor x
    weight, the same three factors account() itself multiplies before sizing an entry, just recomputed here instead
    of trusting account()'s own `realized` dict -- grouped by EXIT year and compared to `pre_info["realized"]` (the
    no-tax run's own bookkeeping) to 1e-6 USD; this catches a gain booked to the wrong year that the `bad` check
    below cannot, since `bad` only compares two numbers both derived from the same account() call."""
    from .account import account
    from .common import capital_usd
    from .tax import tax_due_usd
    tc = cfg["tax"]
    cap = capital_usd(cfg)
    wft, weq, closes = wf_inputs(cfg)
    tr = wft[wft["variant"] == "open"].reset_index(drop=True)
    cal = weq.index
    w, fac = np.ones(len(tr)), np.ones(len(cal))
    pre, pre_info = account(cal, tr, closes, cap, w, fac)
    post, info = account(cal, tr, closes, cap, w, fac, tax=tc)
    bad = [y for y, g in info["realized"].items() if abs(info["taxes"].get(y, 0.0) - tax_due_usd(g, tc)) > 1e-9]

    # independent re-derivation (see docstring): does not touch account()'s own `realized`/`taxes` dicts at all.
    idx = {d: i for i, d in enumerate(cal)}
    indep_realized: dict[int, float] = {}
    for j, t in tr.iterrows():
        i = idx[t["entry_date"]]
        prior_eq = cap if i == 0 else float(pre.iloc[i - 1])
        alloc_j = float(w[j]) * prior_eq * float(fac[i])
        indep_realized[t["exit_date"].year] = indep_realized.get(t["exit_date"].year, 0.0) + alloc_j * float(t["ret"])
    indep_diff = {y: indep_realized[y] - pre_info["realized"].get(y, 0.0) for y in indep_realized}
    indep_ok = all(abs(v) < 1e-6 for v in indep_diff.values()) and set(indep_realized) == set(pre_info["realized"])

    loss_paid = [y for y, g in info["realized"].items() if g <= 0 and info["taxes"].get(y, 0.0) != 0]
    last_y = cal[-1].year

    def rule_day(y: int) -> pd.Timestamp | None:
        if y == last_y:
            return cal[-1]
        sel = cal[(cal.year == y + 1) & (cal.month >= tc["pay_month"])]
        return sel[0] if len(sel) else None

    bad_day = [y for y, d in info["paid_on"].items() if d != rule_day(y)]
    taxed = [y for y, v in info["taxes"].items() if v > 0]
    if not taxed:
        return False, ["- 세금을 낸 해가 없어 이 검증을 할 수 없다"]
    # the exact-tax-difference assertion is only made on a payment day without an entry (weight is always 1 here, so
    # an entry day's gap would need the general tax x (1 - w + w x close/entry_px) form; we avoid that complication).
    by_paid = sorted(taxed, key=lambda y: info["paid_on"][y])
    no_entry_years = [y for y in by_paid if len(tr.index[tr["entry_date"] == info["paid_on"][y]]) == 0]
    first = no_entry_years[0] if no_entry_years else by_paid[0]
    d0 = info["paid_on"][first]
    before = float((pre[pre.index < d0] - post[post.index < d0]).abs().max())
    drop = float(pre[d0] - post[d0])
    today = tr.index[tr["entry_date"] == d0]
    if len(today) == 0:
        expected = info["taxes"][first]
    else:
        j = int(today[0])
        expected = info["taxes"][first] * float(closes.at[d0, tr.at[j, "ticker"]]) / float(tr.at[j, "entry_px"])
    lines = ["| 해 | 실현 손익(원) | 세금(원) | 낸 날 |", "|---|---|---|---|"]
    lines += [f"| {y} | {info['realized'].get(y, 0.0) * tc['fx']:,.0f} | {info['taxes'][y] * tc['fx']:,.0f} | {info['paid_on'][y].date()} |"
              for y in sorted(info["taxes"])]
    indep_lines = ["| 해 | 독립 재계산(원) | account() realized(원) | 차이(달러) |", "|---|---|---|---|"]
    indep_lines += [f"| {y} | {indep_realized[y] * tc['fx']:,.0f} | {pre_info['realized'].get(y, 0.0) * tc['fx']:,.0f} | {indep_diff[y]:.2e} |"
                     for y in sorted(indep_realized)]
    ok = not bad and not loss_paid and not bad_day and before < 1e-9 and abs(drop - expected) < 1e-6 and indep_ok
    return ok, lines + [f"- 첫 납부일(진입 없는 날 우선) {d0.date()} 전 세전·세후 차이 {before:.1e}, 그날 차이 {drop:,.2f} 달러 (기대 {expected:,.2f})",
                        f"- 납부일이 규칙과 다른 해: {bad_day or '없음'}, 가장 낮은 현금 {info['min_cash']:,.2f} 달러",
                        "- 독립 재계산(진입 전날 무세금 자산 × 진입일 현금 계수 × 비중 1 로 거래별 alloc 을 다시 구해 "
                        "청산 연도로 묶은 값과 무세금 계좌의 realized 대조, account() 내부 realized/taxes 를 쓰지 않는다):"] + indep_lines


TRADE_COLS = ["signal_date", "entry_date", "exit_date", "entry_px", "exit_px", "ret"]


def perturb_prices(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """The last n days' OHLC (and raw open/close, by the same factors) x U(0.9, 1.1); high/low re-bound the bar."""
    g = df.copy()
    rng = np.random.default_rng(seed)
    ix = g.index[-n:]
    for col in ("open", "high", "low", "close"):
        m = rng.uniform(0.9, 1.1, n)
        g.loc[ix, col] = g.loc[ix, col].to_numpy() * m
        if f"raw_{col}" in g:
            g.loc[ix, f"raw_{col}"] = g.loc[ix, f"raw_{col}"].to_numpy() * m
    g.loc[ix, "high"] = g.loc[ix, ["open", "high", "low", "close"]].max(axis=1)
    g.loc[ix, "low"] = g.loc[ix, ["open", "high", "low", "close"]].min(axis=1)
    return g


def trade_diff(A: pd.DataFrame, B: pd.DataFrame) -> int:
    if len(A) != len(B):
        return abs(len(A) - len(B))
    a, b = A[TRADE_COLS].reset_index(drop=True), B[TRADE_COLS].reset_index(drop=True)
    return int((~(a == b).all(axis=1)).sum())


@check
def lookahead(cfg: dict) -> tuple[bool, list[str]]:
    """Change the last N days' prices by +-10%: signals before those days and trades that ended before them stay the
    same (4 tickers x 4 RSI(2) combos + the IBS rule on the engine, and the stage-0 loop on the original CSV)."""
    from .common import with_costs
    from .replicate import load_csv, run_part, sample, with_indicators
    from .strategy import add_indicators, arrays, frames, ibs_rule, rule, run_rule
    n = cfg["checks"]["perturb_days"]
    w = cfg["indicators"]["ma_window"]
    ucfg = with_costs(cfg, "us")
    combos = [(15, "above", "green2", "open"), (5, "none", "sma5", "close"), (10, "below", "rsi70", "open"),
              (15, "none", "green2", "close")]
    lines = [f"마지막 {n}일 OHLC 를 ±10% 무작위로 바꿨다.", "", "| 종목 | 규칙 | 신호 차이 | 거래 차이 | 거래 수 (원본, 자른 날 전) |",
             "|---|---|---|---|---|"]
    ok = True
    for t, f in frames(cfg, cfg["grid"]["tickers"]).items():
        h = perturb_prices(f, n)
        h["ma"] = h["close"].where(h["valid"]).rolling(w, min_periods=w).mean()
        h = add_indicators(h, cfg)
        cut = f.index[-n]
        early = f.index < cut
        a, b = arrays(f), arrays(h)
        rules = [(f"RSI<{thr} {trend} {ex} {mode}", rule(a, cfg, thr, trend, ex), rule(b, cfg, thr, trend, ex), mode)
                 for thr, trend, ex, mode in combos]
        rules.append(("IBS 0.2/0.8 open", ibs_rule(a, cfg), ibs_rule(b, cfg), "open"))
        for name, (ca, ta), (cb, tb), mode in rules:
            sig = int((ca[early] != cb[early]).sum() + (ta[early] != tb[early]).sum())
            A, B = run_rule(a, ucfg, ca, ta, mode), run_rule(b, ucfg, cb, tb, mode)
            A_early = A[A["exit_date"] < cut]
            d = trade_diff(A_early, B[B["exit_date"] < cut])
            n_trades = len(A_early)
            ok &= sig == 0 and d == 0 and n_trades > 0
            lines.append(f"| {t} | {name} | {sig} | {d} | {n_trades} |")
    raw = load_csv(cfg)
    cut = raw.index[-n]
    res = []
    for df in (raw, perturb_prices(raw, n)):
        tr, _ = run_part(cfg, sample(with_indicators(cfg, df)), 0)
        res.append(tr[tr["exit_date"] < cut])
    d0 = trade_diff(res[0], res[1])
    ok &= d0 == 0 and len(res[0]) > 0
    lines.append(f"| QQQ 원본 CSV (단계 0 반복문) | 원본 규칙 | - | {d0} | {len(res[0])} |")
    return ok, lines


@check
def decision_missing(cfg: dict) -> tuple[bool, list[str]]:
    """The decision rule fails a variant whose rows are missing (never passes it), and on the real rows its four flags
    agree with an independent reading of the csv values (DSR from the pre-registered primary row)."""
    from .report import decision
    rd = results_dir()
    wfs, ov = pd.read_csv(rd / "walkforward_summary.csv"), pd.read_csv(rd / "overfit.csv")
    d_full, _ = decision(cfg, wfs, ov)
    d_cut, line = decision(cfg, wfs[wfs["variant"] != "open"], ov[ov["variant"] != "close"])
    r_open, r_close = one(d_cut, variant="open"), one(d_cut, variant="close")
    missing_ok = (r_open is not None and r_close is not None and not r_open["pass"] and not r_close["pass"]
                  and r_open["note"] == "행 없음" and r_close["note"] == "행 없음")
    agree = True
    dmin, top = cfg["overfit"]["dsr_min"], 1 - cfg["overfit"]["monkey_top"]
    for _, r in d_full.iterrows():
        v = r["variant"]
        s, h = one(wfs, variant=v, name="wf"), one(wfs, variant=v, name="hold")
        kd, ke = one(wfs, variant=v, name="kdd"), one(wfs, variant=v, name="kexp")
        o = ov[(ov["variant"] == v) & ov["primary"].astype(str).eq("True")]
        if any(x is None for x in (s, h, kd, ke)) or len(o) != 1:
            agree &= not bool(r["pass"])
            continue
        o = o.iloc[0]
        want = [s["sharpe"] > h["sharpe"], s["cagr"] > max(kd["cagr"], ke["cagr"]), o["dsr"] >= dmin, o["monkey_pctile"] >= top]
        agree &= [bool(r[c]) for c in ("c1", "c2", "c3", "c4")] == [bool(x) for x in want]
        agree &= bool(r["pass"]) == all(want)
    ok = missing_ok and agree and len(d_full) == 3
    return ok, [f"- 행을 뺀 판본(open, close)이 불합격으로 나옴: {missing_ok}", f"- 실제 행의 판정이 csv 값과 맞음: {agree}",
                f"- 판정 문장 예: {line}"]


@check
def readme_sync(cfg: dict) -> tuple[bool, list[str]]:
    """The README block between the result markers is exactly what src.report generated (numbers come from code)."""
    from .report import END, START
    doc = (ROOT / "README.md").read_text(encoding="utf-8")
    if START not in doc or END not in doc:
        return False, ["- README 에 결과 표시(markers)가 없다"]
    block = doc[doc.index(START) + len(START):doc.index(END)].strip()
    gen = (results_dir() / "readme_results.md").read_text(encoding="utf-8").strip()
    return block == gen and len(gen) > 0, [f"- README 결과 부분 {len(block)}자, 생성본 {len(gen)}자, 같음: {block == gen}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cfg = load_config()
    names = a.only or list(CHECKS)
    out = ["# 검증 결과", "", "`python -m src.checks` 가 만든다.", ""]
    failed = []
    for n in names:
        ok, lines = CHECKS[n](cfg)
        out += [f"## {n}: {'통과' if ok else '실패'}", "", *lines, ""]
        if not ok:
            failed.append(n)
    out.append(f"실패: {', '.join(failed) if failed else '없음'}")
    text = "\n".join(out) + "\n"
    if not a.only:
        (results_dir() / "checks.md").write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
