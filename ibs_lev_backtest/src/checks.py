"""Verification checks. Writes results/checks.md (full run only).

Every check returns (passed, markdown lines). Tasks add checks above main().

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
from krxbt import portfolio as kp

from .common import ROOT, capital_usd, ibs, load_config, results_dir
from .crosscheck import compare
from .data import cash_factors, frames, irx
from .strategy import arrays, calendar, signals, trades
from .sizing import account, realized_vol, weights
from .tax import hold_after_tax, tax_due_usd, tax_pay_days

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
    """Thresholds fixed at 0.13 / 0.5, no grid section, 48 combos."""
    L = cfg["lev"]
    n = len(L["tickers"]) * len(L["exec"]) * len(L["sizing"]) * len(L["cash"])
    ok = cfg["ibs"] == {"entry": 0.13, "exit": 0.5} and "grid" not in cfg and n == 48
    return ok, [f"- ibs: {cfg['ibs']}", f"- grid 섹션 없음: {'grid' not in cfg}", f"- 조합 수: {n}"]


@check
def data_files(cfg: dict) -> tuple[bool, list[str]]:
    """Every ticker loads; the leveraged ETFs start on their listing days."""
    cl, L = cfg["claims"], cfg["lev"]
    need = list(dict.fromkeys(cl["tickers"] + cl["leveraged"] + list(L["underlying"].values())))
    fr = frames(cfg, need)
    lines = ["| 종목 | 첫 거래일 | 마지막 거래일 | 거래일 |", "|---|---|---|---|"]
    for t, f in fr.items():
        v = f.index[f["valid"]]
        lines.append(f"| {t} | {v[0].date()} | {v[-1].date()} | {len(v):,} |")
    first = {t: str(fr[t].index[fr[t]["valid"]][0].date()) for t in cfg["checks"]["listing_first"]}
    last = {str(f.index[f["valid"]][-1].date()) for f in fr.values()}
    ok = first == cfg["checks"]["listing_first"] and last == {cfg["data"]["end"]}
    return ok, lines + ["", f"- 상장 첫날: {first} (기대 {cfg['checks']['listing_first']})",
                        f"- 마지막 날: {sorted(last)} (기대 {cfg['data']['end']}, 이 날 포함)"]


@check
def irx_lag(cfg: dict) -> tuple[bool, list[str]]:
    """Cash interest on day i uses ^IRX of day i-1 (lag 1); lag 0 reproduces the original."""
    rate = irx(cfg).sort_index()
    f = frames(cfg, ["TQQQ"])["TQQQ"]
    cal = f.index[f.index >= pd.Timestamp(cfg["data"]["start"])]
    days = cfg["cash"]["days_per_year"]
    f1, f0 = cash_factors(cal, rate, 1, days), cash_factors(cal, rate, 0, days)
    e1 = np.array([1 + rate.asof(cal[i - 1]) / days for i in range(1, len(cal))])
    e0 = np.array([1 + rate.asof(cal[i]) / days for i in range(1, len(cal))])
    d1, d0 = float(np.abs(f1[1:] - e1).max()), float(np.abs(f0[1:] - e0).max())
    changed = int((f1[1:] != f0[1:]).sum())
    ok = d1 < 1e-15 and d0 < 1e-15 and f1[0] == 1.0 and changed > 0 and rate.min() > -0.01 and rate.max() < 0.07
    return ok, [f"- ^IRX {rate.index[0].date()} → {rate.index[-1].date()}, {len(rate):,}일, 최소 {rate.min():.4f} 최대 {rate.max():.4f}",
                f"- lag 1 과 전날 값의 최대 차이 {d1:.1e}, lag 0 과 같은 날 값의 최대 차이 {d0:.1e}",
                f"- lag 0 과 1 이 다른 날 {changed:,}일 (0 이면 시차가 적용되지 않은 것)"]


@check
def ibs_flat(cfg: dict) -> tuple[bool, list[str]]:
    """high == low bars: IBS NaN, neither entry nor exit signal."""
    cl = cfg["claims"]
    lines = ["| 종목 | 고가 = 저가인 날 | 그날 신호 |", "|---|---|---|"]
    ok = True
    for t, f in frames(cfg, cl["tickers"] + cl["leveraged"]).items():
        flat = ((f["high"] == f["low"]) & f["valid"]).to_numpy()
        cand, target = signals(arrays(f), cfg["ibs"]["entry"], cfg["ibs"]["exit"])
        bad = int((cand[flat] | target[flat]).sum())
        nan_ok = bool(np.isnan(f["ibs"].to_numpy()[flat]).all())
        ok &= bad == 0 and nan_ok
        lines.append(f"| {t} | {int(flat.sum())} | {bad} |")
    # synthetic case: a flat bar between two signals must block both
    a = {"ibs": np.array([0.05, np.nan, 0.9]), "eligible": np.ones(3, bool)}
    c, g = signals(a, 0.13, 0.5)
    ok &= c.tolist() == [True, False, False] and g.tolist() == [False, False, True]
    return ok, lines


@check
def listing(cfg: dict) -> tuple[bool, list[str]]:
    """Signals may start on the listing day (no engine warm-up), as in the original."""
    lines, ok = [], True
    for t in cfg["lev"]["tickers"]:
        f = frames(cfg, [t])[t]
        cal = calendar(cfg, f)
        first_valid = f.index[f["valid"]][0]
        first_elig = f.index[f["eligible"]][0]
        tr = trades(cfg, f, t, "open", True)
        ok &= first_elig == first_valid == cal[0]
        lines.append(f"- {t}: 첫 거래일 {first_valid.date()}, 첫 eligible {first_elig.date()}, "
                     f"첫 신호 {tr['signal_date'].iloc[0].date()}, 거래 {len(tr)}")
    return ok, lines


@check
def tax_rules(cfg: dict) -> tuple[bool, list[str]]:
    """Unit cases: loss year 0, deduction per year, 22% of the excess, KRW at the fixed rate,
    payment on the first trading day of May of the next year (last day for the final year)."""
    tc = cfg["tax"]
    fx = tc["fx"]
    cases = [(-1000.0, 0.0), (0.0, 0.0), (2_500_000 / fx, 0.0),
             (3_500_000 / fx, 220_000 / fx), (10_000_000 / fx, 1_650_000 / fx)]
    lines, ok = ["| 연간 순이익(원) | 세금(원) | 기대(원) |", "|---|---|---|"], True
    for g, want in cases:
        got = tax_due_usd(g, tc)
        ok &= abs(got - want) < 1e-9
        lines.append(f"| {g * fx:,.0f} | {got * fx:,.0f} | {want * fx:,.0f} |")
    # two years of +3.5M KRW each: the deduction applies in both years
    two = tax_due_usd(3_500_000 / fx, tc) + tax_due_usd(3_500_000 / fx, tc)
    ok &= abs(two * fx - 440_000) < 1e-6
    eq = pd.Series([100.0, 150.0, 200.0], pd.date_range("2020-01-01", periods=3))
    h = hold_after_tax(eq, 100.0, tc)
    ok &= h.iloc[:2].tolist() == [100.0, 150.0] and abs(h.iloc[-1] - (200.0 - tax_due_usd(100.0, tc))) < 1e-12
    # payment days: 2020 -> first trading day of May 2021; 2021 (no May 2022 in the sample) -> last day
    cal = pd.bdate_range("2020-01-02", "2021-07-30")
    pay = tax_pay_days(cal, tc["pay_month"])
    ok &= pay == {2020: pd.Timestamp("2021-05-03"), 2021: pd.Timestamp("2021-07-30")}
    return ok, lines + [f"- 두 해 각각 350만 원 이익: 세금 합 {two * fx:,.0f}원 (기대 440,000원)",
                        f"- 납부일 예: {', '.join(f'{y}년 → {d.date()}' for y, d in pay.items())}"]


@check
def account_engine(cfg: dict) -> tuple[bool, list[str]]:
    """weight 1, no interest, no tax: account == run_portfolio(slots=1) to 1e-6 USD."""
    cap = capital_usd(cfg)
    lines, worst = [], 0.0
    for t in cfg["lev"]["tickers"]:
        f = frames(cfg, [t])[t]
        cal = calendar(cfg, f)
        for mode in ("open", "close"):
            tr = trades(cfg, f, t, mode, True)
            a, info = account(cal, tr, f["close"], cap, np.ones(len(tr)), np.ones(len(cal)))
            closes = pd.DataFrame({t: f["close"]}).reindex(cal).ffill()
            b, binfo = kp.run_portfolio(cal, tr.assign(ticker=t), closes, cap, slots=1, rank_by="signal_date")
            d = float((a - b).abs().max())
            worst = max(worst, d)
            lines.append(f"- {t} {mode}: 거래 {len(tr)}, 최대 차이 {d:.2e} 달러, 노출 {info['invested_share']:.3f} / {binfo['invested_share']:.3f}")
    return worst < 1e-6, lines


@check
def interest_only(cfg: dict) -> tuple[bool, list[str]]:
    """No trades: equity = capital x product of the cash factors; partial weight: leftover earns."""
    cap = capital_usd(cfg)
    f = frames(cfg, ["TQQQ"])["TQQQ"]
    cal = calendar(cfg, f)
    fac = cash_factors(cal, irx(cfg), cfg["cash"]["irx_lag"], cfg["cash"]["days_per_year"])
    empty = trades(cfg, f, "TQQQ", "open", True).iloc[:0]
    e, _ = account(cal, empty, f["close"], cap, np.ones(0), fac)
    d1 = float(abs(e.iloc[-1] - cap * np.prod(fac)))
    # weight 0: the trade happens with alloc 0, so the result must equal the no-trade curve
    tr = trades(cfg, f, "TQQQ", "open", True)
    z, _ = account(cal, tr, f["close"], cap, np.zeros(len(tr)), fac)
    d2 = float((z - e).abs().max())
    # weight 0.5, one trade, no tax: the leftover half keeps compounding while the position is open
    one = tr.iloc[[len(tr) // 2]].reset_index(drop=True)
    h, _ = account(cal, one, f["close"], cap, np.array([0.5]), fac)
    ie, ix = cal.get_loc(one.at[0, "entry_date"]), cal.get_loc(one.at[0, "exit_date"])
    px = f["close"].reindex(cal).ffill().to_numpy(np.float64)
    g = np.cumprod(fac)  # g[i] = product of fac[0..i]
    alloc = 0.5 * cap * g[ie]
    ref = cap * g.copy()  # before the entry: all cash
    for i in range(ie, len(cal)):
        left = alloc * g[i] / g[ie]  # leftover half compounding from the entry day
        if i < ix:
            ref[i] = left + alloc * px[i] / float(one.at[0, "entry_px"])
        else:
            ref[i] = (alloc * g[ix] / g[ie] + alloc * (1 + float(one.at[0, "ret"]))) * g[i] / g[ix]
    d3 = float(np.abs(h.to_numpy() - ref).max())
    earned = alloc * (g[ix] / g[ie] - 1)
    return d1 < 1e-6 and d2 < 1e-6 and d3 < 1e-6, [
        f"- 거래 없음: 최종 {e.iloc[-1]:,.2f} 달러, Π(이자 계수) 와 차이 {d1:.1e}",
        f"- 비중 0 거래: 거래 없음 곡선과 최대 차이 {d2:.1e}",
        f"- 비중 0.5 거래 1건 ({one.at[0, 'entry_date'].date()} → {one.at[0, 'exit_date'].date()}): "
        f"닫힌 식과 최대 차이 {d3:.1e} 달러, 보유 중 남은 현금 이자 {earned:,.2f} 달러, 최종 {h.iloc[-1]:,.2f} 달러"]


@check
def vol_window(cfg: dict) -> tuple[bool, list[str]]:
    """The weight uses returns up to the signal day's close; the entry day's return is not in it."""
    w = cfg["indicators"]["vol_window"]
    f = frames(cfg, ["TQQQ"])["TQQQ"]
    tr = trades(cfg, f, "TQQQ", "open", True)
    rv = realized_vol(f, w)
    c = f.loc[f["valid"], "close"]
    diff, moved = 0.0, 0
    for j, t in tr.iloc[w + 5: w + 55].iterrows():
        hist = c[c.index <= t["signal_date"]].iloc[-(w + 1):]
        ref = float(np.std(hist.pct_change().dropna().to_numpy(), ddof=1) * np.sqrt(252))
        diff = max(diff, abs(ref - rv[t["signal_date"]]))
        g = f.copy()
        g.loc[t["entry_date"], "close"] *= 1.5  # change only the entry day
        w0 = weights(tr.loc[[j]], rv, 0.3)[0]
        w1 = weights(tr.loc[[j]], realized_vol(g, w), 0.3)[0]
        moved += w0 != w1
    return diff < 1e-12 and moved == 0, [f"- 50건: 직접 계산과의 최대 차이 {diff:.1e}",
                                         f"- 진입일 종가를 1.5배로 바꿨을 때 비중이 바뀐 거래 {moved}건"]


@check
def tax_account(cfg: dict) -> tuple[bool, list[str]]:
    """On a real run (TQQQ open, vol 0.4, ^IRX, costs, 100M KRW): every year's tax equals
    tax_due_usd of that year's realized gain; loss years pay 0; each year is paid on the first
    trading day of May of the next year (the last year on the last day); pre- and after-tax curves
    are equal until the first payment day.

    On the first payment day itself, if no entry happens that day, equity differs by exactly that
    day's tax (a same-day exit is unaffected: its position was opened before the split, so it books
    an identical dollar gain into both cash streams). If an entry DOES happen that day, this simple
    equality is false in general: for exec modes where the entry executes away from the day's close
    (exec="open": bought at the open, marked at the close), the smaller post-tax cash buys a smaller
    position whose gain/loss to the close is scaled by weight w, so the exact identity is instead
    diff = tax x (1 - w + w x close / entry_px) (w = 1 and/or close == entry_px collapses this back
    to diff == tax, e.g. exec="close", where the entry executes at the close itself). Both exact
    forms are checked directly, whichever applies to the first payment day found.

    Both `info["taxes"][y]` and `tax_due_usd(info["realized"][y])` come from the same account()
    call, so this alone cannot catch a gain booked to the wrong year or the wrong dollar amount
    (a bug there would move both together and still look consistent). `indep_realized` re-derives
    each trade's dollar gain from the no-tax run independently of account()'s own `realized` dict:
    alloc_j = equity the day before entry (from the no-tax equity curve `pre`, or `cap` if the
    trade enters on cal[0]) x that entry day's cash factor x the trade's weight -- the same three
    factors account() itself multiplies (cash *= factors[i] before the day's entry, then
    alloc = weight x cash), just computed from account()'s own output instead of from inside it.
    gain_j = alloc_j x ret_j is grouped by EXIT year (gains are booked to the exit, not the entry)
    and compared to `pre_info["realized"]` to 1e-6 USD."""
    tc = cfg["tax"]
    cap = capital_usd(cfg)
    f = frames(cfg, ["TQQQ"])["TQQQ"]
    cal = calendar(cfg, f)
    tr = trades(cfg, f, "TQQQ", "open", True)
    w = weights(tr, realized_vol(f, cfg["indicators"]["vol_window"]), 0.4)
    fac = cash_factors(cal, irx(cfg), cfg["cash"]["irx_lag"], cfg["cash"]["days_per_year"])
    pre, pre_info = account(cal, tr, f["close"], cap, w, fac)
    post, info = account(cal, tr, f["close"], cap, w, fac, tax=tc)
    bad = [y for y, g in info["realized"].items() if abs(info["taxes"].get(y, 0.0) - tax_due_usd(g, tc)) > 1e-9]

    # independent re-derivation (see docstring): catches a wrong booking year or amount that the
    # `bad` check above cannot, since it does not touch account()'s own `realized` dict at all.
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
        return sel[0] if len(sel) else None  # no May-or-later day that year: fails the comparison below, not a crash

    bad_day = [y for y, d in info["paid_on"].items() if d != rule_day(y)]
    taxed_years = [y for y, v in info["taxes"].items() if v > 0]
    if not taxed_years:
        return False, ["- 세금을 낸 해가 없어 이 검증을 할 수 없음 (실현 이익이 있는지 확인)"]
    first = min(taxed_years, key=lambda y: info["paid_on"][y])
    d0 = info["paid_on"][first]
    before = float((pre[pre.index < d0] - post[post.index < d0]).abs().max())
    drop = float(pre[d0] - post[d0])
    entry_today = tr.index[tr["entry_date"] == d0]
    lines = ["| 해 | 실현 손익(원) | 세금(원) | 낸 날 |", "|---|---|---|---|"]
    lines += [f"| {y} | {info['realized'].get(y, 0.0) * tc['fx']:,.0f} | {info['taxes'][y] * tc['fx']:,.0f} | {info['paid_on'][y].date()} |"
              for y in sorted(info["taxes"])]
    if len(entry_today) == 0:
        exact_ok = abs(drop - info["taxes"][first]) < 1e-6
        exact_line = (f"- 첫 납부일 {d0.date()} 전까지 세전·세후 차이 {before:.1e}, 그날(진입 없음) 차이 {drop:,.2f} 달러 "
                      f"= {first}년 세금 {info['taxes'][first]:,.2f} 달러")
    else:
        j = int(entry_today[0])
        ratio = float(f.loc[d0, "close"]) / float(tr.at[j, "entry_px"])
        wj = float(w[j])
        expected = info["taxes"][first] * (1 - wj + wj * ratio)
        exact_ok = abs(drop - expected) < 1e-6
        exact_line = (f"- 첫 납부일 {d0.date()} 전까지 세전·세후 차이 {before:.1e}, 그날 진입이 있어 차이 {drop:,.2f} 달러, "
                      f"기대 세금×(1-w+w×종가/진입가) = {expected:,.2f} 달러 "
                      f"(세금 {info['taxes'][first]:,.2f}, w {wj:.3f}, 종가/진입가 {ratio:.4f})")
    ok = not bad and not loss_paid and not bad_day and before < 1e-9 and exact_ok and indep_ok
    indep_lines = ["| 해 | 독립 재계산(원) | account() realized(원) | 차이(달러) |", "|---|---|---|---|"]
    indep_lines += [f"| {y} | {indep_realized[y] * tc['fx']:,.0f} | {pre_info['realized'].get(y, 0.0) * tc['fx']:,.0f} | {indep_diff[y]:.2e} |"
                     for y in sorted(indep_realized)]
    return ok, lines + [exact_line,
                        f"- 납부일이 규칙과 다른 해: {bad_day or '없음'}",
                        f"- 가장 낮은 현금 {info['min_cash']:,.2f} 달러 (음수면 전액 보유 중 세금을 낸 것)",
                        "- 독립 재계산(진입 전날 무세금 자산 × 진입일 현금 계수 × 비중 w 로 거래별 alloc 을 다시 구해 "
                        "청산 연도로 묶은 값과 account() 의 realized 대조, account() 내부 realized 를 쓰지 않는다):"] + indep_lines


@check
def crosscheck(cfg: dict) -> tuple[bool, list[str]]:
    """Trade count within crosscheck.max_trade_diff of the original package (spec: 5%)."""
    df = compare(cfg)
    ours = df[df["source"] != "original"]
    ok = bool((ours["trade_diff"] <= cfg["crosscheck"]["max_trade_diff"]).all())
    lines = ["| 출처 | 거래 | 연 수익률 | 샤프 | 최대 낙폭 | 노출 | 같은 진입일 | 신호가 다른 날 |", "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.source} | {r.trades} | {r.cagr:+.2%} | {r.sharpe:.2f} | {r.mdd:.1%} | {r.exposure:.1%} | "
                     f"{'' if pd.isna(getattr(r, 'same_entries', np.nan)) else int(r.same_entries)} | "
                     f"{'' if pd.isna(getattr(r, 'signal_days_differ', np.nan)) else int(r.signal_days_differ)} |")
    return ok, lines


def perturb(f: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    g = f.copy()
    rng = np.random.default_rng(seed)
    ix = g.index[-n:]
    for col in ("open", "high", "low", "close"):
        g.loc[ix, col] = g.loc[ix, col] * rng.uniform(0.9, 1.1, n)
    g.loc[ix, "high"] = g.loc[ix, ["open", "high", "low", "close"]].max(axis=1)
    g.loc[ix, "low"] = g.loc[ix, ["open", "high", "low", "close"]].min(axis=1)
    g["ibs"] = ibs(g["high"], g["low"], g["close"])
    return g


@check
def lookahead(cfg: dict) -> tuple[bool, list[str]]:
    """Change the last N days' OHLC by +-10%: signals, trades, weights and equity before those days stay the same."""
    n = cfg["checks"]["perturb_days"]
    cap = capital_usd(cfg)
    vw = cfg["indicators"]["vol_window"]
    e, x = cfg["ibs"]["entry"], cfg["ibs"]["exit"]
    rate = irx(cfg)
    lines = [f"마지막 {n}일 OHLC 를 ±10% 무작위로 바꿨다.", "",
             "| 종목 | 체결 | 신호 차이 | 거래 차이 | 비중 차이 | 자산 최대 차이(달러) |", "|---|---|---|---|---|---|"]
    ok = True
    for t in cfg["lev"]["tickers"]:
        f = frames(cfg, [t])[t]
        g = perturb(f, n)
        cut = f.index[-n]
        early = f.index < cut
        cal = calendar(cfg, f)
        fac = cash_factors(cal, rate, cfg["cash"]["irx_lag"], cfg["cash"]["days_per_year"])
        ca, ta = signals(arrays(f), e, x)
        cb, tb = signals(arrays(g), e, x)
        sig = int((ca[early] != cb[early]).sum() + (ta[early] != tb[early]).sum())
        for mode in ("open", "close"):
            tra, trb = trades(cfg, f, t, mode, True), trades(cfg, g, t, mode, True)
            A = tra[tra["exit_date"] < cut].reset_index(drop=True)
            B = trb[trb["exit_date"] < cut].reset_index(drop=True)
            tr_diff = abs(len(A) - len(B)) if len(A) != len(B) else int((~(A.round(12) == B.round(12)).all(axis=1)).sum())
            wa, wb = weights(tra, realized_vol(f, vw), 0.3), weights(trb, realized_vol(g, vw), 0.3)
            k = min(len(A), len(B))
            w_diff = int((np.abs(wa[:k] - wb[:k]) > 1e-12).sum())
            ea, _ = account(cal, tra, f["close"], cap, wa, fac, tax=cfg["tax"])
            eb, _ = account(cal, trb, g["close"], cap, wb, fac, tax=cfg["tax"])
            eq_diff = float((ea[ea.index < cut] - eb[eb.index < cut]).abs().max())
            ok &= sig == 0 and tr_diff == 0 and w_diff == 0 and eq_diff < 1e-9
            lines.append(f"| {t} | {mode} | {sig} | {tr_diff} | {w_diff} | {eq_diff:.1e} |")
    return ok, lines


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
    text = "\n".join(out)
    if not a.only:
        (results_dir() / "checks.md").write_text(text, encoding="utf-8")
    print(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
