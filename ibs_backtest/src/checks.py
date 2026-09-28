"""Verification checks. Writes results/checks.md.

1. Engine vs stage 0: SPY trades from 2008 (IBS 0.20 / 0.80, no filter, no
   max hold, zero cost), both executions: same signal, entry and exit dates
   and the same return.
2. No look-ahead: change the last N days' prices; every signal and every
   trade that ended before those days must stay the same.
3. Entry day: close execution enters on the signal day, open execution on
   the next trading day.
4. Equity on every day = initial capital x product of the returns of the
   trades closed so far (x open position value), to the cent. Checked on
   exit days and flat days for every grid combo.
5. Days with high == low and what IBS they got.

Usage:  python -m src.checks        (after src.replicate and src.grid)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import load_config, results_dir, zero_costs
from .grid import combos
from .replicate import download, signal_pairs, simulate
from .strategy import arrays, frames, run, signals


def engine_vs_replicate(cfg: dict, f: pd.DataFrame) -> list[str]:
    out = ["## 1. 엔진 대조 (SPY, 2008 이후, 진입 0.20 / 청산 0.80, 필터·보유제한 없음, 비용 0)", ""]
    rp = pd.read_csv(results_dir() / "replicate_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    z = zero_costs(cfg)
    a = arrays(f)
    start = pd.Timestamp(cfg["data"]["start"])
    out += ["| 체결 | 재현 거래 | 엔진 거래 | 날짜 모두 같음 | 수익률 최대 차이 | 다른 거래 |", "|---|---|---|---|---|---|"]
    rows = []
    for mode in ("close", "open"):
        ref = rp[(rp["mode"] == mode) & (~rp["costs"]) & (rp["signal_date"] >= start)].reset_index(drop=True)
        # a stage-0 position opened before data.start blocks signals the engine sees; start both at the
        # first stage-0 signal on or after data.start
        eng = run(a, z, cfg["ibs"]["entry"], cfg["ibs"]["exit"], mode)
        eng = eng[eng["signal_date"] >= ref["signal_date"].iloc[0]].reset_index(drop=True)
        n = min(len(ref), len(eng))
        cols = ["signal_date", "entry_date", "exit_date"]
        same = (ref[cols].iloc[:n].to_numpy() == eng[cols].iloc[:n].to_numpy()).all(axis=1)
        dr = np.abs(ref["ret"].iloc[:n].to_numpy() - eng["ret"].iloc[:n].to_numpy())
        # the last stage-0 trade may still be open ("forced"): its exit is the last bar in both
        bad = int((~same).sum()) + abs(len(ref) - len(eng))
        out.append(f"| {mode} | {len(ref)} | {len(eng)} | {'예' if bad == 0 else '아니오'} | {dr.max():.2e} | {bad} |")
        rows.append({"exec": mode, "replicate_trades": len(ref), "engine_trades": len(eng), "mismatched": bad,
                     "max_ret_diff": dr.max(), "first_signal": ref["signal_date"].iloc[0].date()})
        if bad:
            k = int(np.argmin(same)) if (~same).any() else n
            out.append(f"\n첫 불일치 {mode}: 재현 {ref.iloc[k][cols].tolist() if k < len(ref) else '-'} / "
                       f"엔진 {eng.iloc[k][cols].tolist() if k < len(eng) else '-'}\n")
    pd.DataFrame(rows).to_csv(results_dir() / "engine_match.csv", index=False)
    out.append("")
    out.append("재현(단계 0)은 yfinance `auto_adjust=True` 1993년부터, 엔진은 us_data 의 `adj_close` 로 수정한 OHLC 다. "
               "수익률 차이는 두 수정 계수의 반올림 차이다.")
    return out + [""]


def perturb(f: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    g = f.copy()
    rng = np.random.default_rng(seed)
    ix = g.index[-n:]
    for col in ("open", "high", "low", "close"):
        g.loc[ix, col] = g.loc[ix, col] * rng.uniform(0.9, 1.1, n)
    g.loc[ix, "high"] = g.loc[ix, ["open", "high", "low", "close"]].max(axis=1)
    g.loc[ix, "low"] = g.loc[ix, ["open", "high", "low", "close"]].min(axis=1)
    return g


def lookahead(cfg: dict, fr: dict[str, pd.DataFrame]) -> list[str]:
    n = cfg["checks"]["perturb_days"]
    win = cfg["indicators"]["ma_window"]
    out = [f"## 2. 미래 데이터 (마지막 {n}일 OHLC 를 ±10% 무작위로 바꿈)", "",
           "| 종목 | 조합 | 바꾼 첫날 전 신호 차이 | 그 전 청산 거래 차이 |", "|---|---|---|---|"]
    ok_all = True
    from .common import ibs
    for t, f in fr.items():
        g = perturb(f, n)
        g["ma"] = g["close"].where(g["valid"]).rolling(win, min_periods=win).mean()
        g["ibs"] = ibs(g["high"], g["low"], g["close"], cfg["ibs"]["flat_value"])
        cut = f.index[-n]
        a, b = arrays(f), arrays(g)
        for entry, exit, mode, trend, hold in [(0.20, 0.80, "close", "none", None), (0.20, 0.80, "open", "none", None),
                                               (0.25, 0.50, "open", "ma200", 5), (0.10, 0.90, "close", "ma200", 10)]:
            ca, _ = signals(a, entry, exit, trend)
            cb, _ = signals(b, entry, exit, trend)
            early = f.index < cut
            sig_diff = int((ca[early] != cb[early]).sum())
            ta = run(a, cfg, entry, exit, mode, trend, hold)
            tb = run(b, cfg, entry, exit, mode, trend, hold)
            ta, tb = ta[ta["exit_date"] < cut].reset_index(drop=True), tb[tb["exit_date"] < cut].reset_index(drop=True)
            tr_diff = abs(len(ta) - len(tb)) if len(ta) != len(tb) else int((~(ta.round(12) == tb.round(12)).all(axis=1)).sum())
            ok_all &= sig_diff == 0 and tr_diff == 0
            out.append(f"| {t} | <{entry} >{exit} {mode} {trend} {hold or '-'} | {sig_diff} | {tr_diff} |")
    # stage 0 loop too
    spy = download(cfg)
    spy2 = perturb(spy.assign(ma=np.nan), n).drop(columns="ma")
    cut = spy.index[-n]
    res = []
    for df in (spy, spy2):
        ib = ibs(df["high"], df["low"], df["close"], cfg["ibs"]["flat_value"])
        tr, _, _ = simulate(df, signal_pairs(ib, cfg["ibs"]["entry"], cfg["ibs"]["exit"]), "open")
        res.append(tr[tr["exit_date"] < cut].reset_index(drop=True))
    d0 = int((~(res[0] == res[1]).all(axis=1)).sum()) if len(res[0]) == len(res[1]) else abs(len(res[0]) - len(res[1]))
    ok_all &= d0 == 0
    out.append(f"| SPY 1993년부터 (단계 0) | <0.20 >0.80 open | - | {d0} |")
    out += ["", f"결과: {'통과' if ok_all else '실패'}", ""]
    return out


def entry_days(cfg: dict, fr: dict[str, pd.DataFrame], tr: pd.DataFrame, g: pd.DataFrame) -> list[str]:
    tr = tr.join(g[["ticker", "exec"]], on="combo")
    bad_close = int((tr.loc[tr["exec"] == "close", "entry_date"] != tr.loc[tr["exec"] == "close", "signal_date"]).sum())
    nxt = []
    for t, x in tr[tr["exec"] == "open"].groupby("ticker"):
        idx = fr[t].index
        pos = idx.get_indexer(x["signal_date"])
        nxt.append((idx[pos + 1] != x["entry_date"].to_numpy()).sum())
    bad_open = int(sum(nxt))
    return ["## 3. 진입일", "",
            f"- close: 진입일 ≠ 신호일인 거래 {bad_close}건 / {int((tr['exec'] == 'close').sum()):,}건",
            f"- open: 진입일 ≠ 신호 다음 거래일인 거래 {bad_open}건 / {int((tr['exec'] == 'open').sum()):,}건",
            f"- 결과: {'통과' if bad_close == 0 and bad_open == 0 else '실패'}", ""]


def equity_product(cfg: dict, fr: dict[str, pd.DataFrame], tr: pd.DataFrame, g: pd.DataFrame) -> list[str]:
    cap = cfg["portfolio"]["initial_capital"]
    daily = pd.read_parquet(results_dir() / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    eq = (1 + daily).cumprod() * cap
    worst, skipped = 0.0, 0
    for cid, x in tr.groupby("combo"):
        e = eq[cid]
        x = x.sort_values("exit_date")
        book = cap * np.cumprod(1 + x["ret"].to_numpy())
        # on each exit day the account is flat again: equity = capital x prod(1 + ret)
        worst = max(worst, float(np.abs(e.loc[x["exit_date"]].to_numpy() - book).max()))
        # flat days (no position at the close) keep the last booked value
        f = fr[g.at[cid, "ticker"]]
        held = np.zeros(len(e), bool)
        ix = e.index
        for en, ex in zip(ix.get_indexer(x["entry_date"]), ix.get_indexer(x["exit_date"])):
            held[en:ex] = True  # entry day .. day before exit hold a position at the close
        booked = pd.Series(book, x["exit_date"].to_numpy()).reindex(ix).ffill().fillna(cap)
        worst = max(worst, float(np.abs(e[~held] - booked[~held]).max()))
    skipped = int((g["trades"] != g["taken"]).sum())
    return ["## 4. 자산곡선 = 초기자본 × Π(1 + 거래 수익률)", "",
            f"- 768개 조합, 청산일과 포지션 없는 날 모두에서 최대 차이 {worst:.6f} 달러 (초기자본 {cap:,} 달러)",
            f"- run_portfolio 가 건너뛴 거래가 있는 조합: {skipped}개",
            f"- 결과: {'통과' if worst < 0.01 and skipped == 0 else '실패'}", ""]


def flat_days(cfg: dict, fr: dict[str, pd.DataFrame]) -> list[str]:
    spy = download(cfg)
    out = ["## 5. 고가 = 저가인 날", "", "IBS 를 계산할 수 없어 설정값(`ibs.flat_value`) "
           f"{cfg['ibs']['flat_value']} 으로 둔다. 0.5 는 가장 큰 진입 기준(0.25)보다 크고 가장 작은 청산 기준(0.50)을 넘지 않아 어느 쪽 신호도 나지 않는다.", "",
           "| 데이터 | 기간 | 날 수 |", "|---|---|---|",
           f"| SPY yfinance (단계 0) | {spy.index[0].date()} → {spy.index[-1].date()} | {int((spy['high'] == spy['low']).sum())} |"]
    for t, f in fr.items():
        v = f[f["valid"]]
        out.append(f"| {t} us_data | {v.index[0].date()} → {v.index[-1].date()} | {int((v['high'] == v['low']).sum())} |")
    return out + [""]


def park_section() -> list[str]:
    p = results_dir() / "park.csv"
    out = ["## 지수 대기 계산", ""]
    if not p.exists():
        return out + ["- park.csv 없음. python -m src.park 를 먼저 돌린다", ""]
    x = pd.read_csv(p)
    ok = (x["w0_diff"].max() < 0.01 and x["w1_diff"].max() < 0.01
          and x["conserve_idle_diff"].max(skipna=True) < 0.01 and x["conserve_switch_diff"].max(skipna=True) < 0.01
          and x["min_cash"].min() > -1e-6)
    return out + [f"- w = 0 곡선과 걸어가며 검증 곡선(포지션 없는 날) 최대 차이 {x['w0_diff'].max():.4f} 달러",
                  f"- SPY 거래만, w = 1 곡선과 SPY 보유 × (1 - 매수 비용) 최대 차이 {x['w1_diff'].max():.4f} 달러",
                  "- 대기 자산 값이 항상 1(공짜)이면, 어떤 w 로 그 자산을 대기해도 실제 w = 0 곡선과 같아야 한다"
                  f"(대기가 현금과 같은 값을 만든다는 실제 보존 검산): 최대 차이 {x['conserve_idle_diff'].max(skipna=True):.4f} 달러",
                  "- 비용 0 에서 SPY 신호가 났을 때, 들고 있던 w 몫을 그대로 두는 계산과 이름만 다른 동일 가격 종목으로 "
                  f"전액 갈아타는 계산의 최대 차이 {x['conserve_switch_diff'].max(skipna=True):.4f} 달러 (SPY 거래가 있는 변형만 계산됨. "
                  "장부(현금 + SPY + 거래 종목 = 평가금액)는 park_equity 안에서 정의상 항상 같아 따로 보지 않는다), "
                  f"현금 최소 {x['min_cash'].min():.2e}" + (" (정상)" if ok else " (오류)"), ""]


def main() -> None:
    cfg = load_config()
    fr = frames(cfg, cfg["grid"]["tickers"])
    g = pd.read_csv(results_dir() / "grid.csv", index_col="combo")
    tr = pd.read_parquet(results_dir() / "grid_trades.parquet")
    assert len(g) == len(combos(cfg))
    out = ["# 검증 결과", "", "`python -m src.checks` 가 만든다.", ""]
    out += engine_vs_replicate(cfg, fr["SPY"])
    out += lookahead(cfg, fr)
    out += entry_days(cfg, fr, tr, g)
    out += equity_product(cfg, fr, tr, g)
    out += flat_days(cfg, fr)
    out += park_section()
    text = "\n".join(out)
    (results_dir() / "checks.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
