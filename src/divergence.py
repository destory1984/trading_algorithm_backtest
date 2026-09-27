"""RSI divergence on daily bars.

The definition is copied from koreainvest/kis_diverge.py (a 5-minute-bar test
on 13 US stocks) and applied to daily bars of every KOSPI/KOSDAQ stock:

  pivot      a low (high) strictly lower (higher) than the PIVOT_K bars on
             each side. It is only known PIVOT_K bars later.
  regular    pivot p vs the previous pivot q within LOOKBACK bars (and at
             least MIN_GAP bars apart): low[p] < low[q] and RSI[p] > RSI[q]
             = bullish. Highs the other way = bearish.
  hidden     low[p] > low[q] and RSI[p] < RSI[q] = bullish hidden.
  double     the MACD histogram diverges the same way too.
  RSI        Wilder RSI(14) on closes (first average is a simple mean).
  MACD       EMA(12) - EMA(26), signal EMA(9), EMA seeded with the first value.

The intraday "previous day's low/high" condition is left out (no meaning on
daily bars).

Entry: next open after the confirmation day (pivot + PIVOT_K), like the BNF
test. Exit: close of holding day N (no target, optional stop), same kernel,
costs and data-break rules as the BNF grid. Every signal is its own trade.
Returns are compared with the average return of all eligible stocks entered
on the same day (date-matched excess), so market swings cancel out.
"Look-ahead" rows buy at the pivot's own close, which needs the next
PIVOT_K bars to be known: it shows how much the confirmation delay costs.

Usage:  python -m src.divergence            # full run, writes results/divergence.md
        python -m src.divergence --show 005930   # list one ticker's divergences
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
import pandas as pd

from .common import load_calendar, load_config, results_dir
from .simulate import arrays, candidates, net_return, run_kernel

PIVOT_K = 3
LOOKBACK = 30
MIN_GAP = 4
RSI_N = 14
HOLDS = [5, 10, 20]
STOPS = [None, -0.08]
BOOT = 1000


@numba.njit(cache=True)
def rsi_wilder(closes, period):
    n = len(closes)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    au = 0.0
    ad = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        au += max(d, 0.0)
        ad += max(-d, 0.0)
    au /= period
    ad /= period
    out[period] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    for i in range(period + 1, n):
        d = closes[i] - closes[i - 1]
        au = (au * (period - 1) + max(d, 0.0)) / period
        ad = (ad * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


def macd_hist(closes: np.ndarray) -> np.ndarray:
    s = pd.Series(closes)
    line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return (line - line.ewm(span=9, adjust=False).mean()).to_numpy()


def pivots(values: np.ndarray, low: bool, k: int = PIVOT_K) -> np.ndarray:
    n = len(values)
    if n < 2 * k + 1:
        return np.array([], dtype=np.int64)
    w = np.lib.stride_tricks.sliding_window_view(values, 2 * k + 1)
    mid = w[:, k]
    side = np.delete(w, k, axis=1)
    ok = (mid[:, None] < side).all(axis=1) if low else (mid[:, None] > side).all(axis=1)
    return np.nonzero(ok)[0] + k


def find(v: pd.DataFrame) -> pd.DataFrame:
    """Divergences on traded bars v (positions are row numbers in v)."""
    closes = v["close"].to_numpy(np.float64)
    rsi = rsi_wilder(closes, RSI_N)
    hist = macd_hist(closes)
    rows = []
    for up in (True, False):
        price = v["low" if up else "high"].to_numpy(np.float64)
        piv = [p for p in pivots(price, up) if not np.isnan(rsi[p])]
        for n, p in enumerate(piv):
            q = next((q for q in reversed(piv[:n]) if MIN_GAP <= p - q <= LOOKBACK), None)
            if q is None:
                continue
            lower = price[p] < price[q] if up else price[p] > price[q]
            weaker = rsi[p] > rsi[q] if up else rsi[p] < rsi[q]
            stronger = rsi[p] < rsi[q] if up else rsi[p] > rsi[q]
            if lower and weaker:
                hidden = False
                macd = hist[p] > hist[q] if up else hist[p] < hist[q]
            elif price[p] != price[q] and not lower and stronger:
                hidden = True
                macd = hist[p] < hist[q] if up else hist[p] > hist[q]
            else:
                continue
            rows.append((p + PIVOT_K, p, q, up, hidden, bool(macd), rsi[p]))
    return pd.DataFrame(rows, columns=["confirm", "pivot", "prev", "up", "hidden", "macd", "rsi"])


SETS = {  # name -> (up, hidden, need_macd, market_filter)
    "상승 보통": (True, False, False, "none"),
    "상승 보통 + MACD(이중)": (True, False, True, "none"),
    "상승 히든": (True, True, False, "none"),
    "상승 보통 + 동반급락≤97": (True, False, False, "crash97"),
    "하락 보통": (False, False, False, "none"),
    "하락 히든": (False, True, False, "none"),
}

_W: dict = {}


def _init(cfg):
    from .grid import _init as grid_init
    grid_init(cfg)


def _trades(cfg, f, a, cand, hold, stop, on_close=False):
    ru = cfg["rules"]
    no_target = np.zeros(len(f), np.bool_)
    sig, ent, ext, epx, xpx, _, rsn, _ = run_kernel(
        a["open"], a["high"], a["low"], a["close"], no_target, a["valid"], cand,
        stop if stop is not None else 0.0, int(hold), on_close, False, 1.0, 1.0, True)
    dates = f.index.to_numpy()
    ok = ~(a["break_cum"][ext] > a["break_cum"][ent])
    return sig[ok], ent[ok], dates[ent[ok]], net_return(cfg, epx[ok], xpx[ok], dates[ext[ok]])


def _run_ticker(args):
    from .grid import _W as G
    from krxbt.frame import ticker_frame
    ticker, market, delisted = args
    cfg = G["cfg"]
    f = ticker_frame(cfg, ticker, market, G["cal"], G["idx"], delisted)
    if f is None:
        return None
    a = arrays(f)
    v_pos = np.nonzero(a["valid"])[0]
    v = f.iloc[v_pos]
    d = find(v)
    out, base = [], []
    # baseline: every eligible day
    for hold in HOLDS:
        _, _, ed, r = _trades(cfg, f, a, a["eligible"].copy(), hold, None)
        if len(r):
            s = pd.DataFrame({"entry_date": ed, "ret": r}).groupby("entry_date")["ret"].agg(["sum", "count"])
            s["hold"] = hold
            base.append(s.reset_index())
    if len(d):
        d["confirm_f"] = v_pos[d["confirm"].clip(upper=len(v_pos) - 1)]
        d.loc[d["confirm"] >= len(v_pos), "confirm_f"] = -1
        d["pivot_f"] = v_pos[d["pivot"]]
        for name, (up, hidden, need_macd, mf) in SETS.items():
            m = (d["up"] == up) & (d["hidden"] == hidden) & (d["confirm_f"] >= 0)
            if need_macd:
                m &= d["macd"]
            sel = d[m]
            if sel.empty:
                continue
            filt = candidates(a, 1e9, mf)  # eligibility + market filter only
            for look in (False, True):
                if look and name not in ("상승 보통", "상승 히든"):
                    continue
                pos = sel["pivot_f"].to_numpy() if look else sel["confirm_f"].to_numpy()
                cand = np.zeros(len(f), np.bool_)
                cand[pos] = True
                cand &= filt
                for hold in HOLDS:
                    for stop in (STOPS if not look else [None]):
                        sg, _, ed, r = _trades(cfg, f, a, cand, hold, stop, on_close=look)
                        if len(r):
                            out.append(pd.DataFrame({
                                "set": name + (" (저점 미리 앎)" if look else ""), "hold": hold,
                                "stop": np.nan if stop is None else stop,
                                "signal_date": f.index.to_numpy()[sg], "entry_date": ed, "ret": r}))
    tr = pd.concat(out, ignore_index=True) if out else None
    if tr is not None:
        tr["ticker"], tr["market"] = ticker, market
    return tr, (pd.concat(base, ignore_index=True) if base else None)


def run(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .grid import universe_tickers
    tk = universe_tickers(cfg)
    jobs = list(zip(tk.index, tk["market"], tk["delisted"]))
    t0 = time.time()
    trades, base = [], []
    with ProcessPoolExecutor(cfg["grid"]["workers"], initializer=_init, initargs=(cfg,)) as ex:
        for i, res in enumerate(ex.map(_run_ticker, jobs, chunksize=8), 1):
            if res is None:
                continue
            tr, b = res
            if tr is not None:
                trades.append(tr)
            if b is not None:
                base.append(b)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} tickers, {time.time() - t0:.0f}s")
    tr = pd.concat(trades, ignore_index=True)
    b = pd.concat(base, ignore_index=True).groupby(["hold", "entry_date"])[["sum", "count"]].sum()
    b["base"] = b["sum"] / b["count"]
    return tr, b.reset_index()


def _boot_ci(x: pd.DataFrame, rng) -> tuple[float, float]:
    """90% interval of the mean excess, resampling signal dates (signals cluster on days)."""
    g = x.groupby("signal_date")["excess"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(g) < 5:
        return np.nan, np.nan
    idx = rng.integers(0, len(g), size=(BOOT, len(g)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def summarize(cfg, tr: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    tr = tr.merge(base[["hold", "entry_date", "base"]], on=["hold", "entry_date"], how="left")
    tr["excess"] = tr["ret"] - tr["base"]
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    rng = np.random.default_rng(7)
    rows = []
    for (name, hold, stop), x in tr.groupby(["set", "hold", "stop"], dropna=False, sort=False):
        lo, hi = _boot_ci(x, rng)
        a, b = x[x["entry_date"] <= te], x[x["entry_date"] > te]
        rows.append({"set": name, "hold": hold, "stop": stop, "n": len(x), "days": x["signal_date"].nunique(),
                     "win": (x["ret"] > 0).mean(), "mean": x["ret"].mean(), "median": x["ret"].median(),
                     "excess": x["excess"].mean(), "ci_lo": lo, "ci_hi": hi,
                     "excess_train": a["excess"].mean(), "excess_test": b["excess"].mean(),
                     "worst": x["ret"].min()})
    return pd.DataFrame(rows)


def pct(x, d=2):
    return "" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def report(cfg, s: pd.DataFrame, base: pd.DataFrame) -> str:
    te = cfg["stage2"]["train_end"][:4]
    bm = base.groupby("hold").apply(lambda g: g["sum"].sum() / g["count"].sum(), include_groups=False)
    out = ["# RSI 다이버전스 백테스트 (일봉)", "",
           f"- 기간 {cfg['data']['start']} → {load_calendar(cfg)[-1].date()}, 코스피·코스닥, BNF 백테스트와 같은 종목·거래대금·데이터 끊김 규칙.",
           f"- 정의는 koreainvest `kis_diverge.py` 그대로(저점·고점 앞뒤 {PIVOT_K}봉, 앞 점 {LOOKBACK}봉 안·{MIN_GAP}봉 넘게, RSI({RSI_N}) 와일더, MACD 12·26·9). "
           "전날 저가·고가 조건은 일봉에선 뜻이 없어 뺐다.",
           f"- 신호는 저점 뒤 {PIVOT_K}일째(확인일) 종가에 나고, 다음 날 시가에 산다. 보유일 N 일째 종가에 판다. 비용은 BNF 와 같다(매도세는 그해 세율).",
           "- 초과 = 그 거래 수익률 − 같은 날 시가에 산 모든 종목의 평균 수익률(같은 보유일). 시장이 오르내린 몫을 뺀 값이다.",
           "- 90% 구간은 신호 날짜를 다시 뽑아(1,000번) 잰 초과 평균의 범위다. 두 끝이 모두 0 위여야 우연이 아니라고 본다.",
           "- 「저점 미리 앎」은 저점 날 종가에 산다고 친 것이다. 뒤 3일을 봐야 저점인 줄 아니 실제로는 못 하는 매매다.",
           "", "모든 종목을 아무 날이나 샀을 때(기준) 평균: " + ", ".join(f"{h}일 {pct(bm[h])}" for h in bm.index), ""]
    for stop in [np.nan, -0.08]:
        x = s[s["stop"].isna()] if pd.isna(stop) else s[np.isclose(s["stop"].fillna(0), stop)]
        if x.empty:
            continue
        out += [f"## 손절 {'없음' if pd.isna(stop) else f'{stop:.0%}'}", "",
                "| 신호 | 보유 | 거래 | 신호 난 날 | 승률 | 평균 | 중앙값 | 초과 | 초과 90% 구간 | 초과 ~" + te + " | 초과 " + str(int(te) + 1) + "~ | 최악 |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            out.append(f"| {r['set']} | {r['hold']}일 | {r['n']:,} | {r['days']:,} | {r['win']:.1%} | {pct(r['mean'])} | "
                       f"{pct(r['median'])} | {pct(r['excess'])} | {pct(r['ci_lo'])} ~ {pct(r['ci_hi'])} | "
                       f"{pct(r['excess_train'])} | {pct(r['excess_test'])} | {pct(r['worst'], 1)} |")
        out.append("")
    return "\n".join(out)


def show(cfg, ticker: str) -> None:
    from .grid import universe_tickers
    from krxbt.frame import index_features, ticker_frame
    info = universe_tickers(cfg).loc[ticker]
    f = ticker_frame(cfg, ticker, info["market"], load_calendar(cfg), index_features(cfg), bool(info["delisted"]))
    v = f[f["valid"]]
    d = find(v)
    rsi = rsi_wilder(v["close"].to_numpy(np.float64), RSI_N)
    for _, r in d.tail(8).iterrows():
        p, q = int(r["pivot"]), int(r["prev"])
        col = "low" if r["up"] else "high"
        kind = ("상승" if r["up"] else "하락") + (" 히든" if r["hidden"] else "") + (" +MACD" if r["macd"] else "")
        conf = v.index[int(r["confirm"])].date() if r["confirm"] < len(v) else "-"
        print(f"{kind}: 앞 점 {v.index[q].date()} {col} {v[col].iloc[q]:,.0f} RSI {rsi[q]:.1f} → "
              f"저점/고점 {v.index[p].date()} {v[col].iloc[p]:,.0f} RSI {rsi[p]:.1f}, 확인 {conf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    a = ap.parse_args()
    cfg = load_config()
    if a.show:
        show(cfg, a.show)
        return
    tr, base = run(cfg)
    rd = results_dir()
    tr.to_parquet(rd / "divergence_trades.parquet", index=False)
    base.to_parquet(rd / "divergence_baseline.parquet", index=False)
    s = summarize(cfg, tr, base)
    s.to_csv(rd / "divergence_summary.csv", index=False, encoding="utf-8-sig")
    (rd / "divergence.md").write_text(report(cfg, s, base), encoding="utf-8")
    print(f"divergence trades: {len(tr):,}")
    print(report(cfg, s, base))


if __name__ == "__main__":
    main()
