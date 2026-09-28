"""Shared pieces of the crash-rule deep dive (SPEC2.md).

Seal: install_seal() swaps krxbt's two file readers (load_prices, load_index, in every krxbt module that imported
them) for versions that cut each table at deep.end right after reading it and assert the last date. It runs in the
main process and in every worker.

Candidates: one pass over the KOSPI tickers (collect()) writes results/deep/cand.parquet with, per ticker,
  kind=seq    the rule's per-ticker sequential trades (the grid's way), data_break kept
  kind=ind    every signal as its own trade (the portfolio's way) for the rule, the rule with deep.check_stop and
              the 18 neighbour combos, data_break kept
  kind=pool   every eligible stock-day as its own trade with the rule's exit (close >= MA25 -> next open, else
              day 20 close, no stop), for the random-stock test
Downstream code drops data_break rows unless it studies them (breaks.py).
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import krxbt.data
import krxbt.frame
import krxbt.portfolio
import numpy as np
import pandas as pd
from krxbt import engine
from krxbt.frame import index_features, ticker_frame

from src.common import ROOT, load_config

_raw_prices = krxbt.data.load_prices
_raw_index = krxbt.data.load_index


def deep_end(cfg: dict) -> pd.Timestamp:
    return pd.Timestamp(cfg["deep"]["end"])


def seal(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    out = df[df.index <= end]
    assert len(out) and out.index[-1] <= end, "seal failed"
    return out


def install_seal(cfg: dict) -> None:
    end = deep_end(cfg)

    def prices(c, ticker):
        p = _raw_prices(c, ticker)
        return None if p is None else seal(p, end)

    def index(c, market):
        return seal(_raw_index(c, market), end)

    for mod in (krxbt.data, krxbt.frame, krxbt.portfolio):
        if hasattr(mod, "load_prices"):
            mod.load_prices = prices
        if hasattr(mod, "load_index"):
            mod.load_index = index


def config() -> dict:
    cfg = load_config()
    install_seal(cfg)
    return cfg


def results_dir():
    d = ROOT / "results" / "deep"
    d.mkdir(parents=True, exist_ok=True)
    return d



# ---------------------------------------------------------------- combos
def rule(cfg: dict) -> dict:
    r = dict(cfg["deep"]["rule"])
    r["stop"] = None if r["stop"] is None else float(r["stop"])
    return r


def combo_key(threshold, stop, hold, market_filter) -> str:
    s = "none" if stop is None or pd.isna(stop) else f"{float(stop):g}"
    return f"{market_filter}|{int(threshold)}|{s}|{int(hold)}"


def rule_key(cfg: dict) -> str:
    r = rule(cfg)
    return combo_key(r["threshold"], r["stop"], r["hold"], r["market_filter"])


def ind_combos(cfg: dict) -> list[tuple]:
    """(threshold, stop, hold, market_filter) run as independent candidates."""
    r = rule(cfg)
    nb = cfg["deep"]["neighbors"]
    out = [(r["threshold"], r["stop"], r["hold"], r["market_filter"]),
           (r["threshold"], float(cfg["deep"]["check_stop"]), r["hold"], r["market_filter"])]
    out += [(t, None, h, m) for t, m, h in itertools.product(nb["threshold"], nb["market_filter"], nb["hold"])]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- candidates
_W: dict = {}


def _init(cfg: dict) -> None:
    install_seal(cfg)
    _W["cfg"] = cfg
    _W["cal"] = krxbt.data.load_calendar(cfg)
    _W["idx"] = index_features(cfg)


def _one(args):
    ticker, market, delisted = args
    cfg = _W["cfg"]
    f = ticker_frame(cfg, ticker, market, _W["cal"], _W["idx"], delisted)
    if f is None:
        return None
    a = engine.arrays(f)
    with np.errstate(invalid="ignore"):
        a["target"] = a["close"] >= a["ma"]
    extra = {"disp": a["disp"], "avg_value": f["avg_value"].to_numpy(np.float64), "sig_close": a["close"]}
    r = rule(cfg)
    parts = []

    def cands(thr, mf):
        return a["eligible"] & (a["disp"] <= thr) & engine.market_mask(a, mf)

    t = engine.run(a, cfg, cands(r["threshold"], r["market_filter"]), a["target"], r["stop"], r["hold"], False, extra)
    parts.append(t.assign(kind="seq", combo=rule_key(cfg)))
    for thr, stop, hold, mf in ind_combos(cfg):
        t = engine.run(a, cfg, cands(thr, mf), a["target"], stop, hold, True, extra)
        parts.append(t.assign(kind="ind", combo=combo_key(thr, stop, hold, mf)))
    t = engine.run(a, cfg, a["eligible"], a["target"], None, r["hold"], True, extra)
    parts.append(t.assign(kind="pool", combo="pool"))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else None
    if df is None:
        return None
    df.insert(0, "ticker", ticker)
    df["market"] = market
    df["delisted"] = bool(delisted)
    for c in ("entry_px", "exit_px", "ret", "disp", "idx_disp", "avg_value", "sig_close"):
        df[c] = df[c].astype("float32")
    return df.drop(columns=["rebound"])


def collect(cfg: dict) -> pd.DataFrame:
    tk = krxbt.data.universe_tickers(cfg)
    tk = tk[tk["market"] == rule(cfg)["market"]]
    jobs = list(zip(tk.index, tk["market"], tk["delisted"]))
    with ProcessPoolExecutor(cfg["grid"]["workers"], initializer=_init, initargs=(cfg,)) as ex:
        parts = [p for p in ex.map(_one, jobs, chunksize=8) if p is not None]
    df = pd.concat(parts, ignore_index=True)
    for c in ("ticker", "market", "kind", "combo", "reason"):
        df[c] = df[c].astype("category")
    df.to_parquet(results_dir() / "cand.parquet", index=False)
    return df


def candidates(kind: str | None = None, combo: str | None = None, breaks: bool = False) -> pd.DataFrame:
    """Rows of cand.parquet; data_break rows dropped unless breaks=True."""
    filters = []
    if kind:
        filters.append(("kind", "==", kind))
    if combo:
        filters.append(("combo", "==", combo))
    df = pd.read_parquet(results_dir() / "cand.parquet", filters=filters or None)
    for c in ("ticker", "kind", "combo", "reason", "market"):
        df[c] = df[c].astype(str)
    return df if breaks else df[~df["data_break"]].reset_index(drop=True)


# ---------------------------------------------------------------- portfolio and baselines
def calendar(cfg: dict) -> pd.DatetimeIndex:
    cal = krxbt.data.load_calendar(cfg)
    return cal[cal >= pd.Timestamp(cfg["data"]["start"])]


_CLOSES: dict = {}


def closes(cfg: dict, tickers) -> pd.DataFrame:
    """Close panel for the tickers, cached across calls (the union grows as needed)."""
    cal = calendar(cfg)
    have = _CLOSES.get("df")
    need = [t for t in sorted(set(map(str, tickers))) if have is None or t not in have.columns]
    if need:
        new = krxbt.portfolio.close_panel(cfg, need, cal)
        _CLOSES["df"] = new if have is None else pd.concat([have, new], axis=1)
    return _CLOSES["df"]


def portfolio(cfg: dict, tr: pd.DataFrame) -> tuple[pd.Series, dict]:
    """5-slot portfolio of the stage2 settings: lowest disparity first, no cooldown."""
    st = cfg["stage2"]
    cal = calendar(cfg)
    return krxbt.portfolio.run_portfolio(cal, tr, closes(cfg, tr["ticker"].unique()), st["initial_capital"],
                                         int(st["max_positions"]), rank_by="disp")


def kospi_tr(cfg: dict) -> pd.Series:
    """Daily returns of the KOSPI price index plus deep.kospi_dividend spread evenly (total-return approximation)."""
    cal = calendar(cfg)
    c = krxbt.data.load_index(cfg, "KOSPI")["close"].reindex(cal).ffill()
    r = c.pct_change().fillna(0.0)
    r.iloc[1:] += (1 + cfg["deep"]["kospi_dividend"]) ** (1 / 252) - 1
    return r


def daily(eq: pd.Series) -> pd.Series:
    r = eq.pct_change()
    r.iloc[0] = 0.0
    return r


def cagr(d: pd.Series) -> float:
    last = float((1 + d).prod())
    years = (d.index[-1] - d.index[0]).days / 365.25
    return last ** (1 / years) - 1 if last > 0 else -1.0


def mdd(d: pd.Series) -> float:
    eq = (1 + d).cumprod()
    return float(min((eq / eq.cummax() - 1).min(), 0.0))


def sharpe(d: pd.Series) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def risk_weight(hold: pd.Series, target: float, iters: int = 100) -> float:
    if target <= mdd(hold):
        return 1.0
    if target >= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        w = (lo + hi) / 2
        lo, hi = (w, hi) if mdd(w * hold) > target else (lo, w)
    return (lo + hi) / 2


def stretches(cfg: dict) -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]:
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    return {"full": (None, None), "train": (None, te), "test": (te + pd.Timedelta(days=1), None)}


def cut(d: pd.Series, lo, hi) -> pd.Series:
    k = np.ones(len(d), bool)
    if lo is not None:
        k &= d.index >= lo
    if hi is not None:
        k &= d.index <= hi
    return d[k]


def compare(cfg: dict, eq: pd.Series, info: dict | None = None) -> list[dict]:
    """Strategy vs KOSPI (total-return approx): hold, risk-matched hold, exposure-matched hold, per stretch.
    Exposure = share of days holding anything (info["invested_share"] on the full stretch)."""
    ds, dk = daily(eq), kospi_tr(cfg)
    held = eq.attrs.get("held")
    rows = []
    for name, (lo, hi) in stretches(cfg).items():
        s, k = cut(ds, lo, hi), cut(dk, lo, hi)
        m = mdd(s)
        w = risk_weight(k, m)
        expo = float(cut(held, lo, hi).mean()) if held is not None else np.nan
        rows.append({"stretch": name, "first_day": s.index[0].date(), "last_day": s.index[-1].date(),
                     "cagr": cagr(s), "mdd": m, "sharpe": sharpe(s), "exposure": expo,
                     "kospi_cagr": cagr(k), "kospi_mdd": mdd(k), "risk_w": w, "risk_cagr": cagr(w * k),
                     "diff_risk": cagr(s) - cagr(w * k),
                     "exp_cagr": cagr(expo * k) if expo == expo else np.nan,
                     "diff_exp": cagr(s) - cagr(expo * k) if expo == expo else np.nan})
    return rows


def held_days(eq: pd.Series) -> pd.Series:
    """Days with at least one position, from the portfolio ledger (entry day through exit day)."""
    led = eq.attrs["ledger"]
    h = pd.Series(False, index=eq.index)
    for e, x in zip(led["entry_date"], led["exit_date"]):
        h.loc[(h.index >= e) & (h.index <= x)] = True
    return h


def run(cfg: dict, tr: pd.DataFrame) -> tuple[pd.Series, dict, list[dict]]:
    eq, info = portfolio(cfg, tr)
    eq.attrs["held"] = held_days(eq)
    return eq, info, compare(cfg, eq, info)


def trade_stats(ret) -> dict:
    r = np.asarray(ret, np.float64)
    if not len(r):
        return {"trades": 0}
    win, loss = r[r > 0], r[r < 0]
    return {"trades": len(r), "win_rate": float((r > 0).mean()), "mean_ret": float(r.mean()),
            "median_ret": float(np.median(r)), "pf": float(win.sum() / -loss.sum()) if len(loss) else np.inf,
            "worst": float(r.min()), "best": float(r.max()), "sd": float(r.std(ddof=1)) if len(r) > 1 else np.nan}


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pct(v, d=1) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.{d}f}%"
