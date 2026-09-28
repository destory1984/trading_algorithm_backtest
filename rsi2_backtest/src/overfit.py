"""Stage 6 (part 1): Deflated Sharpe and the random-entry ("monkey") test for each walk-forward candidate.

Deflated Sharpe (Bailey and Lopez de Prado 2014, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality", J. Portfolio Management 40(5), 94-107, eqs. for SR0 and DSR; written here from the
paper, no code copied):
  SR0 = sqrt(V) x ((1 - g) x Z(1 - 1/N) + g x Z(1 - 1/(N e))),  g = Euler-Mascheroni, Z = inverse normal CDF
  DSR = Phi((SR - SR0) x sqrt(T - 1) / sqrt(1 - skew x SR + (kurt - 1) / 4 x SR^2))
  SR, skew, kurt (not excess) of the candidate's daily returns (us costs, data.start -> data.end, flat days included);
  N = overfit.trials summed = 768 + 48 + 216 = 1,032; V = variance of the daily Sharpe of this grid's 216 combos (the
  other folders' trial Sharpes are not read, to keep this folder self-contained). This primary choice (N = 1,032,
  V from the 216 combos) was fixed in the plan before any result was seen; it is the decision value (primary = True).
Sensitivity rows (primary = False, variant "<open|close>|<case>", DSR only, no monkey test), shown so readers see how
the verdict moves; they never replace the primary:
  n216_v216    N = 216 (this grid only), V from the 216 combos
  n1032_v1032  N = 1,032, V from all 1,032 trials' daily Sharpes: ibs_backtest grid_daily (768 combos),
               ibs_lev_backtest lev_daily (the 48 combos with costs) and this grid's 216; read as data from the sibling
               folders' results (a missing file gives NaN and a note)
  n1032_vqqq54 N = 1,032, V from this grid's 54 QQQ combos
Random entry (as the original's monkey test): as many trades as the candidate, entry days drawn without replacement
from the days its trend filter allows, holding bars drawn with replacement from the candidate's own trades, same
execution and costs; 2,000 runs; percentile = share of runs whose PF is below the candidate's PF. The allowed days are
the rule's own entry array with the RSI threshold set to infinity (eligible, RSI known, and close vs SMA200 for
above/below), so the pool follows the strategy's definition. Row stage0: the original rule on the original CSV (5bp),
next to the original's own monkey pct_pf.

Usage:  python -m src.overfit      (after src.walkforward and src.stage0)
Writes  results/overfit.csv
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .common import ROOT, load_config, profit_factor, results_dir, with_costs
from .indicators import signals
from .replicate import load_csv, sample, with_indicators
from .strategy import arrays, frames, rule

EULER = 0.5772156649015329
ND = NormalDist()
IBS_DAILY = ROOT.parent / "ibs_backtest" / "results" / "grid_daily.parquet"
LEV_DAILY = ROOT.parent / "ibs_lev_backtest" / "results" / "lev_daily.parquet"
LEV_ROWS = ROOT.parent / "ibs_lev_backtest" / "results" / "lev.csv"
SENS_VARIANTS = ("open", "close")


def expected_max_z(n: int) -> float:
    """Approximate expected maximum of n independent standard normals (the paper's SR0 factor)."""
    return (1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e))


def dsr_from_moments(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> tuple[float, float]:
    sr0 = math.sqrt(var_sr) * expected_max_z(n_trials)
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return sr0, ND.cdf(z)


def deflated_sharpe(daily: np.ndarray, n_trials: int, var_sr: float) -> dict:
    x = np.asarray(daily, np.float64)
    x = x[np.isfinite(x)]
    m, s0 = x.mean(), x.std(ddof=0)
    sr = float(m / x.std(ddof=1))
    skew = float(np.mean((x - m) ** 3) / s0 ** 3)
    kurt = float(np.mean((x - m) ** 4) / s0 ** 4)
    sr0, dsr = dsr_from_moments(sr, len(x), skew, kurt, n_trials, var_sr)
    return {"sr_daily": sr, "T": len(x), "skew": skew, "kurt": kurt, "n_trials": n_trials, "var_sr": var_sr,
            "sr0": sr0, "dsr": dsr}


def pool_days(a: dict, cfg: dict, trend: str) -> np.ndarray:
    """Days the candidate's trend filter allows: the rule's entry array with no RSI threshold."""
    return np.flatnonzero(rule(a, cfg, np.inf, trend, "green2")[0])


def trial_sharpes(daily: pd.DataFrame) -> pd.Series:
    """Daily Sharpe of every column (mean / std, ddof = 1, NaN days skipped), finite values only."""
    sr = daily.mean() / daily.std(ddof=1)
    return sr[np.isfinite(sr)]


def sibling_sharpes() -> tuple[pd.Series | None, str]:
    """The 768 + 48 trial Sharpes of the two sibling folders, read from their result files, or (None, note)."""
    missing = [p.parent.parent.name + "/results/" + p.name for p in (IBS_DAILY, LEV_DAILY, LEV_ROWS) if not p.exists()]
    if missing:
        return None, "파일 없음: " + ", ".join(missing)
    ibs = pd.read_parquet(IBS_DAILY)
    lev = pd.read_parquet(LEV_DAILY)
    rows = pd.read_csv(LEV_ROWS)
    lev = lev[[str(c) for c in rows.loc[rows["costs"] == "with", "combo"]]]
    s1, s2 = trial_sharpes(ibs), trial_sharpes(lev)
    if len(s1) != 768 or len(s2) != 48:
        return None, f"시행 수가 다름: ibs_backtest {len(s1)}, ibs_lev_backtest {len(s2)}"
    return pd.concat([s1, s2], ignore_index=True), ""


def trade_returns(o, c, s, bars, mode: str, buy_cost: float, sell_keep: float) -> np.ndarray:
    """open: buy o[s+1], sell o[s+1+bars]; close: buy c[s], sell c[s+bars]; the exit is capped at the last bar."""
    n = len(c)
    s, bars = np.asarray(s), np.asarray(bars)
    if mode == "open":
        e = s + 1
        x = np.minimum(e + bars, n - 1)
        return o[x] * sell_keep / (o[e] * buy_cost) - 1
    if mode == "close":
        x = np.minimum(s + bars, n - 1)
        return c[x] * sell_keep / (c[s] * buy_cost) - 1
    raise ValueError(mode)


def monkey_pfs(o, c, pool, n_trades: int, bars, mode: str, buy_cost: float, sell_keep: float, runs: int,
               rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(pool)
    if mode == "open":
        pool = pool[pool + 1 < len(c)]
    if n_trades > len(pool):
        raise ValueError(f"{n_trades} trades but only {len(pool)} allowed days")
    out = np.empty(runs)
    for k in range(runs):
        s = rng.choice(pool, size=n_trades, replace=False)
        d = rng.choice(np.asarray(bars), size=n_trades, replace=True)
        out[k] = profit_factor(trade_returns(o, c, s, d, mode, buy_cost, sell_keep))
    return out


def main() -> None:
    cfg = load_config()
    rd = results_dir()
    ov = cfg["overfit"]
    n_trials = int(sum(ov["trials"].values()))
    g = pd.read_csv(rd / "grid.csv")
    g = g[g["costs"] == "us"].set_index("combo")
    daily = pd.read_parquet(rd / "grid_daily.parquet")
    daily.columns = daily.columns.astype(int)
    sr_all = trial_sharpes(daily)
    var_sr = float(sr_all.var(ddof=1))
    tr = pd.read_parquet(rd / "grid_trades.parquet")
    tr = tr[tr["costs"] == "us"]
    cands = pd.read_csv(rd / "candidates.csv")
    fr = frames(cfg, cfg["grid"]["tickers"])
    bc, sk = cost_factors(with_costs(cfg, "us"))
    rows = []
    for k, c in enumerate(cands.itertuples(index=False)):
        a = arrays(fr[c.ticker])
        ix = pd.DatetimeIndex(a["dates"])
        x = tr[tr["combo"] == c.combo]
        e, xx = ix.get_indexer(x["entry_date"]), ix.get_indexer(x["exit_date"])
        rng = np.random.default_rng([ov["seed"], k])
        pfs = monkey_pfs(a["open"], a["close"], pool_days(a, cfg, c.trend), len(x), xx - e, c.exec, bc, sk,
                         ov["monkey_runs"], rng)
        pf = profit_factor(x["ret"].to_numpy())
        rows.append({"variant": c.variant, "primary": True, "case": "primary", "combo": c.combo, "label": c.label,
                     "ticker": c.ticker, "exec": c.exec,
                     "sharpe": float(g.at[c.combo, "sharpe"]), "trades": len(x), "profit_factor": pf,
                     **deflated_sharpe(daily[c.combo].to_numpy(), n_trials, var_sr),
                     "monkey_runs": ov["monkey_runs"], "monkey_pctile": float((pfs < pf).mean()),
                     "monkey_median_pf": float(np.median(pfs))})
    smp = sample(with_indicators(cfg, load_csv(cfg)))
    rt = pd.read_csv(rd / "replicate_trades.csv")
    rt = rt[(rt["variant"] == "original") & (rt["part"] == "ALL")]
    # stage 0's own entry array (replicate.run_part: first row never signals) with no RSI threshold
    elig = np.ones(len(smp), bool)
    elig[0] = False
    ind = {k: smp[k].to_numpy(np.float64) for k in ("rsi", "close", "ma", "sma5")}
    ind["green"] = smp["green"].to_numpy(bool)
    pool = np.flatnonzero(signals(ind, np.inf, "above", "green2", elig, cfg["rsi"]["exit_above"])[0])
    obc, osk = cost_factors(with_costs(cfg, "orig"))
    rng = np.random.default_rng([ov["seed"], len(rows)])
    pfs = monkey_pfs(smp["open"].to_numpy(np.float64), smp["close"].to_numpy(np.float64), pool, len(rt),
                     (rt["x"] - rt["e"]).to_numpy(), "open", obc, osk, ov["monkey_runs"], rng)
    pf = profit_factor(rt["ret"].to_numpy())
    meta = pd.read_csv(rd / "stage0_meta.csv", dtype=str).set_index("key")["value"]
    rows.append({"variant": "stage0", "primary": True, "case": "primary", "label": "원본 규칙, 원본 CSV 전체 (편도 5bp)",
                 "ticker": "QQQ", "exec": "open", "trades": len(rt), "profit_factor": pf, "monkey_runs": ov["monkey_runs"],
                 "monkey_pctile": float((pfs < pf).mean()), "monkey_median_pf": float(np.median(pfs)),
                 "orig_monkey_pct_pf": float(meta["monkey_pct_pf"])})
    # sensitivity rows (not the decision value)
    sib, sib_note = sibling_sharpes()
    qqq = trial_sharpes(daily[g.index[g["ticker"] == "QQQ"]])
    cases = [("n216_v216", int(ov["trials"]["rsi2_backtest"]), sr_all, "이 그리드 216개", ""),
             ("n1032_v1032", n_trials, None if sib is None else pd.concat([sib, sr_all], ignore_index=True),
              "세 폴더 1,032개", sib_note),
             ("n1032_vqqq54", n_trials, qqq, f"이 그리드 QQQ {len(qqq)}개", "")]
    for c in cands[cands["variant"].isin(SENS_VARIANTS)].itertuples(index=False):
        for case, n, srs, src, note in cases:
            base = {"variant": f"{c.variant}|{case}", "primary": False, "case": case, "combo": c.combo, "label": c.label,
                    "ticker": c.ticker, "exec": c.exec, "sharpe": float(g.at[c.combo, "sharpe"]), "v_source": src,
                    "v_trials": 0 if srs is None else len(srs), "note": note}
            if srs is None:
                rows.append({**base, "n_trials": n, "var_sr": np.nan, "dsr": np.nan})
            else:
                rows.append({**base, **deflated_sharpe(daily[c.combo].to_numpy(), n, float(srs.var(ddof=1)))})
    out = pd.DataFrame(rows)
    out.to_csv(rd / "overfit.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"trials {n_trials}, V[SR] {var_sr:.3e}")
    print(out[["variant", "primary", "sharpe", "sr_daily", "T", "n_trials", "var_sr", "sr0", "dsr", "profit_factor",
               "monkey_pctile", "monkey_median_pf"]].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
