"""SPEC 5.2 / 4: metrics per curve and stretch, plus the drawdown-matched fixed mix (EW5 x k + cash).
Run after run.py.
Usage: python -m src.metrics   -> results/metrics.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C
from src.rules import fixed
from src.sim import simulate


class Matcher:
    """EW5 x k + cash curves (monthly rebalance, net of costs), cached by (k, cash)."""

    def __init__(self, cfg, opn, cls):
        self.cfg, self.opn, self.cls = cfg, opn, cls
        self.me = cls.loc[C.month_ends(cls.index)]
        self.ew = fixed(self.me, cfg["baselines"]["EW5"]["weights"])
        self.start = pd.Timestamp(cfg["data"]["start"])
        self.cache = {}

    def curve(self, k: float, cash: float) -> pd.Series:
        kk = round(k, 6)
        if (kk, cash) not in self.cache:
            self.cache[(kk, cash)] = simulate(self.opn, self.cls, self.ew * kk, self.start, self.cfg["cost"], cash)[0]
        return self.cache[(kk, cash)]

    def match(self, target: float, cash: float, mask: np.ndarray, iters: int = 30) -> float:
        """k in [0, 1] whose curve has max drawdown `target` over `mask`; 1 if the rule is deeper than EW5."""
        if target <= C.mdd(self.curve(1.0, cash)[mask]):
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            k = (lo + hi) / 2
            lo, hi = (k, hi) if C.mdd(self.curve(k, cash)[mask]) > target else (lo, k)
        return (lo + hi) / 2


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    opn, cls = C.prices(cfg)
    daily = pd.read_parquet(out / "daily.parquet")
    tg = pd.read_csv(out / "targets.csv", parse_dates=["date"])
    tr = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    tick = cfg["data"]["tickers"]
    mt = Matcher(cfg, opn, cls)
    rows = []
    for k in daily.columns:
        info = C.parse(k)
        d_all = daily[k].dropna()
        g_all = daily[C.key(info["rule"], info["param"], info["cash"], False)].dropna()
        base = C.key(info["rule"], info["param"], cfg["cash"], True)
        t_run = tg[tg["run"] == base].set_index("date")[tick]
        changes = t_run.diff().abs().sum(axis=1).gt(1e-9)
        tr_run = tr[tr["run"] == C.key(info["rule"], info["param"], cfg["cash"], True)]
        for s in C.STRETCHES:
            msk = C.stretch_mask(cfg, d_all.index, s)
            d, g = d_all[msk], g_all[msk]
            first, last = d.index[0], d.index[-1]
            years = (last - first).days / 365.25
            # targets in force during the stretch: the last decision before `first` and every decision executed inside
            t_in = t_run[(t_run.index >= t_run.index[t_run.index < first][-1]) & (t_run.index < last)]
            prev = cls.index[cls.index.searchsorted(first) - 1]  # a decision on this day executes on `first`
            n_chg = int(changes[(changes.index >= prev) & (changes.index < last)].sum())
            turn = tr_run[(tr_run["date"] >= first) & (tr_run["date"] <= last)]["turnover"].sum()
            row = {**info, "run": k, "stretch": s, "first_day": first.date(), "last_day": last.date(),
                   "cagr": C.cagr(d), "mdd": C.mdd(d), "sharpe": C.sharpe(d),
                   "sharpe_ex": C.sharpe(d - ((1 + info["cash"]) ** (1 / 252) - 1)),
                   "calmar": C.cagr(d) / abs(C.mdd(d)) if C.mdd(d) < 0 else np.nan,
                   "gross_cagr": C.cagr(g), "cost_drag": C.cagr(g) - C.cagr(d),
                   "changes_per_year": n_chg / years, "turnover_per_year": turn / years,
                   "underwater_days": C.longest_underwater(d),
                   **{f"w_{t}": float(t_in[t].mean()) for t in tick}, "w_cash": float(1 - t_in.sum(axis=1).mean())}
            if info["net"] and info["rule"] in cfg["rules"]:
                mk = C.stretch_mask(cfg, mt.curve(1.0, info["cash"]).index, s)
                kk = mt.match(C.mdd(d), info["cash"], mk)
                m = mt.curve(kk, info["cash"])[mk]
                row.update({"match_k": kk, "match_cagr": C.cagr(m), "match_mdd": C.mdd(m)})
            rows.append(row)
    df = pd.DataFrame(rows)
    # the same-cash, same-stretch EW5 and SPY rows, for criteria 2 and 3
    ref = {(r.rule, r.cash, r.stretch): r for r in df[df["net"] & df["rule"].isin(["EW5", "SPY"])].itertuples()}
    for col, rule, field in [("ew5_sharpe", "EW5", "sharpe"), ("ew5_sharpe_ex", "EW5", "sharpe_ex"), ("ew5_cagr", "EW5", "cagr"), ("ew5_mdd", "EW5", "mdd"),
                             ("spy_cagr", "SPY", "cagr"), ("spy_mdd", "SPY", "mdd")]:
        df[col] = [getattr(ref[(rule, c, s)], field) for c, s in zip(df["cash"], df["stretch"])]
    df["diff_match"] = df["cagr"] - df["match_cagr"]
    df["diff_ew5_sharpe"] = df["sharpe"] - df["ew5_sharpe"]
    df["mdd_ratio_spy"] = df["mdd"] / df["spy_mdd"]
    df.to_csv(out / "metrics.csv", index=False)
    show = df[df["net"] & (df["cash"] == cfg["cash"]) & (df["stretch"] == "full")]
    print(show[["rule", "param", "cagr", "mdd", "sharpe", "cost_drag", "changes_per_year", "match_k", "match_cagr",
                "diff_match", "diff_ew5_sharpe", "mdd_ratio_spy"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
