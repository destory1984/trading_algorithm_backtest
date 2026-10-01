"""SPEC 4.7 / criterion 7: signal-day concentration, yearly results and the big declines.
Usage: python -m src.crashes   -> results/signal_days.csv, drop_days.csv, yearly.csv, stress.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    trades = pd.read_parquet(out / "trades.parquet")
    daily = pd.read_parquet(out / "daily.parquet")
    t = trades[trades["variant"] == "95"]
    days = t.groupby("signal_day")["ret"].agg(trades="size", total="sum", mean="mean").sort_values("total", ascending=False)
    days.to_csv(out / "signal_days.csv")
    k = cfg["pass"]["drop_days"]
    rest = t[~t["signal_day"].isin(days.index[:k])]
    many = days.index[days["trades"] >= 5]
    drop = pd.DataFrame([{"signal_days": len(days), "trades": len(t), "mean_all": t["ret"].mean(),
                          "mean_without_best": rest["ret"].mean(), "trades_dropped": len(t) - len(rest),
                          "best_total": days["total"].iloc[:k].sum(), "all_total": days["total"].sum(),
                          "days_with_5plus": len(many), "trades_on_5plus_days": int(days.loc[many, "trades"].sum()),
                          "mean_on_5plus_days": t[t["signal_day"].isin(many)]["ret"].mean(),
                          "mean_on_other_days": t[~t["signal_day"].isin(many)]["ret"].mean()}])
    drop.to_csv(out / "drop_days.csv", index=False)

    s, h = daily["95"].dropna(), daily["95|hold"].dropna()
    yr = pd.DataFrame({"rule": (1 + s).groupby(s.index.year).prod() - 1, "hold": (1 + h).groupby(h.index.year).prod() - 1,
                       "trades": t.groupby(t["exit_day"].dt.year).size(),
                       "mean": t.groupby(t["exit_day"].dt.year)["ret"].mean()})
    yr.to_csv(out / "yearly.csv")

    rows = []
    for name, (lo, hi) in cfg["stress"].items():
        sp, hp = C.cut(s, lo, hi), C.cut(h, lo, hi)
        tp = t[(t["entry_day"] >= lo) & (t["entry_day"] <= hi)]
        rows.append({"window": name, "rule_ret": (1 + sp).prod() - 1, "rule_mdd": C.mdd(sp),
                     "hold_ret": (1 + hp).prod() - 1, "hold_mdd": C.mdd(hp), "trades": len(tp),
                     "mean": tp["ret"].mean(), "win": (tp["ret"] > 0).mean()})
    st = pd.DataFrame(rows)
    st.to_csv(out / "stress.csv", index=False)
    pd.set_option("display.width", 250)
    print(drop.round(4).T.to_string())
    print(days.head(5).round(4).to_string())
    print(days.tail(5).round(4).to_string())
    print(yr.round(4).to_string())
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
