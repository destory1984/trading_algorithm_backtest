"""SPEC 4.4: the neighbour values (T1 150 / 250 days, T2 8 / 12 months, T3 12 / 18%) against criteria 1 and 3, full
stretch, base dividend. Run after metrics.py.
Usage: python -m src.neighbors   -> results/neighbors.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def base_rows(cfg: dict, met: pd.DataFrame) -> pd.DataFrame:
    kr = cfg["markets"]["KR"]["dividend"]
    keep = (met["market"] == "US") | (met["dividend"].round(6) == round(kr, 6))
    return met[keep & (met["stretch"] == "full")]


def main() -> None:
    cfg = C.load_config()
    met = base_rows(cfg, pd.read_csv(C.results_dir() / "metrics.csv"))
    p = cfg["pass"]
    rows = []
    for r, rc in cfg["rules"].items():
        for m in C.MARKETS:
            for v in rc["neighbors"]:
                x = met[(met["market"] == m) & (met["rule"] == r) & (met["param"].astype(float) == float(v))].iloc[0]
                c1 = abs(x["mdd"]) <= p["mdd_ratio"] * abs(x["hold_mdd"])
                c3 = x["cagr"] > x["risk_cagr"]
                rows.append({"market": m, "rule": r, "param": v, "cagr": x["cagr"], "mdd": x["mdd"], "mdd_ratio": x["mdd_ratio"],
                             "risk_cagr": x["risk_cagr"], "diff_risk": x["diff_risk"], "c1": c1, "c3": c3, "ok": c1 and c3})
    df = pd.DataFrame(rows)
    df.to_csv(C.results_dir() / "neighbors.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
