"""SPEC 4.2: per-ticker data check. Usage: python -m src.data_check -> results/data_check.md, big_moves.csv"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    rows, moves, dv_year = [], [], {}
    for t in cfg["tickers"]:
        df = C.raw(cfg, t)
        px = df[["open", "high", "low", "close", "adj_close"]]
        ret = df["adj_close"].pct_change()
        big = ret[ret.abs() > cfg["checks"]["big_move"]]
        for day, v in big.items():
            moves.append({"ticker": t, "date": day.date(), "ret": v, "raw_ret": df["close"].pct_change()[day],
                          "volume": df.loc[day, "volume"]})
        ratio = df["adj_close"] / df["close"]
        rows.append({"종목": t, "첫날": df.index[0].date(), "행 수": len(df),
                     "빠진 값": int(px.isna().sum().sum()), "중복 날짜": int(df.index.duplicated().sum()),
                     "0 이하 값": int((px <= 0).sum().sum()), "거래량 0 인 날": int((df["volume"] <= 0).sum()),
                     "그중 1997 이후": int((df["volume"][df.index >= cfg["data"]["start"]] <= 0).sum()),
                     "시가가 고저 밖": int(((df["open"] > df["high"] * 1.0001) | (df["open"] < df["low"] * 0.9999)).sum()),
                     "±15% 넘는 날": len(big),
                     "adj/close 가 줄어든 날": int((ratio.diff() < -1e-9).sum())})
        dv = (df["close"] * df["volume"]).rolling(cfg["rule"]["dv_window"]).mean()
        dv_year[t] = dv.groupby(dv.index.year).median() / 1e6
    mv = pd.DataFrame(moves)
    mv.to_csv(out / "big_moves.csv", index=False)
    dvy = pd.DataFrame(dv_year).round(2)
    dvy.to_csv(out / "dollar_volume_by_year.csv")
    below = (dvy < cfg["rule"]["min_dollar_volume"] / 1e6).sum(axis=1)
    md = ["# 데이터 점검", "", C.md_table(pd.DataFrame(rows)), "",
          f"## 하루 ±{cfg['checks']['big_move']:.0%} 넘게 움직인 날 (배당 반영 종가)", ""]
    if len(mv):
        m2 = mv.copy()
        m2["ret"] = m2["ret"].map(C.pct)
        m2["raw_ret"] = m2["raw_ret"].map(C.pct)
        md += [C.md_table(m2), ""]
    md += ["## 해마다 20일 평균 거래대금 중앙값(백만 달러)과 하한(1) 아래인 종목 수", "",
           C.md_table(dvy.assign(**{"하한 아래 종목 수": below}).reset_index(names="해")), ""]
    (out / "data_check.md").write_text("\n".join(md), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(mv.to_string(index=False))
    print(below.to_string())


if __name__ == "__main__":
    main()
