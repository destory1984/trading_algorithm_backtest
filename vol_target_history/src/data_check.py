"""SPEC 4.2: trading days per year, missing / duplicate / non-positive values, Saturday rows, whether open / high /
low carry information, and every day the close moved more than checks.big_move.
Usage: python -m src.data_check   -> results/data_check.md, results/big_moves.csv
"""
from __future__ import annotations

import pandas as pd

from src import common as C


def main() -> None:
    cfg = C.load_config()
    out = C.results_dir()
    full = pd.read_parquet(C.hist_dir(cfg) / cfg["data"]["file"])
    r = C.raw(cfg)
    ret = r["close"].pct_change()
    big = ret[ret.abs() > cfg["checks"]["big_move"]]
    big_df = pd.DataFrame({"date": big.index.date, "return": big.values, "close": r.loc[big.index, "close"].values,
                           "prev_close": r["close"].shift(1).loc[big.index].values})
    big_df.to_csv(out / "big_moves.csv", index=False)
    per_year = r.groupby(r.index.year).size()
    sat = r[r.index.dayofweek == 5]
    ohl_same = (r["open"].eq(r["close"]) & r["high"].eq(r["close"]) & r["low"].eq(r["close"])) | r["open"].le(0)
    first_real_open = r.index[~ohl_same][0] if (~ohl_same).any() else None
    L = ["# 데이터 점검", "",
         f"- 파일 {len(full)}행, 봉인 뒤 {len(r)}행, {r.index[0].date()} → {r.index[-1].date()}. "
         f"2007-01-01 이후 행 {int((full.index >= '2007-01-01').sum())}개",
         f"- 빠진 종가 {int(r['close'].isna().sum())}, 중복 날짜 {int(full.index.duplicated().sum())}, "
         f"0 이하 종가 {int((r['close'] <= 0).sum())}",
         f"- 토요일 행 {len(sat)}개",
         f"- 시가·고가·저가가 종가와 같거나 0 인 날 {int(ohl_same.sum())}개. 시가 등이 처음 따로 있는 날: "
         f"{first_real_open.date() if first_real_open is not None else '없음'}",
         f"- 하루 ±{cfg['checks']['big_move']:.0%} 넘게 움직인 날 {len(big_df)}개", "",
         "## 하루 큰 움직임", "",
         C.md_table(big_df.assign(**{"return": big_df["return"].map(C.pct)})), "",
         "## 해마다 거래일 수", "",
         C.md_table(pd.DataFrame({"해": per_year.index, "거래일": per_year.values})), ""]
    (out / "data_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:9]))
    print(big_df.to_string(index=False))
    print(per_year.describe().round(1).to_string())
    print(per_year[per_year < 240].to_string())


if __name__ == "__main__":
    main()
