"""SPEC2: the forward log of the US volatility-target rule (T3). No orders are ever sent.

Each run:
  1. download SPY into live.dir (krxbt.fetch_us.fetch; the research us_data is never touched)
  2. rebuild the T3 decisions from 2007 with the backtest code (src.rules.t3) for every value in live.targets.
     The last bar counts as a week-end once its week is over in New York (see week_over); the backtest itself only
     knows that when the next week's first bar exists, so this gives the same decisions a day earlier
  3. for every week-end bar not processed yet (a missed run catches up), append one row per target to
     results/live/decisions.csv and, when the main target changes, send a Telegram message
  4. self-check: every logged decision up to the last bar the backtest can already judge must equal the backtest's
  5. append one row to results/live/log.csv; on the first run of a month also send the paper-account summary
The first run fixes the start (state.json): the paper account starts at the first NYSE open after that run.
Usage: python -m src.live [--no-send]   (--no-send logs what would be sent without contacting Telegram)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src import common as C
from src import rules as R

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


# ---------------------------------------------------------------- NYSE Fridays
def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    """n-th weekday wd (0 = Monday) of the month; n = -1 is the last."""
    days = [date(y, m, d) for d in range(1, 32) if _valid(y, m, d) and date(y, m, d).weekday() == wd]
    return days[n]


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def nyse_holidays(y: int) -> set[date]:
    """NYSE full-day closures under the standard rules (special closures such as days of mourning are not known in
    advance; a missing bar on such a day shows up as "lag")."""
    out = {_easter(y) - timedelta(days=2), _nth_weekday(y, 1, 0, 2), _nth_weekday(y, 2, 0, 2),
           _nth_weekday(y, 5, 0, -1), _nth_weekday(y, 9, 0, 0), _nth_weekday(y, 11, 3, 3)}
    for m, d in ((1, 1), (6, 19), (7, 4), (12, 25)):
        if (m, d) == (6, 19) and y < 2022:  # Juneteenth closes the NYSE from 2022
            continue
        x = date(y, m, d)
        if x.weekday() == 5:
            if (m, d) == (1, 1):  # New Year's Day on a Saturday: no closure
                continue
            x -= timedelta(days=1)
        elif x.weekday() == 6:
            x += timedelta(days=1)
        out.add(x)
    return out


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    x = d + timedelta(days=1)
    while not is_trading_day(x):
        x += timedelta(days=1)
    return x


def week_over(last_bar: date, now_et: datetime) -> str:
    """"yes" if last_bar is its week's final bar, "no" if the week is still running, "lag" if the week is over but
    a later weekday of it that is not a known holiday has no bar (data not in yet)."""
    friday = last_bar + timedelta(days=4 - last_bar.weekday())
    if now_et < datetime.combine(friday, datetime.min.time(), ET).replace(hour=16, minute=15):
        return "no"
    rest = [last_bar + timedelta(days=k) for k in range(1, (friday - last_bar).days + 1)]
    return "yes" if not any(is_trading_day(d) for d in rest) else "lag"


# ---------------------------------------------------------------- data
def live_dir(cfg: dict) -> Path:
    e = os.environ.get("LIVE_DATA_DIR")
    return Path(e) if e else (C.ROOT / cfg["live"]["dir"]).resolve()


def download(cfg: dict) -> None:
    from krxbt.fetch_us import fetch
    d = live_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fetch(d, [cfg["live"]["ticker"]], 0.0)


def load_prices(cfg: dict) -> pd.DataFrame:
    r = pd.read_parquet(live_dir(cfg) / "prices" / f"{cfg['live']['ticker']}.parquet")
    r = r[~r.index.duplicated(keep="last")]
    k = r["adj_close"] / r["close"]
    return pd.DataFrame({"open": r["open"] * k, "close": r["adj_close"]})


def decisions(cfg: dict, close: pd.Series, target: float, last_is_week_end: bool) -> pd.Series:
    rc = cfg["rules"]["T3"]
    we = R.week_end(close.index)
    we[-1] = last_is_week_end
    return R.t3(close, float(target), int(rc["vol_window"]), float(rc["band"]), we)


def targets_in_force(dec: pd.Series) -> pd.Series:
    return dec.ffill().fillna(0.0)


# ---------------------------------------------------------------- telegram
def send(cfg: dict, text: str, really: bool) -> str:
    tok_env, chat_env = cfg["live"]["telegram_env"]
    tok, chat = os.environ.get(tok_env), os.environ.get(chat_env)
    if not really:
        return "보내지 않음(--no-send)"
    if not tok or not chat:
        return "보내지 못함(환경변수 없음)"
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text}, timeout=20)
        return "보냄" if r.ok else f"보내지 못함(HTTP {r.status_code})"
    except Exception as e:  # never let the token-bearing URL reach a log
        return f"보내지 못함({type(e).__name__})"


# ---------------------------------------------------------------- state and files
def out_dir() -> Path:
    e = os.environ.get("LIVE_OUT_DIR")  # tests write elsewhere
    d = Path(e) if e else C.results_dir() / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_state() -> dict:
    p = out_dir() / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(st: dict) -> None:
    (out_dir() / "state.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def append(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    p = out_dir() / name
    df = pd.DataFrame(rows)
    df.to_csv(p, mode="a", header=not p.exists(), index=False, encoding="utf-8")


def read(name: str) -> pd.DataFrame:
    p = out_dir() / name
    return pd.read_csv(p, encoding="utf-8") if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------- main
def run(really_send: bool, now_et: datetime | None = None, skip_download: bool = False) -> dict:
    cfg = C.load_config()
    lv = cfg["live"]
    now_et = now_et or datetime.now(ET)
    now_kst = now_et.astimezone(KST)
    st = load_state()
    row = {"run_kst": now_kst.strftime("%Y-%m-%d %H:%M"), "status": "", "last_bar": "", "close": np.nan, "vol20": np.nan,
           "week_end": "", "new_decisions": 0, "self_check": "", "alerts": ""}
    try:
        if not skip_download:
            download(cfg)
        px = load_prices(cfg)
        if px.index[-1].date() >= now_et.date() and now_et < datetime.combine(now_et.date(), datetime.min.time(), ET).replace(hour=16, minute=15):
            px = px.iloc[:-1]  # Yahoo shows today's bar while the market is open: not a finished bar
    except Exception as e:
        row["status"] = f"데이터 실패({type(e).__name__})"
        row["alerts"] = send(cfg, f"[기록 장치] SPY 데이터를 받지 못했다 ({type(e).__name__}).", really_send)
        append("log.csv", [row])
        return row
    last = px.index[-1].date()
    row["last_bar"] = str(last)
    row["close"] = float(px["close"].iloc[-1])
    row["vol20"] = float(px["close"].pct_change().iloc[-20:].std(ddof=1) * np.sqrt(252))
    if (now_kst.date() - last).days > int(lv["stale_days"]):
        row["status"] = "데이터 실패(오래됨)"
        row["alerts"] = send(cfg, f"[기록 장치] SPY 마지막 봉이 {last} 로 오래됐다. 계산을 건너뛴다.", really_send)
        append("log.csv", [row])
        return row
    wo = week_over(last, now_et)
    row["week_end"] = wo
    if wo == "lag":
        row["status"] = "데이터 지연"
        row["alerts"] = send(cfg, f"[기록 장치] 주가 끝났는데 SPY 봉이 {last} 까지만 있다. 다음 실행에서 다시 본다.", really_send)
        append("log.csv", [row])
        return row
    close = px["close"]
    decs = {t: decisions(cfg, close, t, wo == "yes") for t in lv["targets"]}
    main_t = lv["targets"][0]
    if "start_bar" not in st:  # first run: fix the start
        today = now_et.date()
        opens = datetime.combine(today, datetime.min.time(), ET).replace(hour=9, minute=30)
        paper_start = today if is_trading_day(today) and now_et < opens else next_trading_day(today)
        st = {"start_bar": str(last), "start_run_kst": row["run_kst"], "paper_start": str(paper_start),
              "start_target": float(targets_in_force(decs[main_t]).iloc[-1]), "processed": str(last), "summary_month": ""}
    # week-end bars after the last processed bar
    we = R.week_end(close.index)
    we[-1] = wo == "yes"
    done = pd.Timestamp(st["processed"])
    todo = [d for d in close.index[we] if d > done]
    rows, texts = [], []
    for d in todo:
        nxt = next_trading_day(d.date())
        late = now_et >= datetime.combine(nxt, datetime.min.time(), ET).replace(hour=9, minute=30)
        for t in lv["targets"]:
            dec = decs[t]
            before = float(targets_in_force(dec[dec.index < d]).iloc[-1]) if (dec.index < d).any() else 0.0
            new = dec.get(d, np.nan)
            changed = bool(np.isfinite(new))
            rows.append({"week_end_bar": str(d.date()), "target_vol": t, "old": round(before, 6),
                         "new": round(float(new), 6) if changed else "", "changed": changed,
                         "trade_day": str(nxt) if changed else "", "late": late, "logged_kst": row["run_kst"]})
            if changed and t == main_t:
                texts.append(f"[기록 장치] SPY 목표 비중 {before:.2f} → {float(new):.2f}. "
                             f"{nxt} 시가에 맞춘다(주문은 내지 않는다)." + (" [늦은 기록]" if late else ""))
    append("decisions.csv", rows)
    row["new_decisions"] = len(todo)
    if todo:
        st["processed"] = str(todo[-1].date())
    # self-check against the backtest's own week-end flags (the last bar is left out until the next bar exists)
    logged = read("decisions.csv")
    bt = {t: R.t3(close, float(t), int(cfg["rules"]["T3"]["vol_window"]), float(cfg["rules"]["T3"]["band"])) for t in lv["targets"]}
    bad = 0
    if len(logged):
        judge_until = close.index[-2] if wo == "yes" else close.index[-1]
        for _, x in logged.iterrows():
            d = pd.Timestamp(x["week_end_bar"])
            if d > judge_until:
                continue
            want = bt[float(x["target_vol"])].get(d, np.nan)
            got = float(x["new"]) if str(x["new"]) not in ("", "nan") else np.nan
            if (np.isfinite(want) != np.isfinite(got)) or (np.isfinite(want) and abs(want - got) > 1e-6):
                bad += 1
    row["self_check"] = "일치" if bad == 0 else f"불일치 {bad}건"
    if bad:
        texts.append(f"[기록 장치] 자가 점검 불일치 {bad}건. 야후가 과거 가격을 고쳤는지 본다.")
    month = now_kst.strftime("%Y-%m")
    if st.get("summary_month") != month and st["start_bar"] != str(last):
        from src.paper import summary
        texts.append(summary(cfg, px))
        st["summary_month"] = month
    row["status"] = "정상"
    row["alerts"] = " / ".join(send(cfg, t, really_send) for t in texts) if texts else ""
    append("log.csv", [row])
    if texts:
        append("alerts.csv", [{"run_kst": row["run_kst"], "text": t, "result": r}
                              for t, r in zip(texts, row["alerts"].split(" / "))])
    save_state(st)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-send", action="store_true")
    a = ap.parse_args()
    r = run(not a.no_send)
    print({k: v for k, v in r.items()})


if __name__ == "__main__":
    main()
