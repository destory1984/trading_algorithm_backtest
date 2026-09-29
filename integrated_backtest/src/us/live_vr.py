"""SPEC2 5: the forward log of value rebalancing on TQQQ. No orders are ever sent.

Each run:
  1. download TQQQ into live.dir (krxbt.fetch_us.fetch; the research us_data is never touched). A bar dated today
     before 16:15 New York time is dropped (Yahoo shows the unfinished bar while the market is open)
  2. the first run fixes the start (state.json): the paper account buys at the first NYSE open after that run
  3. from the paper start, rerun the rule (vr_live_rule.run) for every band in live.bands on all bars so far
  4. every rebalance day not logged yet (a missed run catches up) is appended to results/us/live/decisions.csv; an
     order of the main band sends a Telegram message
  5. self-check: every logged decision must equal the rerun's (same bar, side, amount within 1e-6 relative)
  6. one row in results/us/live/log.csv; the first run of a month also sends the paper-account summary
Usage: python -m src.us.live_vr [--no-send]   (--no-send logs what would be sent without contacting Telegram)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import common as C
from src.us import vr_live_rule as VR
from src.us.nyse import ET, KST, is_trading_day, next_trading_day


def live_dir(cfg: dict) -> Path:
    return C.env_dir(cfg["live"]["dir"], "LIVE_DATA_DIR")


def download(cfg: dict) -> None:
    from krxbt.fetch_us import fetch
    d = live_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    fetch(d, [cfg["live"]["ticker"]], 0.0)


def load_prices(cfg: dict) -> pd.DataFrame:
    r = pd.read_parquet(live_dir(cfg) / "prices" / f"{cfg['live']['ticker']}.parquet")
    r = r[~r.index.duplicated(keep="last")].sort_index()
    k = r["adj_close"] / r["close"]
    return pd.DataFrame({"open": r["open"] * k, "close": r["adj_close"]})


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


def out_dir() -> Path:
    e = os.environ.get("LIVE_OUT_DIR")
    d = Path(e) if e else C.results_dir("us") / "live"
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
    pd.DataFrame(rows).to_csv(p, mode="a", header=not p.exists(), index=False, encoding="utf-8")


def read(name: str) -> pd.DataFrame:
    p = out_dir() / name
    return pd.read_csv(p, encoding="utf-8") if p.exists() else pd.DataFrame()


def paper_px(px: pd.DataFrame, st: dict) -> pd.DataFrame:
    return px[px.index >= pd.Timestamp(st["paper_start"])]


def summary(cfg: dict, px: pd.DataFrame, st: dict) -> str:
    p = paper_px(px, st)
    r = VR.run(cfg, p, cfg["live"]["bands"][0])
    cap = cfg["us"]["capital"]
    d, h = C.from_equity(r["equity"], cap), C.from_equity(VR.hold(cfg, p), cap)
    pool = r["state"]["pool"] / r["equity"].iloc[-1]
    return (f"[VR 기록 장치] {st['paper_start']} 시작 뒤 {len(p)}거래일: 종이 계좌 {(1 + d).prod() - 1:+.1%}(최대 낙폭 {C.mdd(d):.1%}), "
            f"TQQQ 보유 {(1 + h).prod() - 1:+.1%}(최대 낙폭 {C.mdd(h):.1%}). 현금 풀 {pool:.0%}. 주문은 내지 않는다.")


def run(really_send: bool, now_et: datetime | None = None, skip_download: bool = False) -> dict:
    cfg = C.load_config()
    lv = cfg["live"]
    tk = lv["ticker"]
    now_et = now_et or datetime.now(ET)
    now_kst = now_et.astimezone(KST)
    st = load_state()
    row = {"run_kst": now_kst.strftime("%Y-%m-%d %H:%M"), "status": "", "last_bar": "", "close": np.nan, "bar_no": "",
           "V": np.nan, "E": np.nan, "pool": np.nan, "new_decisions": 0, "self_check": "", "alerts": ""}
    try:
        if not skip_download:
            download(cfg)
        px = load_prices(cfg)
        cutoff = datetime.combine(now_et.date(), datetime.min.time(), ET).replace(hour=16, minute=15)
        if px.index[-1].date() >= now_et.date() and now_et < cutoff:
            px = px.iloc[:-1]
    except Exception as e:
        row["status"] = f"데이터 실패({type(e).__name__})"
        row["alerts"] = send(cfg, f"[VR 기록 장치] {tk} 데이터를 받지 못했다 ({type(e).__name__}).", really_send)
        append("log.csv", [row])
        return row
    last = px.index[-1].date()
    row["last_bar"], row["close"] = str(last), float(px["close"].iloc[-1])
    if (now_kst.date() - last).days > int(lv["stale_days"]):
        row["status"] = "데이터 실패(오래됨)"
        row["alerts"] = send(cfg, f"[VR 기록 장치] {tk} 마지막 봉이 {last} 로 오래됐다. 계산을 건너뛴다.", really_send)
        append("log.csv", [row])
        return row
    if "paper_start" not in st:  # first run: fix the start
        today = now_et.date()
        opens = datetime.combine(today, datetime.min.time(), ET).replace(hour=9, minute=30)
        start = today if is_trading_day(today) and now_et < opens else next_trading_day(today)
        st = {"start_run_kst": row["run_kst"], "start_last_bar": str(last), "paper_start": str(start),
              "processed": "", "summary_month": ""}
    p = paper_px(px, st)
    if not len(p):
        row["status"] = "시작 대기"
        append("log.csv", [row])
        save_state(st)
        return row
    if "start_V" not in st:
        r0 = VR.run(cfg, p.iloc[:1], lv["bands"][0])
        st["start_V"], st["start_shares"] = r0["state"]["V"], r0["state"]["shares"]
    runs = {b: VR.run(cfg, p, b) for b in lv["bands"]}
    main_b = lv["bands"][0]
    rm = runs[main_b]
    row["bar_no"] = len(p) - 1
    row["V"], row["pool"] = rm["state"]["V"], rm["state"]["pool"]
    row["E"] = rm["state"]["shares"] * float(p["close"].iloc[-1])
    done = pd.Timestamp(st["processed"]) if st["processed"] else pd.Timestamp("1900-01-01")
    rows, texts = [], []
    for b, r in runs.items():
        o = r["orders"]
        if not len(o):
            continue
        for _, x in o[pd.to_datetime(o["decision"]) > done].iterrows():
            d = pd.Timestamp(x["decision"]).date()
            nxt = next_trading_day(d)
            late = now_et >= datetime.combine(nxt, datetime.min.time(), ET).replace(hour=9, minute=30)
            rows.append({"decision": x["decision"], "band": b, "bar": int(x["bar"]), "close": x["close"], "V": x["V"],
                         "E": x["E"], "pool": x["pool"], "side": x["side"], "amount": x["amount"],
                         "trade_day": str(nxt) if x["side"] else "", "late": late, "logged_kst": row["run_kst"]})
            if b == main_b and x["side"]:
                verb = "매수" if x["side"] == "buy" else "매도"
                texts.append(f"[VR 기록 장치] {tk} 평가금 ${x['E']:,.0f}, 목표 ${x['V']:,.0f}(밴드 ±{b:.0%} 밖) → "
                             f"{nxt} 시가에 ${x['amount']:,.0f} {verb}. 풀 ${x['pool']:,.0f}. 종이 계좌 10만 달러 기준, 주문은 내지 않는다."
                             + (" [늦은 기록]" if late else ""))
    append("decisions.csv", rows)
    row["new_decisions"] = len({r["decision"] for r in rows})
    last_dec = max((pd.Timestamp(x) for r in runs.values() if len(r["orders"]) for x in r["orders"]["decision"]), default=None)
    if last_dec is not None and last_dec > done:
        st["processed"] = str(last_dec.date())
    # self-check
    logged = read("decisions.csv")
    bad = 0
    for _, x in logged.iterrows():
        o = runs[float(x["band"])]["orders"]
        m = o[o["decision"] == x["decision"]]
        side = "" if pd.isna(x["side"]) else str(x["side"])
        if len(m) != 1 or str(m["side"].iloc[0]) != side or abs(float(m["amount"].iloc[0]) - float(x["amount"])) > 1e-6 * max(1.0, abs(float(x["amount"]))):
            bad += 1
    row["self_check"] = "일치" if bad == 0 else f"불일치 {bad}건"
    if bad:
        texts.append(f"[VR 기록 장치] 자가 점검 불일치 {bad}건. 야후가 과거 가격을 고쳤는지 본다.")
    month = now_kst.strftime("%Y-%m")
    if st.get("summary_month") != month and len(p) > 1:
        texts.append(summary(cfg, px, st))
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
    print(run(not a.no_send))


if __name__ == "__main__":
    main()
