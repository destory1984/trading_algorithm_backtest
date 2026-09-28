# Index Parking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put idle cash into the index (bnf: KOSPI / NDX price index; ibs: SPY at weight w) and compare with the current cash-idle results and with index buy-and-hold.

**Architecture:** One new module per algorithm folder (`bnf_backtest/src/park.py`, `ibs_backtest/src/park.py`). bnf's module is a copy of `krxbt.portfolio.run_portfolio` with an optional index sleeve; with the sleeve off it must reproduce the engine exactly. ibs's module replays the walk-forward trades with unit-based bookkeeping. The engine (`krx_backtest_core`, tag v0.2.1) is not touched. Every check result is written as columns of `park.csv` and summarised in `results/checks.md`.

**Tech Stack:** Python 3.13, pandas, numpy, krxbt v0.2.1 (editable install from the collection folder).

**Spec:** `docs/superpowers/specs/2026-09-28-index-parking-design.md`

## Global Constraints

- Engine code must stay equal to tag v0.2.1. No edits under `krx_backtest_core/`.
- Existing result files are not changed by this work (new files only: `park.csv`, `park_equity.csv`, and `checks.md` gains a section).
- bnf Korea parks in the KOSPI **price** index (`krxbt.data.load_index(cfg, "KOSPI")`); bnf US parks in the NDX **price** index (`krxbt.us.load_index(cfg, "NDX")`). Dividends are left out on both sides.
- Park costs one-way: Korea fee 0.015% + slippage 0.005%, sell tax 0; US = `us.costs` (fee 0.07% + slippage 0.05%). Cost factors come from `krxbt.costs.cost_factors` (buy ×(1+slip)(1+fee), sell ×(1−slip)(1−fee−tax)).
- bnf: sell the index at the index **open** of the entry day; buy it back at the index **close** of the exit day and with any other cash at every close.
- ibs: w ∈ {0.5, 0.7, 0.8, 0.9}, park ticker SPY, cash earns 0%, no rebalancing while idle, costs only on amounts actually traded.
- All parameters live in each folder's `config.yaml`.
- No pytest in this repo. Checks are functions whose results go to csv columns / `results/checks.md`.
- Windows: code using `ProcessPoolExecutor` (bnf `candidate_trades`) must run as `python -m src.park`, never via `python - <<EOF`. Prefix runs with `PYTHONIOENCODING=utf-8`.
- Repo docs: plain Korean, "~다" declarative, concrete numbers, no local disk paths, no folder-tree diagrams.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push.

## Review Focus

1. Index calendar gaps (a trading day where the index has no row, or no open) — expected: open falls back to the forward-filled close, never NaN in equity. Test: Task 1 Step 4 asserts `eq.notna().all()` and `park_px.notna().all().all()`.
2. Several entries on one open that together need more than the parked value — expected: `alloc` capped by cash + sellable index value, cash and units never negative. Test: Task 1 Step 4 checks `book.min()` on the crash95 run (many same-day entries on crash days).
3. Sleeve off must equal the engine to the last won (not "close") — expected: `np.array_equal`. Test: Task 1 Step 4 and the `engine_diff` column in Task 2.
4. ibs trades that exit at the close in `open` mode (max-hold exits) — expected: the park rebuy uses the same bar as the exit price. Test: Task 4 Step 4 asserts every trade's entry/exit price matched an open or close of its bar (`unmatched == 0`).
5. The US universe grows when new tickers are fetched into `us_data` (another session adds leveraged ETFs) — expected: the run records the ticker count it used. Test: Task 2 writes `n_tickers` into `results/us/park.csv`, and the README table states it.

---

### Task 1: bnf park engine (`park_portfolio`)

**Files:**
- Create: `bnf_backtest/src/park.py`
- Modify: `bnf_backtest/config.yaml` (add `park` section at the end)

**Interfaces:**
- Consumes: `krxbt.costs.cost_factors(cfg) -> (buy_cost, sell_keep)`; `src.portfolio.candidate_trades`, `compare_selection`, `combo_trades`, `_cal`, `_close_panel`, `run_portfolio`.
- Produces:
  - `park_frame(ix: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame` with columns `open`, `close` on `cal`, no NaN after the first valid row.
  - `park_portfolio(cal, tr, closes, capital, slots, rank_by, cooldown=0, crash_slots=0, crash_level=None, park=None, costs=(1.0, 1.0)) -> tuple[pd.Series, dict]`. `eq.attrs["ledger"]` as in the engine, `eq.attrs["book"]` = DataFrame(index=cal, columns=`cash`, `units`, `park_value`, `stock_value`). `info` = engine info keys + `park_buys`, `park_sells`.

- [ ] **Step 1: Add the config block**

Append to `bnf_backtest/config.yaml`:

```yaml

# 쉬는 돈을 지수에 넣어 두기 (python -m src.park [--us]). 설계: docs/superpowers/specs/2026-09-28-index-parking-design.md
park:
  kr_index: KOSPI            # 코스피 가격 지수(배당 빠짐)
  us_index: NDX              # 나스닥100 가격 지수(배당 빠짐)
  kr_costs:                  # 지수 ETF 매매 가정. 편도 수수료 0.015% + 슬리피지 0.005%, ETF 는 매도세 없음
    buy_fee: 0.00015
    sell_fee: 0.00015
    sell_tax: 0.0
    slippage: 0.00005
  # 미국은 us.costs 를 그대로 쓴다
```

- [ ] **Step 2: Write `park_frame` and `park_portfolio`**

Create `bnf_backtest/src/park.py`:

```python
"""Park idle cash in the index while the bnf portfolio waits for signals.

Same rules as krxbt.portfolio.run_portfolio (slots, 1/slots of the previous
close's equity, lowest disparity first, re-entry cooldown, crash slots).
Only the cash changes: with a park frame, cash left at a close buys the
index at that close, and an entry that needs more cash than is on hand
sells the index at the index open of the entry day. Index trades pay the
park costs. park=None runs the engine's bookkeeping unchanged.

Usage:  python -m src.park          # Korea, results/park.csv
        python -m src.park --us     # US, results/us/park.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors
from krxbt.portfolio import stats


def park_frame(ix: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Index open/close on the trading calendar. Missing closes are
    forward-filled; a missing open falls back to that filled close."""
    close = ix["close"].reindex(cal).ffill()
    opn = ix["open"].reindex(cal).where(lambda s: s > 0).fillna(close)
    return pd.DataFrame({"open": opn, "close": close})


def park_portfolio(cal: pd.DatetimeIndex, tr: pd.DataFrame, closes: pd.DataFrame, capital: float,
                   slots: int, rank_by: str, cooldown: int = 0, crash_slots: int = 0,
                   crash_level: float | None = None, park: pd.DataFrame | None = None,
                   costs: tuple[float, float] = (1.0, 1.0)) -> tuple[pd.Series, dict]:
    buy_cost, sell_keep = costs
    base = int(slots)
    wide = max(base, int(crash_slots or 0))
    day_no = {d: i for i, d in enumerate(cal)}
    blocked_until: dict[str, int] = {}
    ledger: list[dict] = []
    by_entry = {d: g.sort_values(rank_by) for d, g in tr.groupby("entry_date")} if len(tr) else {}
    cash = float(capital)
    units = 0.0
    park_buys = park_sells = 0
    pos: dict[str, dict] = {}
    equity = np.empty(len(cal))
    book = np.zeros((len(cal), 4))
    prev_eq = cash
    taken = 0
    taken_rets: list[float] = []
    invested_days = 0
    max_held = 0
    exposure: list[float] = []
    for i, d in enumerate(cal):
        if d in by_entry and len(pos) < wide:
            p_open = float(park.at[d, "open"]) if park is not None else 0.0
            for _, t in by_entry[d].iterrows():
                cap = wide if crash_level is not None and t["idx_disp"] <= crash_level else base
                if len(pos) >= cap:
                    continue
                tk = str(t["ticker"])
                if tk in pos:
                    continue
                if i <= blocked_until.get(tk, -1):
                    continue
                avail = cash + units * p_open * sell_keep
                alloc = min(prev_eq / cap, avail)
                if alloc <= 0:
                    break
                if alloc > cash:  # sell just enough index at today's index open
                    need = alloc - cash
                    sold = min(units, need / (p_open * sell_keep))
                    units -= sold
                    cash += sold * p_open * sell_keep
                    park_sells += 1
                    alloc = min(alloc, cash)
                cash -= alloc
                ledger.append({"ticker": tk, "entry_date": d, "exit_date": t["exit_date"], "alloc": alloc,
                               "ret": float(t["ret"]), "reason": t["reason"]})
                pos[tk] = {"alloc": alloc, "entry_px": float(t["entry_px"]),
                           "exit_date": t["exit_date"], "ret": float(t["ret"]), "reason": t["reason"]}
                taken += 1
                taken_rets.append(float(t["ret"]))
        max_held = max(max_held, len(pos))
        for tk in [k for k, p in pos.items() if p["exit_date"] <= d]:
            p = pos.pop(tk)
            cash += p["alloc"] * (1 + p["ret"])
            if cooldown and p["reason"] == "stop":
                blocked_until[tk] = day_no.get(p["exit_date"], i) + cooldown
        p_close = float(park.at[d, "close"]) if park is not None else 0.0
        if park is not None and cash > 0:  # idle cash buys the index at today's close
            units += cash / (p_close * buy_cost)
            cash = 0.0
            park_buys += 1
        row = closes.loc[d]
        val = 0.0
        for tk, p in pos.items():
            c = row.get(tk)
            val += p["alloc"] * (c / p["entry_px"] if pd.notna(c) else 1.0)
        pv = units * p_close
        prev_eq = cash + pv + val
        equity[i] = prev_eq
        book[i] = (cash, units, pv, val)
        invested_days += bool(pos)
        if pos:
            exposure.append(val / prev_eq)
    eq = pd.Series(equity, index=cal, name="equity")
    eq.attrs["ledger"] = pd.DataFrame(ledger)
    eq.attrs["book"] = pd.DataFrame(book, index=cal, columns=["cash", "units", "park_value", "stock_value"])
    return eq, {"trades_taken": taken, "trades_available": len(tr),
                "taken_mean_ret": float(np.mean(taken_rets)) if taken_rets else np.nan,
                "all_mean_ret": float(tr["ret"].mean()) if len(tr) else np.nan,
                "invested_share": invested_days / len(cal), "max_held": max_held,
                "exposure": float(np.mean(exposure)) if exposure else 0.0,
                "park_buys": park_buys, "park_sells": park_sells}


def park_costs(cost_cfg: dict) -> tuple[float, float]:
    return cost_factors({"costs": cost_cfg})
```

- [ ] **Step 3: Write a throwaway check script in the session scratchpad (not in the repo)**

Save as `<scratchpad>/park_task1.py` and run it from `bnf_backtest/` with `PYTHONIOENCODING=utf-8 python <scratchpad>/park_task1.py`. `candidate_trades` uses processes, so this must be a file, not a heredoc.

```python
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from krxbt import portfolio as kp
from krxbt.data import load_index
from src.common import load_config
from src.portfolio import _cal, _close_panel, candidate_trades, combo_trades, compare_selection
from src.park import park_costs, park_frame, park_portfolio

if __name__ == "__main__":
    cfg = load_config()
    st = cfg["stage2"]
    sel = compare_selection(cfg)
    cid = int(sel.index[sel["market_filter"] == "crash95"][0])
    cand = candidate_trades(cfg, [cid])
    tr = combo_trades(cand, cid, st["compare_combo"]["market"])
    cal = _cal(cfg)
    closes = _close_panel(cfg, tr["ticker"].astype(str).unique(), cal)
    cap, n = st["initial_capital"], st["max_positions"]
    eng, _ = kp.run_portfolio(cal, tr, closes, cap, n, rank_by="disp")
    off, _ = park_portfolio(cal, tr, closes, cap, n, rank_by="disp")
    print("engine equal:", np.array_equal(eng.to_numpy(), off.to_numpy()))
    pf = park_frame(load_index(cfg, cfg["park"]["kr_index"]), cal)
    on, info = park_portfolio(cal, tr, closes, cap, n, rank_by="disp", park=pf, costs=park_costs(cfg["park"]["kr_costs"]))
    b = on.attrs["book"]
    print("nan:", on.isna().sum(), pf.isna().sum().sum())
    print("ledger diff:", float((b[["cash", "park_value", "stock_value"]].sum(axis=1) - on).abs().max()))
    print("min cash/units:", float(b["cash"].min()), float(b["units"].min()))
    print("cash cagr/mdd:", kp.stats(off)["cagr"], kp.stats(off)["mdd"])
    print("park cagr/mdd:", kp.stats(on)["cagr"], kp.stats(on)["mdd"], info["park_buys"], info["park_sells"])
```

- [ ] **Step 4: Run it and check the output**

Run: `cd bnf_backtest && PYTHONIOENCODING=utf-8 python <scratchpad>/park_task1.py`
Expected:
- `engine equal: True`
- `nan: 0 0`
- `ledger diff:` ≤ 1e-6
- `min cash/units:` both ≥ 0 (cash may be 0.0; a tiny negative like -1e-9 from float rounding is acceptable, anything below -1 is a bug)
- `cash cagr/mdd:` ≈ 0.099 / -0.451 (the crash95, cooldown 0 row of `results/portfolio_filters.csv`)
If `engine equal` is False, diff the two loops line by line against `krx_backtest_core/krxbt/portfolio.py:31-98`; the park=None path must execute the same arithmetic in the same order.

- [ ] **Step 5: Commit**

```bash
git add bnf_backtest/src/park.py bnf_backtest/config.yaml
git commit -m "bnf: park_portfolio, the engine portfolio with an optional index sleeve

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: bnf runner (Korea and US) with built-in checks

**Files:**
- Modify: `bnf_backtest/src/park.py` (add runner functions and `main`)

**Interfaces:**
- Consumes: Task 1 functions; `src.portfolio.candidate_trades`, `compare_selection`, `combo_trades`, `_cal`, `_close_panel`, `filter_label`; `src.walkforward.us_candidates(cfg, pairs) -> (trades_df, closes_df)`; `src.us.us_config()`, `src.us.frames`; `krxbt.us.load_calendar`, `krxbt.us.load_index`; `krxbt.data.load_index`.
- Produces: `results/park.csv`, `results/park_equity.csv`, `results/us/park.csv`, `results/us/park_equity.csv`. Columns of park.csv: `market, filter, cooldown, parking, cagr, mdd, cagr_train, cagr_test, trades_taken, invested_share, park_buys, park_sells, engine_diff, ledger_diff, min_cash, min_units, old_cagr_diff` (+ `kind, n_tickers` for US). One final row per index buy-and-hold, and one row `parking="신호 없음 점검"` holding `nosignal_diff`.

- [ ] **Step 1: Add the runner code**

Append to `bnf_backtest/src/park.py` (below `park_costs`):

```python
def _row(eq: pd.Series, te: pd.Timestamp) -> dict:
    return {**stats(eq), "cagr_train": stats(eq[eq.index <= te])["cagr"],
            "cagr_test": stats(eq[eq.index > te])["cagr"]}


def _checks(eq: pd.Series, eng: pd.Series | None) -> dict:
    b = eq.attrs["book"]
    out = {"ledger_diff": float((b[["cash", "park_value", "stock_value"]].sum(axis=1) - eq).abs().max()),
           "min_cash": float(b["cash"].min()), "min_units": float(b["units"].min())}
    if eng is not None:
        out["engine_diff"] = float((eq - eng).abs().max())
    return out


def _pair(cal, tr, closes, cap, slots, cooldown, pf, costs, te) -> tuple[list[dict], dict]:
    """Cash-idle (checked against the engine) and index-parked runs of one trade list."""
    from krxbt import portfolio as kp
    eng, _ = kp.run_portfolio(cal, tr, closes, cap, slots, rank_by="disp", cooldown=cooldown)
    off, info0 = park_portfolio(cal, tr, closes, cap, slots, "disp", cooldown=cooldown)
    on, info1 = park_portfolio(cal, tr, closes, cap, slots, "disp", cooldown=cooldown, park=pf, costs=costs)
    rows = [{"parking": "현금", **_row(off, te), **info0, **_checks(off, eng)},
            {"parking": "지수", **_row(on, te), **info1, **_checks(on, None)}]
    return rows, {"현금": off, "지수": on}


def _nosignal(cal, closes, cap, slots, pf, costs) -> float:
    """No trades: the parked curve must be index buy-and-hold less one buy cost."""
    empty = pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "entry_px", "ret", "reason",
                                  "idx_disp", "disp"])
    eq, _ = park_portfolio(cal, empty, closes, cap, slots, "disp", park=pf, costs=costs)
    want = cap / (pf["close"].iloc[0] * costs[0]) * pf["close"]
    return float((eq - want).abs().max())


def run_kr() -> None:
    from krxbt.data import load_index
    from .common import load_config, results_dir
    from .portfolio import _cal, _close_panel, candidate_trades, combo_trades, compare_selection, filter_label
    cfg = load_config()
    st, pk = cfg["stage2"], cfg["park"]
    te = pd.Timestamp(st["train_end"])
    cap, slots, mk = st["initial_capital"], int(st["max_positions"]), st["compare_combo"]["market"]
    sel = compare_selection(cfg)
    cand = candidate_trades(cfg, list(sel.index))
    cal = _cal(cfg)
    closes = _close_panel(cfg, cand["ticker"].astype(str).unique(), cal)
    pf = park_frame(load_index(cfg, pk["kr_index"]), cal)
    costs = park_costs(pk["kr_costs"])
    rd = results_dir()
    old = pd.read_csv(rd / "portfolio_filters.csv")
    old = old[old["method"] == "portfolio"].set_index(["filter", "cooldown"])["cagr"]
    rows, curves = [], {}
    for cid, c in sel.iterrows():
        tr = combo_trades(cand, int(cid), mk)
        for cd in st.get("reentry_cooldowns", [0]):
            pair, eqs = _pair(cal, tr, closes, cap, slots, int(cd), pf, costs, te)
            for r in pair:
                r.update({"market": mk, "filter": filter_label(c["market_filter"]), "cooldown": int(cd)})
                key = (c["market_filter"], int(cd))
                if r["parking"] == "현금" and key in old.index:
                    r["old_cagr_diff"] = abs(r["cagr"] - float(old.loc[key]))
                rows.append(r)
                curves[f"{c['market_filter']}_cd{cd}_{r['parking']}"] = eqs[r["parking"]]
            print(f"{c['market_filter']} cd {cd}: cash {pair[0]['cagr']:+.3f}/{pair[0]['mdd']:.3f} "
                  f"index {pair[1]['cagr']:+.3f}/{pair[1]['mdd']:.3f}")
    bh = cap * pf["close"] / pf["close"].iloc[0]
    rows.append({"market": mk, "filter": f"{pk['kr_index']} 보유", "parking": "-", **_row(bh, te)})
    rows.append({"market": mk, "filter": "-", "parking": "신호 없음 점검",
                 "nosignal_diff": _nosignal(cal, closes, cap, slots, pf, costs)})
    curves[f"{pk['kr_index']} 보유"] = bh
    pd.DataFrame(rows).to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")


def run_us() -> None:
    from krxbt import us
    from .common import results_dir
    from .portfolio import filter_label
    from .us import us_config
    from .walkforward import us_candidates
    cfg = us_config()
    u, pk = cfg["us"], cfg["park"]
    te = pd.Timestamp(cfg["stage2"]["train_end"])
    cap, slots = u["initial_capital"], int(cfg["stage2"]["max_positions"])
    rd = results_dir() / "us"
    s = pd.read_csv(rd / "grid_summary.csv")
    top = s[(s["kind"] == "stock") & (s["trades"] >= u["min_trades"])].head(u["portfolio_top_n"])
    pick = pd.concat([top, top.assign(kind="ALL")], ignore_index=True)
    cand, closes = us_candidates(cfg, [(int(c), "") for c in pick["combo"]])
    cal = us.load_calendar(cfg)
    cal = cal[cal >= pd.Timestamp(cfg["data"]["start"])]
    closes = closes.reindex(cal).ffill()
    n_tickers = int(closes.shape[1])
    pf = park_frame(us.load_index(cfg, pk["us_index"]), cal)
    costs = park_costs(u["costs"])
    rows, curves = [], {}
    for _, c in pick.iterrows():
        tr = cand[cand["combo"] == int(c["combo"])]
        if c["kind"] != "ALL":
            tr = tr[tr["kind"] == c["kind"]]
        pair, eqs = _pair(cal, tr, closes, cap, slots, 0, pf, costs, te)
        kind = "주식" if c["kind"] == "stock" else "주식+ETF"
        for r in pair:
            r.update({"market": "US", "kind": kind, "filter": filter_label(c["market_filter"]), "cooldown": 0,
                      "n_tickers": n_tickers})
            rows.append(r)
            curves[f"{kind}_{c['market_filter']}_{r['parking']}"] = eqs[r["parking"]]
        print(f"{kind} {c['market_filter']}: cash {pair[0]['cagr']:+.3f}/{pair[0]['mdd']:.3f} "
              f"index {pair[1]['cagr']:+.3f}/{pair[1]['mdd']:.3f}")
    bh = cap * pf["close"] / pf["close"].iloc[0]
    rows.append({"market": "US", "filter": f"{pk['us_index']} 보유", "parking": "-", **_row(bh, te)})
    rows.append({"market": "US", "filter": "-", "parking": "신호 없음 점검",
                 "nosignal_diff": _nosignal(cal, closes, cap, slots, pf, costs)})
    curves[f"{pk['us_index']} 보유"] = bh
    pd.DataFrame(rows).to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true")
    run_us() if ap.parse_args().us else run_kr()


if __name__ == "__main__":
    main()
```

Note: `us_candidates` builds trades for every `(combo, _)` pair and returns closes for all US tickers; its combo ids are the grid combo ids stored in `grid_summary.csv` column `combo`.

- [ ] **Step 2: Run Korea**

Run: `cd bnf_backtest && PYTHONIOENCODING=utf-8 python -m src.park`
Expected: 20 printed lines (5 filters × 4 cooldowns), then `results/park.csv` exists. Then:

```bash
PYTHONIOENCODING=utf-8 python -c "import pandas as pd; p=pd.read_csv('results/park.csv'); c=p[p.parking=='현금']; print(len(c), c.engine_diff.max(), c.old_cagr_diff.max(), p.ledger_diff.max(), p.min_cash.min(), p.min_units.min(), p.nosignal_diff.max())"
```

Expected: `20 0.0 <1e-12 <1e-4 >=-1e-6 >=0 <1e-4`. `old_cagr_diff` ≈ 0 proves the cash-idle side reproduces `portfolio_filters.csv`. If it is not ≈ 0 but `engine_diff` is 0, the Korean data changed since that file was made; stop and report it instead of continuing.

- [ ] **Step 3: Run US**

Run: `cd bnf_backtest && PYTHONIOENCODING=utf-8 python -m src.park --us`
Expected: 6 printed lines, `results/us/park.csv` exists. Then:

```bash
PYTHONIOENCODING=utf-8 python -c "import pandas as pd; p=pd.read_csv('results/us/park.csv'); c=p[p.parking=='현금']; print(len(c), c.engine_diff.max(), p.ledger_diff.max(), p.min_cash.min(), p.min_units.min(), p.nosignal_diff.max(), p.n_tickers.dropna().unique())"
```

Expected: `6 0.0 <1e-6 >=-1e-6 >=0 <1e-6 [N]`. Record N. The US cash-idle numbers will not match the older `results/us/portfolio.csv` exactly because `us_data` gained tickers and days after that file was made; this is expected and goes into NOTES.

- [ ] **Step 4: Commit**

```bash
git add bnf_backtest/src/park.py
git commit -m "bnf: park runner for Korea and US with built-in checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: bnf checks section, README, NOTES

**Files:**
- Modify: `bnf_backtest/src/checks.py` (new section 6 before the file is written)
- Modify: `bnf_backtest/README.md` (new result section and one line in 「돌리는 법」)
- Modify: `bnf_backtest/NOTES.md`

**Interfaces:**
- Consumes: `results/park.csv`, `results/us/park.csv` from Task 2.
- Produces: section `## 6. 지수 대기 계산` in `results/checks.md`.

- [ ] **Step 1: Add the checks section**

In `bnf_backtest/src/checks.py`, add this function above `main()`:

```python
def park_section(rd) -> list[str]:
    out = ["## 6. 지수 대기 계산 (python -m src.park [--us] 결과를 읽는다)"]
    for name, p in (("한국", rd / "park.csv"), ("미국", rd / "us" / "park.csv")):
        if not p.exists():
            out.append(f"- {name}: {p.name} 없음. python -m src.park 를 먼저 돌린다")
            continue
        x = pd.read_csv(p)
        c = x[x["parking"] == "현금"]
        ok = (c["engine_diff"].max() == 0 and x["ledger_diff"].max() < 1e-4
              and x["min_cash"].min() > -1e-6 and x["min_units"].min() >= 0 and x["nosignal_diff"].max() < 1e-4)
        out.append(f"- {name}: 대기 끔 = 엔진 {len(c)}줄 최대 차이 {c['engine_diff'].max():.0f}, "
                   f"장부 차이 최대 {x['ledger_diff'].max():.2e}, 현금 최소 {x['min_cash'].min():.2e}, "
                   f"지수 단위 최소 {x['min_units'].min():.2e}, 신호 없음 곡선 차이 {x['nosignal_diff'].max():.2e}"
                   + (" (정상)" if ok else " (오류)"))
    return out + [""]
```

and in `main()` insert, right before `(rd / "checks.md").write_text(...)`:

```python
    lines += park_section(rd)
```

- [ ] **Step 2: Run checks**

Run: `cd bnf_backtest && PYTHONIOENCODING=utf-8 python -m src.checks`
Expected: the output ends with section 6, two lines, both ending `(정상)`. Sections 1–5 are byte-identical to before: compare with the copy saved earlier in the session scratchpad (`<scratchpad>/bnf/checks.md`) — `diff` shows only the added section.

- [ ] **Step 3: README and NOTES**

In `bnf_backtest/README.md`:
- Add a section `## 쉬는 돈을 지수에 넣어 두기` after the portfolio results. It holds two tables built from `results/park.csv` and `results/us/park.csv`: one row per filter (cooldown 0 only, to keep it short; the csv has the rest), columns `필터 | 현금 대기 연환산 / MDD | 지수 대기 연환산 / MDD | 지수 사고판 횟수`, final row index buy-and-hold. Under the tables, in plain sentences with numbers: that both sides use the price index so dividends (about 1.5–2% a year) are missing on both; that the US run used N tickers; one sentence on whether index parking beats index buy-and-hold, with the numbers.
- In 「돌리는 법」 add `python -m src.park          # 쉬는 돈을 지수에 (--us 는 미국)`.

In `bnf_backtest/NOTES.md`:
- Under 「판단과 이유」 add a bullet **지수 대기** with: why the index is sold at the open and bought back at the close; that a stop exit at the open leaves cash idle for that one day; the park cost values; that the engine was not changed (tag v0.2.1).
- Under 「검증한 방법」 add the five checks from section 6 and the `old_cagr_diff` reproduction of `portfolio_filters.csv`.
- Strike through item 5 of 「다음 할 일 후보」 and write the date and the one-line result.

- [ ] **Step 4: Commit**

```bash
git add bnf_backtest/src/checks.py bnf_backtest/README.md bnf_backtest/NOTES.md
git commit -m "bnf: index parking results, checks and notes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: ibs park (`park_equity`) and runner

**Files:**
- Create: `ibs_backtest/src/park.py`
- Modify: `ibs_backtest/config.yaml` (add `park` section)

**Interfaces:**
- Consumes: `src.common.load_config, results_dir, stats`; `src.strategy.frames(cfg, tickers) -> dict[str, DataFrame]` (adjusted `open`/`close`); `results/walkforward_trades.parquet` (columns `variant, ticker, entry_date, exit_date, entry_px, exit_px, ret`); `results/walkforward_equity.csv` (columns `wf_all, wf_close, wf_open`); `krxbt.costs.cost_factors`.
- Produces:
  - `bar_time(f: pd.DataFrame, d: pd.Timestamp, px: float) -> str` returning `"open"` or `"close"`, raising `ValueError` if `px` matches neither.
  - `park_equity(cal, trades, fr, cap, w, park_ticker, costs) -> pd.Series` with `attrs["book"]` = DataFrame(`cash, park_value, trade_value`) and `attrs["weights"]` = DataFrame(`park_w, trade_w`) at each close.
  - `static_mix(cal, fr, cap, w, park_ticker, costs) -> pd.Series`.
  - `results/park.csv`, `results/park_equity.csv`.

- [ ] **Step 1: Config**

Append to `ibs_backtest/config.yaml`:

```yaml

# 쉬는 날 SPY 를 일부 들고, 신호 날 전액 (python -m src.park). 설계: docs/superpowers/specs/2026-09-28-index-parking-design.md
park:
  ticker: SPY
  weights: [0.5, 0.7, 0.8, 0.9]   # 쉬는 날 평가금액 중 SPY 비중. 네 값 모두 보이고 하나를 골라내지 않는다
```

- [ ] **Step 2: Write the module**

Create `ibs_backtest/src/park.py`:

```python
"""Hold SPY at weight w while idle, go all in on an IBS signal.

Trades are the walk-forward account's trades (results/walkforward_trades.parquet),
so entries and exits do not change. Bookkeeping is in units:
  idle         w of equity in SPY, the rest cash at 0%, no rebalancing
  entry        signal on SPY: the cash buys more SPY (the parked SPY is kept);
               other ETF: all SPY is sold, everything buys the ETF
  exit         back to w of equity in SPY (on SPY: only the excess is sold)
Every trade happens at the bar of the trade's own price (open or close),
found by matching entry_px / exit_px to the adjusted open or close.
Costs (costs section) apply only to amounts actually bought or sold.

Usage:  python -m src.park     (after src.walkforward)
Writes  results/park.csv, results/park_equity.csv
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from krxbt.costs import cost_factors

from .common import load_config, results_dir, stats
from .strategy import frames


def bar_time(f: pd.DataFrame, d: pd.Timestamp, px: float) -> str:
    for k in ("open", "close"):
        if math.isclose(float(f.at[d, k]), px, rel_tol=1e-9):
            return k
    raise ValueError(f"{d.date()} price {px} is neither open nor close")


def park_equity(cal: pd.DatetimeIndex, trades: pd.DataFrame, fr: dict[str, pd.DataFrame], cap: float,
                w: float, park_ticker: str, costs: tuple[float, float]) -> pd.Series:
    bc, sk = costs
    sp = fr[park_ticker]
    ent = {d: r for d, r in zip(trades["entry_date"], trades.itertuples())}
    ext = {d: r for d, r in zip(trades["exit_date"], trades.itertuples())}
    cash, spy, units, tk = float(cap), 0.0, 0.0, None  # spy = parked SPY units; units/tk = open trade
    started = False
    eq = np.empty(len(cal))
    book = np.zeros((len(cal), 3))
    wts = np.zeros((len(cal), 2))
    for i, d in enumerate(cal):
        if d in ent and tk is None:
            r = ent[d]
            f = fr[r.ticker]
            px = float(r.entry_px)
            bar_time(f, d, px)
            if r.ticker == park_ticker:
                units, spy = spy + cash / (px * bc), 0.0
            else:
                cash += spy * float(sp.at[d, bar_time(f, d, px)]) * sk
                spy = 0.0
                units = cash / (px * bc)
            cash, tk = 0.0, r.ticker
            started = True
        if d in ext and tk is not None and ext[d].ticker == tk:
            r = ext[d]
            f = fr[tk]
            px = float(r.exit_px)
            t = bar_time(f, d, px)
            if tk == park_ticker:
                target = w * (units * px + cash) / px  # SPY units to keep
                cash += (units - target) * px * sk
                spy = target
            else:
                cash += units * px * sk
                spx = float(sp.at[d, t])
                spy = w * cash / (spx * bc)
                cash -= w * cash
            units, tk = 0.0, None
        if not started and tk is None:  # first idle close: park w of the capital
            spx = float(sp.at[d, "close"])
            spy = w * cash / (spx * bc)
            cash -= w * cash
            started = True
        pv = spy * float(sp.at[d, "close"])
        tv = units * float(fr[tk].at[d, "close"]) if tk is not None else 0.0
        eq[i] = cash + pv + tv
        book[i] = (cash, pv, tv)
        wts[i] = (pv / eq[i], tv / eq[i])
    s = pd.Series(eq, cal)
    s.attrs["book"] = pd.DataFrame(book, index=cal, columns=["cash", "park_value", "trade_value"])
    s.attrs["weights"] = pd.DataFrame(wts, index=cal, columns=["park_w", "trade_w"])
    return s


def static_mix(cal: pd.DatetimeIndex, fr: dict[str, pd.DataFrame], cap: float, w: float, park_ticker: str,
               costs: tuple[float, float]) -> pd.Series:
    c = fr[park_ticker]["close"].reindex(cal)
    units = w * cap / (float(c.iloc[0]) * costs[0])
    return cap * (1 - w) + units * c


def _stats(eq: pd.Series, cap: float) -> dict:
    prev = eq.shift(1)
    prev.iloc[0] = cap
    st = stats(eq / prev - 1)
    st.pop("exposure")
    return st


def main() -> None:
    cfg = load_config()
    pk = cfg["park"]
    cap = cfg["portfolio"]["initial_capital"]
    rd = results_dir()
    costs = cost_factors(cfg)
    fr = frames(cfg, cfg["grid"]["tickers"])
    wfe = pd.read_csv(rd / "walkforward_equity.csv", index_col=0, parse_dates=True)
    cal = pd.DatetimeIndex(wfe.index)
    fr = {t: f.reindex(cal).ffill() for t, f in fr.items()}
    tr = pd.read_parquet(rd / "walkforward_trades.parquet")
    rows, curves = [], {}
    for var, t in tr.groupby("variant", sort=False):
        t = t.sort_values("entry_date", ignore_index=True)
        base = park_equity(cal, t, fr, cap, 0.0, pk["ticker"], costs)
        idle = base.attrs["weights"]["trade_w"] == 0
        w0_diff = float((base - wfe[f"wf_{var}"])[idle].abs().max())
        spy_only = t[t["ticker"] == pk["ticker"]]
        full = park_equity(cal, spy_only, fr, cap, 1.0, pk["ticker"], costs)
        bh = cap / costs[0] * fr[pk["ticker"]]["close"] / float(fr[pk["ticker"]]["close"].iloc[0])
        w1_diff = float((full - bh).abs().max())
        for w in [0.0] + list(pk["weights"]):
            eq = base if w == 0 else park_equity(cal, t, fr, cap, w, pk["ticker"], costs)
            b, wt = eq.attrs["book"], eq.attrs["weights"]
            rows.append({"variant": var, "name": "IBS + SPY 대기" if w else "IBS 만 (현금 대기)", "w": w,
                         **_stats(eq, cap), "trades": len(t),
                         "avg_spy_w": float(wt["park_w"].mean()), "avg_etf_w": float(wt["trade_w"].mean()),
                         "ledger_diff": float((b.sum(axis=1) - eq).abs().max()),
                         "min_cash": float(b["cash"].min()),
                         "w0_diff": w0_diff if w == 0 else np.nan, "w1_diff": w1_diff if w == 0 else np.nan,
                         "spy_only_trades": len(spy_only) if w == 0 else np.nan})
            curves[f"{var}_w{w:.1f}"] = eq
    for w in pk["weights"]:
        eq = static_mix(cal, fr, cap, w, pk["ticker"], costs)
        rows.append({"variant": "-", "name": f"{pk['ticker']} {w:.0%} + 현금 고정", "w": w, **_stats(eq, cap),
                     "avg_spy_w": float((eq - cap * (1 - w)).div(eq).mean())})
        curves[f"static_w{w:.1f}"] = eq
    bh = cap * fr[pk["ticker"]]["close"] / float(fr[pk["ticker"]]["close"].iloc[0])
    rows.append({"variant": "-", "name": f"{pk['ticker']} 보유", "w": 1.0, **_stats(bh, cap), "avg_spy_w": 1.0})
    curves[f"{pk['ticker']} 보유"] = bh
    out = pd.DataFrame(rows)
    out.to_csv(rd / "park.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(rd / "park_equity.csv")
    pd.set_option("display.width", 250)
    print(out[["variant", "name", "w", "cagr", "mdd", "sharpe", "avg_spy_w", "avg_etf_w"]].round(4).to_string())


if __name__ == "__main__":
    main()
```

Notes for the implementer:
- `frames` returns adjusted prices (engine `ticker_frame`), the same prices the trades were priced with, so `bar_time` must find every `entry_px`/`exit_px`. A `ValueError` means the frames and the parquet disagree — re-run `python -m src.walkforward` rather than loosening the tolerance.
- The `static_mix` `avg_spy_w` expression is the SPY share of equity at each close.

- [ ] **Step 3: Run**

Run: `cd ibs_backtest && PYTHONIOENCODING=utf-8 python -m src.park`
Expected: a printed table of 3 variants × 5 rows (w = 0, 0.5, 0.7, 0.8, 0.9) + 4 static rows + 1 SPY row = 20 rows. No exception.

- [ ] **Step 4: Check the built-in columns**

```bash
cd ibs_backtest && PYTHONIOENCODING=utf-8 python -c "import pandas as pd; p=pd.read_csv('results/park.csv'); print(p.w0_diff.max(), p.w1_diff.max(), p.ledger_diff.max(), p.min_cash.min(), p.spy_only_trades.dropna().tolist())"
```

Expected: `w0_diff` < 0.01 (cents: w = 0 equals the walk-forward curve on idle days), `w1_diff` < 0.01, `ledger_diff` < 1e-6, `min_cash` ≥ -1e-6, and `spy_only_trades` shows how many trades the w = 1 check used (0 is allowed for the `all`/`close` variants, which picked only QQQ; the `open` variant has SPY trades). The fact that `bar_time` raised nothing is the check of Review Focus 4. Also confirm the w = 0 rows' `cagr` equals the `걸어가며` rows of `results/walkforward_summary.csv` to 1e-9.

- [ ] **Step 5: Commit**

```bash
git add ibs_backtest/src/park.py ibs_backtest/config.yaml
git commit -m "ibs: hold SPY at weight w while idle, all in on the signal

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: ibs checks section, README, NOTES

**Files:**
- Modify: `ibs_backtest/src/checks.py`
- Modify: `ibs_backtest/README.md`, `ibs_backtest/NOTES.md`

**Interfaces:**
- Consumes: `results/park.csv` from Task 4.
- Produces: section `## 지수 대기 계산` at the end of `results/checks.md`.

- [ ] **Step 1: Checks section**

Add above `main()` in `ibs_backtest/src/checks.py`:

```python
def park_section() -> list[str]:
    p = results_dir() / "park.csv"
    out = ["## 지수 대기 계산", ""]
    if not p.exists():
        return out + ["- park.csv 없음. python -m src.park 를 먼저 돌린다", ""]
    x = pd.read_csv(p)
    ok = (x["w0_diff"].max() < 0.01 and x["w1_diff"].max() < 0.01 and x["ledger_diff"].max() < 1e-6
          and x["min_cash"].min() > -1e-6)
    return out + [f"- w = 0 곡선과 걸어가며 검증 곡선(포지션 없는 날) 최대 차이 {x['w0_diff'].max():.4f} 달러",
                  f"- SPY 거래만, w = 1 곡선과 SPY 보유 × (1 - 매수 비용) 최대 차이 {x['w1_diff'].max():.4f} 달러",
                  f"- 장부(현금 + SPY + 거래 종목 = 평가금액) 최대 차이 {x['ledger_diff'].max():.2e}, "
                  f"현금 최소 {x['min_cash'].min():.2e}" + (" (정상)" if ok else " (오류)"), ""]
```

and in `main()` add `out += park_section()` after `out += flat_days(cfg, fr)`.

- [ ] **Step 2: Run checks**

Run: `cd ibs_backtest && PYTHONIOENCODING=utf-8 python -m src.checks`
Expected: output ends with the new section ending `(정상)`; everything above it byte-identical to `<scratchpad>/ibs/checks.md` saved earlier in the session (`diff` shows only the added lines).

- [ ] **Step 3: README and NOTES**

`ibs_backtest/README.md`: add `## 쉬는 날 SPY 들고 있기` after the results table. One table from `results/park.csv`, variants `close` and `open` only (`all` equals `close` in the current run; say so in one line): rows w = 0, 0.5, 0.7, 0.8, 0.9, then the four static mixes and SPY 보유; columns `연 수익률 | 최대 낙폭 | 샤프 | 평균 SPY 비중 | 평균 ETF 비중`. Below it, plain sentences with numbers comparing each w row to the static mix with the same w. Add `python -m src.park        # 쉬는 날 SPY 들고 있기` to 「돌리는 법」 and `src/park.py` to the file table.

`ibs_backtest/NOTES.md`: in 「판단과 이유」 add **SPY 대기** (why weights instead of full SPY parking; why unit bookkeeping; why the w part of an SPY trade pays no cost; why static mixes are the fair comparison). In 「검증한 방법」 add the three checks. In 「다음 할 일 후보」 strike item 2 with the date and one-line result.

- [ ] **Step 4: Commit**

```bash
git add ibs_backtest/src/checks.py ibs_backtest/README.md ibs_backtest/NOTES.md
git commit -m "ibs: SPY parking results, checks and notes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
