"""Stage 0, the original's side: the original repo's own numbers next to ours.

The original (no license) lives outside this repo in original.ref_dir and is run there by hand:
    python ../../_ref/TIS-RSI2-Research/motor/rsi2_motor.py
which rewrites its own resultados/. The shipped (committed) outputs are read with `git show <commit>:<path>`, the
re-run ones from the working tree. --export imports the original's motor module (its module level only defines
constants and functions; read through before the first run) and writes its indicator columns (rsi2, sma, ma5) for the
RSI digit check. No original code is copied into this repo.

Usage:  python -m src.stage0 --export   (once, after the original was re-run; also does the comparison)
        python -m src.stage0
Writes  results/stage0/orig_indicators.csv (--export), results/stage0.csv, results/stage0_trades.csv,
        results/stage0_meta.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, load_config, results_dir
from .replicate import PARTS, load_csv, with_indicators

KEYS = {"n": "trades", "wr": "win_rate", "pf": "profit_factor", "pfnb": "pf_ex_best", "expectancy": "expectancy",
        "cagr": "cagr", "mdd_diario": "mdd", "exposicion": "exposure"}
COLS = list(KEYS.values())
PART_OF = {"IS": "IS", "OOS": "OOS", "TODO": "ALL"}


def ref_dir(cfg: dict) -> Path:
    return (ROOT / cfg["original"]["ref_dir"]).resolve()


def head_commit(cfg: dict) -> str:
    p = subprocess.run(["git", "-C", str(ref_dir(cfg)), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip()


def shipped_text(cfg: dict, rel: str) -> str:
    p = subprocess.run(["git", "-C", str(ref_dir(cfg)), "show", f"{cfg['original']['commit']}:{rel}"],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"git show {rel} failed: {p.stderr.strip()[:200]}")
    return p.stdout.replace("\r\n", "\n")


def rerun_text(cfg: dict, rel: str) -> str:
    return (ref_dir(cfg) / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


def fase1_rows(source: str, m: dict) -> list[dict]:
    return [{"source": source, "part": part, **{ours: m["fase1"][key][theirs] for theirs, ours in KEYS.items()}}
            for key, part in PART_OF.items()]


def same_fase1(a: dict, b: dict, tol: float = 1e-12) -> bool:
    return all(abs(a["fase1"][k][j] - b["fase1"][k][j]) <= tol for k in PART_OF for j in KEYS)


def export(cfg: dict) -> None:
    ref = ref_dir(cfg)
    spec = importlib.util.spec_from_file_location("orig_rsi2_motor", ref / "motor" / "rsi2_motor.py")
    mod = importlib.util.module_from_spec(spec)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    df = mod.preparar(mod.cargar(ref / cfg["original"]["csv"]))
    df[["rsi2", "sma", "ma5"]].to_csv(results_dir("stage0") / "orig_indicators.csv")
    print(f"exported {len(df)} rows of the original's indicators")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    rd = results_dir()
    if a.export:
        export(cfg)
    shipped = json.loads(shipped_text(cfg, "resultados/metricas.json"))
    rerun = json.loads(rerun_text(cfg, "resultados/metricas.json"))
    rp = pd.read_csv(rd / "replicate.csv")
    ours = rp[rp["variant"] == "original"]
    rows = [{"source": "reported", "part": p, **cfg["original"]["reported"][p]} for p in PARTS]
    rows += fase1_rows("shipped", shipped) + fase1_rows("rerun", rerun)
    rows += [{"source": "ours", "part": r["part"], **{k: r[k] for k in COLS}} for _, r in ours.iterrows()]
    s0 = pd.DataFrame(rows)
    s0.to_csv(rd / "stage0.csv", index=False, encoding="utf-8-sig")

    orig = pd.read_csv(io.StringIO(rerun_text(cfg, "resultados/trades.csv")), parse_dates=["entrada", "salida"])
    mine = pd.read_csv(rd / "replicate_trades.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    mine = mine[(mine["variant"] == "original") & (mine["part"] == "ALL")].reset_index(drop=True)
    n = max(len(orig), len(mine))
    o2, m2 = orig.reindex(range(n)), mine.reindex(range(n))
    st = pd.DataFrame({"orig_entry": o2["entrada"], "our_entry": m2["entry_date"], "orig_exit": o2["salida"],
                       "our_exit": m2["exit_date"], "orig_ret": o2["ret"], "our_ret": m2["ret"]})
    st["same_dates"] = (st["orig_entry"] == st["our_entry"]) & (st["orig_exit"] == st["our_exit"])
    st["ret_diff"] = (st["orig_ret"] - st["our_ret"]).abs()
    st.to_csv(rd / "stage0_trades.csv", index_label="i", encoding="utf-8-sig")

    oi = pd.read_csv(rd / "stage0" / "orig_indicators.csv", index_col=0, parse_dates=True)
    ind = with_indicators(cfg, load_csv(cfg)).reindex(oi.index)
    ours_rsi, theirs = ind["rsi"].to_numpy(np.float64), oi["rsi2"].to_numpy(np.float64)
    meta = {"commit": head_commit(cfg), "shipped_generated": shipped["meta"]["generado"],
            "rerun_generated": rerun["meta"]["generado"], "json_same": same_fase1(shipped, rerun),
            "trades_csv_same": shipped_text(cfg, "resultados/trades.csv") == rerun_text(cfg, "resultados/trades.csv"),
            "first": rerun["meta"]["desde"], "last": rerun["meta"]["hasta"], "bars": rerun["meta"]["velas"],
            "is_last": rerun["meta"]["is_hasta"], "oos_first": rerun["meta"]["oos_desde"],
            "rsi_days": len(oi), "rsi_mismatch_4dp": int((np.round(ours_rsi, 4) != np.round(theirs, 4)).sum()),
            "rsi_max_diff": float(np.nanmax(np.abs(ours_rsi - theirs))),
            "monkey_pct_pf": rerun["monkey"]["pct_pf"]}
    pd.DataFrame(list(meta.items()), columns=["key", "value"]).to_csv(rd / "stage0_meta.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(s0.round(4).to_string())
    print(f"trades: original {int(st['orig_entry'].notna().sum())}, ours {int(st['our_entry'].notna().sum())}, "
          f"same dates {int(st['same_dates'].sum())}, max ret diff {st['ret_diff'].max():.1e}")
    print({k: meta[k] for k in ("json_same", "trades_csv_same", "rsi_days", "rsi_mismatch_4dp", "rsi_max_diff")})


if __name__ == "__main__":
    main()
