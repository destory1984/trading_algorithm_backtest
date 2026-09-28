"""Stage 0: run the original package (outside this repo) and put its numbers next to the README's.

The original repo has no license: nothing is copied. Its CLI and its public functions are called
in ../../_ref/IBS-Strategy with `uv run`.

Usage:  python -m src.stage0 --run     run the five CLI commands and the TQQQ trade export
        python -m src.stage0           parse results/stage0/*.txt only
Writes  results/stage0/<name>.txt, results/stage0/orig_tqqq_trades.csv,
        results/stage0/orig_tqqq_daily.csv, results/stage0.csv
"""
from __future__ import annotations

import argparse
import re
import subprocess

import pandas as pd

from .common import ROOT, load_config, results_dir

PAT = {
    "sharpe": r"Sharpe ratio\s+(-?[\d.]+)",
    "cagr": r"CAGR\s+(-?[\d.]+)%",
    "mdd": r"Max drawdown\s+(-?[\d.]+)%",
    "trades": r"\((\d+) closed trades\)",
    "exposure": r"Time in market\s+(-?[\d.]+)%",
}
BARS = r": (\d+) daily bars, (\S+) to (\S+)"

# our call into the original's public API (not a copy of its code): trades and daily bars of the
# default TQQQ run (real listing, full sizing, ^IRX cash, open fills, $10,000)
EXPORT = (
    "import dataclasses, sys\n"
    "import pandas as pd\n"
    "from ibs_strategy.data import load_data, load_cash_rate\n"
    "from ibs_strategy.backtest import run_backtest\n"
    "end, out = sys.argv[1], sys.argv[2]\n"
    "d = load_data('TQQQ', end=end)\n"
    "r = run_backtest(d, cash_rate=load_cash_rate(end=end))\n"
    "pd.DataFrame([dataclasses.asdict(t) for t in r.trades]).to_csv(out + '/orig_tqqq_trades.csv', index=False)\n"
    "r.data[['Open', 'High', 'Low', 'Close', 'IBS', 'Position', 'Cash', 'Capital']].to_csv(out + '/orig_tqqq_daily.csv')\n"
    "print(r.summary())\n"
)


def ref_dir(cfg: dict):
    return (ROOT / cfg["stage0"]["ref_dir"]).resolve()


def run_cli(cfg: dict, run: dict) -> str:
    cmd = ["uv", "run", "ibs", "backtest", *run["args"].split(), "--end", cfg["stage0"]["end"], "--no-plot"]
    p = subprocess.run(cmd, cwd=ref_dir(cfg), capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr[-500:]}")
    (results_dir("stage0") / f"{run['name']}.txt").write_text(p.stdout, encoding="utf-8")
    return p.stdout


def export_tqqq(cfg: dict) -> None:
    out = results_dir("stage0")
    p = subprocess.run(["uv", "run", "python", "-c", EXPORT, cfg["stage0"]["end"], str(out)],
                       cwd=ref_dir(cfg), capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"TQQQ export failed: {p.stderr[-500:]}")
    print(p.stdout.strip())


def parse(text: str) -> dict:
    out = {}
    for k, p in PAT.items():
        m = re.search(p, text)
        v = float(m.group(1)) if m else float("nan")
        out[k] = v / 100 if k in ("cagr", "mdd", "exposure") else v
    m = re.search(BARS, text)
    out.update({"bars": int(m.group(1)), "first": m.group(2), "last": m.group(3).rstrip(" (")} if m else {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    s0 = cfg["stage0"]
    rows = []
    for run in s0["runs"]:
        path = results_dir("stage0") / f"{run['name']}.txt"
        text = run_cli(cfg, run) if a.run else path.read_text(encoding="utf-8")
        got = parse(text)
        rows.append({"name": run["name"], "label": run["label"], "args": run["args"], **got,
                     **{f"orig_{k}": run[k] for k in ("cagr", "sharpe", "mdd", "trades", "exposure")},
                     "cagr_gap": got["cagr"] - run["cagr"]})
    if a.run:
        export_tqqq(cfg)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir() / "stage0.csv", index=False, encoding="utf-8-sig")
    print(df[["name", "first", "last", "cagr", "orig_cagr", "sharpe", "orig_sharpe", "mdd", "trades", "exposure", "cagr_gap"]]
          .round(4).to_string())
    big = df[df["cagr_gap"].abs() > s0["cagr_tol"]]
    if len(big):
        print(f"CAGR differs by more than {s0['cagr_tol']:.0%}p: {', '.join(big['name'])} -> find the cause, write it in NOTES.md")


if __name__ == "__main__":
    main()
