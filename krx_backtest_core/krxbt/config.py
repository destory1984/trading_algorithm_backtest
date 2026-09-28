"""Config loading and data paths.

The price data lives outside every algorithm repo so that all of them read the
same files. Its folder is, in order:
  1. env KRX_DATA_DIR
  2. config `data.dir`, relative to the config file (e.g. "../krx_data")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Windows consoles default to cp949; force UTF-8 so Korean names print correctly.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(path: Path | str = "config.yaml") -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.parent)
    return cfg


def data_dir(cfg: dict) -> Path:
    env = os.environ.get("KRX_DATA_DIR")
    d = Path(env) if env else Path(cfg.get("_root", ".")) / cfg["data"]["dir"]
    d = d.resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def price_dir(cfg: dict) -> Path:
    d = data_dir(cfg) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def end_date(cfg: dict) -> pd.Timestamp:
    e = cfg["data"].get("end")
    return pd.Timestamp(e) if e else pd.Timestamp.today().normalize()
