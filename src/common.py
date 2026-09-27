"""Shared helpers: config loading and data paths."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Windows consoles default to cp949; force UTF-8 so Korean names print correctly.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_config(path: Path | str | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> Path:
    d = ROOT / cfg["data"]["dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def price_dir(cfg: dict) -> Path:
    d = data_dir(cfg) / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def end_date(cfg: dict) -> pd.Timestamp:
    e = cfg["data"].get("end")
    return pd.Timestamp(e) if e else pd.Timestamp.today().normalize()


def load_index(cfg: dict, market: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir(cfg) / f"index_{market}.parquet")


def load_calendar(cfg: dict) -> pd.DatetimeIndex:
    """Trading calendar = dates of the KOSPI index."""
    return pd.DatetimeIndex(load_index(cfg, "KOSPI").index)


def load_universe(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(data_dir(cfg) / "universe.parquet")


def load_prices(cfg: dict, ticker: str) -> pd.DataFrame | None:
    p = price_dir(cfg) / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else None
