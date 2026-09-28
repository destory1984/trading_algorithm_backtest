"""Shared helpers: this repo's config and results folder. Data access comes from krxbt."""
from __future__ import annotations

from pathlib import Path

import krxbt.config
from krxbt.config import data_dir, end_date, price_dir  # noqa: F401
from krxbt.data import load_calendar, load_index, load_prices, load_universe  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    return krxbt.config.load_config(path or ROOT / "config.yaml")


def results_dir() -> Path:
    d = ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
