"""Trading costs (config `costs`)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cost_factors(cfg: dict) -> tuple[float, float]:
    """(buy_cost, sell_keep) with the flat sell_tax."""
    cs = cfg["costs"]
    buy_cost = (1 + cs["slippage"]) * (1 + cs["buy_fee"])
    sell_keep = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - cs["sell_tax"])
    return buy_cost, sell_keep


def net_return(cfg: dict, entry_px, exit_px, exit_dates) -> np.ndarray:
    """Return after fees, slippage and the sell tax in force on each exit date."""
    cs = cfg["costs"]
    buy_cost, _ = cost_factors(cfg)
    dates = pd.DatetimeIndex(exit_dates)
    tax = np.full(len(dates), cs["sell_tax"], dtype=np.float64)
    for since, rate in cs.get("sell_tax_schedule") or []:
        tax[dates >= pd.Timestamp(since)] = rate
    keep = (1 - cs["slippage"]) * (1 - cs["sell_fee"] - tax)
    return np.asarray(exit_px, np.float64) * keep / (np.asarray(entry_px, np.float64) * buy_cost) - 1.0
