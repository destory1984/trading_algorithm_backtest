import numpy as np
import pandas as pd
import pytest

from src import equity

COSTS = {"buy_fee": 0.0007, "sell_fee": 0.0007, "sell_tax": 0.0, "slippage": 0.0005}


def trades(rows, dates, costs):
    tr = pd.DataFrame(rows, columns=["e", "x", "entry_px", "exit_px"])
    tr["entry_date"], tr["exit_date"] = dates[tr["e"]], dates[tr["x"]]
    tr["signal_date"] = tr["entry_date"]
    tr["ret"] = equity.trade_rets(tr, costs)
    return tr


@pytest.mark.parametrize("on_close", [True, False])
def test_curve_last_value_equals_product(on_close):
    dates = pd.bdate_range("2020-01-01", periods=12)
    close = np.array([10, 10.5, 10.2, 11, 10.8, 11.3, 11.1, 11.9, 12.2, 11.7, 12.4, 12.0])
    # a multi-day trade, a same-day trade, and an exit at the open followed by a new entry that day
    rows = [(1, 4, 10.4, 10.9), (5, 5, 11.0, 11.3), (7, 8, 11.8, 12.1), (8, 10, 12.0, 12.3)]
    tr = trades(rows, dates, COSTS)
    m, d = equity.metrics(tr, dates, close, on_close, COSTS)
    assert m["curve_last"] == pytest.approx(np.prod(1 + tr["ret"]), abs=1e-12)
    assert m["curve_check"]


def test_exposure_counts_same_day_trade_and_skips_close_entry_day():
    dates = pd.bdate_range("2020-01-01", periods=6)
    close = np.array([10, 10.5, 10.2, 11, 10.8, 11.3])
    tr = trades([(1, 3, 10.5, 11.0), (4, 4, 10.9, 10.8)], dates, COSTS)
    _, pos = equity.curve(tr, dates, close, True, COSTS)
    assert pos.tolist() == [False, False, True, True, True, False]


def test_breakeven():
    dates = pd.bdate_range("2020-01-01", periods=3)
    tr = trades([(0, 1, 100.0, 101.0)], dates, COSTS)
    bp = equity.breakeven_bp(tr)
    assert 101 * (1 - bp / 1e4) / (100 * (1 + bp / 1e4)) == pytest.approx(1, abs=1e-12)
