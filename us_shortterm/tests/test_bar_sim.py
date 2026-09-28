import numpy as np
import pandas as pd

from src.screen import bar_sim
from src.screen.strategies import s10_vol_breakout as s10, s11_nr7 as s11


def frame(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.bdate_range("2020-01-01", periods=len(df))
    return df


def sim(mod, f):
    enter, px = mod.entries(f)
    return bar_sim.simulate(f.index, f["open"], f["close"], np.ones(len(f), bool), enter, px, mod.EXIT, 1.0, 1.0)


# S11: days 0-5 have range 2, day 6 range 1 (narrowest of 7), so day 7 has level L = high[6] = 10.5
BASE = [[10, 11, 9, 10]] * 6 + [[10, 10.5, 9.5, 10]]


def test_s11_open_above_level_buys_at_open():
    tr = sim(s11, frame(BASE + [[11, 12, 10.75, 11.5]]))
    assert len(tr) == 1 and tr["entry_px"].iloc[0] == 11 and tr["exit_px"].iloc[0] == 11.5
    assert tr["entry_date"].iloc[0] == tr["exit_date"].iloc[0]


def test_s11_high_below_level_no_entry():
    assert sim(s11, frame(BASE + [[10, 10.25, 9.5, 10]])).empty


def test_s11_high_equal_level_no_entry():
    assert sim(s11, frame(BASE + [[10, 10.5, 9.5, 10]])).empty


# S10: day 0 range 2, so day 1 level = open + 1
def test_s10_high_equal_level_buys_at_level():
    tr = sim(s10, frame([[10, 11, 9, 10], [10, 11, 9.5, 10.5], [10.75, 10.75, 10.75, 10.75]]))
    assert len(tr) >= 1
    t = tr.iloc[0]
    assert t["entry_px"] == 11 and t["exit_px"] == 10.75 and t["hold_days"] == 2


def test_s10_high_below_level_no_entry():
    tr = sim(s10, frame([[10, 11, 9, 10], [10, 10.5, 9.5, 10.5], [10.5, 10.5, 10.5, 10.5]]))
    assert tr.empty


def test_s10_open_at_level_buys_at_open():
    # a zero range yesterday puts the level at today's open, so today's open is the fill
    tr = sim(s10, frame([[10, 10, 10, 10], [10.5, 11, 10, 10.75], [11, 11, 11, 11]]))
    assert tr["entry_px"].iloc[0] == 10.5


def test_s10_reenters_on_exit_day():
    tr = sim(s10, frame([[10, 11, 9, 10], [10, 12, 9.5, 11], [11, 14, 11, 13], [13, 13, 13, 13]]))
    assert tr["exit_date"].iloc[0] == tr["entry_date"].iloc[1]
