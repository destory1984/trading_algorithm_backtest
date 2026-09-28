import numpy as np
import pandas as pd
import pytest

from src.indicators import ibs, pct_b, rsi


def test_rsi2_by_hand():
    # diffs: +1, -0.5, 0, +1.5, -1, +0.5; AG/AL are ewm(alpha=1/2, adjust=False) seeded by the first diff
    # AG: 1, 0.5, 0.25, 0.875, 0.4375, 0.46875   AL: 0, 0.25, 0.125, 0.0625, 0.53125, 0.265625
    c = pd.Series([10, 11, 10.5, 10.5, 12, 11, 11.5])
    r = rsi(c, 2)
    assert r.iloc[:2].isna().all()  # min_periods=2
    want = [100 - 100 / 3, 100 - 100 / 3, 100 - 100 / 15, 100 - 100 / (1 + 0.4375 / 0.53125),
            100 - 100 / (1 + 0.46875 / 0.265625)]
    assert r.iloc[2:].to_numpy() == pytest.approx(want, abs=1e-12)
    assert r.iloc[4] == pytest.approx(93.3333333333, abs=1e-9)
    assert r.iloc[5] == pytest.approx(45.1612903226, abs=1e-9)


def test_rsi2_edge_values():
    assert rsi(pd.Series([1.0, 2, 3, 4]), 2).iloc[-1] == 100.0  # AL = 0, AG > 0
    assert rsi(pd.Series([5.0, 5, 5, 5]), 2).iloc[-1] == 50.0  # both 0


def test_pct_b():
    rng = np.random.default_rng(0)
    c = pd.Series(100 + rng.normal(0, 1, 30).cumsum())
    b = pct_b(c, 20, 2.0)
    assert b.iloc[:19].isna().all()
    w = c.iloc[10:30].to_numpy()
    mid, sd = w.mean(), w.std(ddof=0)
    assert b.iloc[29] == pytest.approx((w[-1] - (mid - 2 * sd)) / (4 * sd), abs=1e-12)


def test_ibs_nan_on_flat_bar():
    x = ibs(pd.Series([12.0, 10.0]), pd.Series([10.0, 10.0]), pd.Series([11.0, 10.0]))
    assert x.iloc[0] == 0.5
    assert np.isnan(x.iloc[1])
