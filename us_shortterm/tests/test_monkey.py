import numpy as np

from src.deep import monkey


def test_place_keeps_blocks_apart_and_inside():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lengths = rng.integers(2, 6, size=20)
        s = monkey.place(200, lengths, rng)
        ends = s + lengths
        assert s[0] >= 0 and ends[-1] <= 199
        assert (s[1:] >= ends[:-1]).all()


def test_place_tight_fit():
    s = monkey.place(11, np.array([5, 5]), np.random.default_rng(0))
    assert list(s) == [0, 5]
