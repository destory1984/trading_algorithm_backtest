"""S1 IBS: buy when IBS < 0.2, exit when IBS > 0.8. Next-day open version only."""
from __future__ import annotations

from src.indicators import ibs

ID, NAME, KIND, VERSIONS = "S1", "IBS", "engine", ("open",)


def signals(f, cal):
    x = ibs(f["high"], f["low"], f["close"])
    return (x < 0.2).to_numpy(), (x > 0.8).to_numpy()
