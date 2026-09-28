"""S9 overnight hold: buy every close, sell at the next open."""
from __future__ import annotations

import numpy as np

ID, NAME, KIND, VERSIONS = "S9", "밤사이 보유", "engine", ("co",)


def signals(f, cal):
    on = np.ones(len(f), np.bool_)
    return on, on.copy()
