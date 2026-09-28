"""SPEC 4.1: trade lists for judged + reference tickers x {bear, all}.
Usage: python -m src.run   -> results/trades_<ticker>_<version>.parquet
"""
from __future__ import annotations

from krxbt import engine

from src import common as C


def main() -> None:
    cfg = C.load_config()
    _, frames, _ = C.load_frames(cfg)
    out = C.results_dir()
    for t, f in frames.items():
        a = engine.arrays(f)
        for v in C.VERSIONS:
            tr = C.trades(f, cfg, v, a)
            tr.to_parquet(out / f"trades_{t}_{v}.parquet")
            print(t, v, len(tr))


if __name__ == "__main__":
    main()
