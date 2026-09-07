"""Benchmark skyline algorithms across dataset size (n) and dimensionality (d).

Generates synthetic anti-correlated data (the worst case for skylines, where
many points are mutually incomparable), times each algorithm over a grid of
n and d, and writes results to ``benchmarks/results.csv``.

Run from the repo root::

    python3 benchmarks/benchmark.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyline import Mode, Point, bnl, dnc, sfs

ALGORITHMS: dict[str, Callable[[Sequence[Point], Sequence[Mode]], list[Point]]] = {
    "bnl": bnl,
    "sfs": sfs,
    "dnc": dnc,
}

# Grid of (num_points, num_dims). Sizes stay small enough for O(n^2) BNL.
SIZES = [100, 250, 500, 1000, 2000]
DIMS = [2, 3, 5]
REPEATS = 3


def make_dataset(n: int, d: int, seed: int) -> list[Point]:
    """Anti-correlated points: high on one dim => low on the others.

    This maximizes skyline size, which is the stress case for the algorithms.
    """
    rng = random.Random(seed)
    points: list[Point] = []
    for _ in range(n):
        base = [rng.uniform(0, 100) for _ in range(d)]
        spike = rng.randrange(d)
        base[spike] += rng.uniform(50, 100)
        points.append(tuple(round(v, 3) for v in base))
    return points


def time_algorithm(
    algo: Callable[[Sequence[Point], Sequence[Mode]], list[Point]],
    points: list[Point],
    modes: Sequence[Mode],
    repeats: int,
) -> float:
    """Return the best-of-``repeats`` wall-clock time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        algo(points, modes)
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


def main() -> int:
    out_path = Path(__file__).resolve().parent / "results.csv"
    rows = []
    print(f"{'n':>6} {'d':>3} | " + "  ".join(f"{name:>10}" for name in ALGORITHMS))
    print("-" * 60)
    for d in DIMS:
        modes = [Mode.MAX] * d
        for n in SIZES:
            points = make_dataset(n, d, seed=n * 10 + d)
            timings = {
                name: time_algorithm(algo, points, modes, REPEATS)
                for name, algo in ALGORITHMS.items()
            }
            for name, ms in timings.items():
                rows.append({"n": n, "d": d, "algo": name, "ms": round(ms, 3)})
            print(f"{n:>6} {d:>3} | " + "  ".join(f"{timings[name]:>9.2f}m" for name in ALGORITHMS))
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "d", "algo", "ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
