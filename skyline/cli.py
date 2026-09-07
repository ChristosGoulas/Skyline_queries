"""Command-line interface for skyline queries over NBA stat CSVs."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

from .algorithms import bnl, dnc, sfs
from .data import extract_points, load_csv
from .dominance import Mode, Point

ALGORITHMS: dict[str, Callable[[Sequence[Point], Sequence[Mode]], list[Point]]] = {
    "bnl": bnl,
    "sfs": sfs,
    "dnc": dnc,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skyline",
        description=(
            "Compute skyline (Pareto-optimal) players over NBA season stats. "
            "A player is in the skyline iff no other player is at least as "
            "good on every selected stat and strictly better on one."
        ),
    )
    parser.add_argument(
        "--data",
        default="data/2017_ALL.csv",
        help="CSV file with Player/Tm/stat columns (default: %(default)s)",
    )
    parser.add_argument(
        "--dims",
        default="PTS,TRB,AST,STL,BLK",
        help="comma-separated stat columns to compare (default: all five)",
    )
    parser.add_argument(
        "--min",
        dest="minimize",
        nargs="*",
        default=[],
        metavar="DIM",
        help="dimensions where smaller is better (default: maximize all)",
    )
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS),
        default="sfs",
        help="algorithm to use (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run all algorithms, print timing, and verify they agree",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    if not dims:
        print("error: at least one dimension is required", file=sys.stderr)
        return 2

    dataset = load_csv(args.data)
    try:
        points, labels = extract_points(dataset, dims)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    modes = [
        Mode.MIN if d in (args.minimize or []) else Mode.MAX for d in dims
    ]

    if args.benchmark:
        return _benchmark(points, modes, labels)

    algo = ALGORITHMS[args.algo]
    start = time.perf_counter()
    result = algo(points, modes)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Skyline over {dims} ({len(points)} players, algo={args.algo})")
    print(f"Computed in {elapsed_ms:.2f} ms — {len(result)} players:\n")
    width = max(len(labels[i]) for i in range(len(labels)))
    for point in sorted(result, reverse=True):
        player = labels[points.index(point)]
        stats = "  ".join(f"{d}={v:g}" for d, v in zip(dims, point, strict=True))
        print(f"  {player:<{width}}  {stats}")
    return 0


def _benchmark(
    points: Sequence[Point], modes: Sequence[Mode], labels: Sequence[str]
) -> int:
    print(f"Benchmarking {len(ALGORITHMS)} algorithms on {len(points)} points:\n")
    reference = None
    for name, algo in ALGORITHMS.items():
        start = time.perf_counter()
        result = algo(points, modes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        as_set = set(result)
        if reference is None:
            reference = as_set
            status = "reference"
        else:
            status = "OK" if as_set == reference else "MISMATCH!"
        print(f"  {name:>4}: {elapsed_ms:8.2f} ms   skyline={len(result):3}   {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
