"""Generate the figures used in the README.

* ``docs/skyline_pts_trb.png`` — the 2017 NBA skyline over PTS vs TRB
* ``docs/benchmark.png`` — algorithm runtime vs. dataset size/dimensionality

Run from the repo root (requires matplotlib, and results.csv from the
benchmark sweep)::

    python3 benchmarks/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skyline import extract_points, load_csv
from skyline.visualize import plot_benchmark, plot_skyline_2d

DOCS = ROOT / "docs"


def main() -> int:
    DOCS.mkdir(exist_ok=True)

    dataset = load_csv(ROOT / "data" / "2017_ALL.csv")
    points, labels = extract_points(dataset, ["PTS", "TRB"])
    out1 = plot_skyline_2d(points, labels, ("PTS", "TRB"), DOCS / "skyline_pts_trb.png")
    print(f"wrote {out1}")

    results = ROOT / "benchmarks" / "results.csv"
    if results.exists():
        out2 = plot_benchmark(results, DOCS / "benchmark.png")
        print(f"wrote {out2}")
    else:
        print("skipped benchmark plot (run benchmarks/benchmark.py first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
