"""Matplotlib visualizations for skyline queries.

Two figures:

* :func:`plot_skyline_2d` — a 2-D scatter of a dataset with the skyline
  points and the Pareto frontier highlighted.
* :func:`plot_benchmark` — runtime vs. dataset size for each algorithm,
  one subplot per dimensionality.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering to file
import matplotlib.pyplot as plt

from .algorithms import sfs
from .dominance import Mode, Point


def plot_skyline_2d(
    points: Sequence[Point],
    labels: Sequence[str],
    dims: tuple[str, str],
    out: str | Path,
    modes: tuple[Mode, Mode] = (Mode.MAX, Mode.MAX),
) -> Path:
    """Scatter all points, highlight the skyline and its frontier line."""
    if len(dims) != 2:
        raise ValueError("plot_skyline_2d needs exactly two dimensions")
    skyline = sfs(points, list(modes))
    skyline_set = set(skyline)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, s=18, alpha=0.35, label="players", color="#7f8c8d")

    # Frontier: sort skyline by dim0, keep it a valid staircase for MAX/MAX.
    frontier = sorted(skyline, key=lambda p: p[0])
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    ax.plot(fx, fy, color="#c0392b", linewidth=1.5, alpha=0.8, zorder=2)
    ax.scatter(fx, fy, s=70, color="#c0392b", zorder=3, label="skyline")

    for p in frontier:
        idx = points.index(p)
        ax.annotate(
            labels[idx],
            (p[0], p[1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_title(f"NBA 2017 skyline: {dims[0]} vs {dims[1]} "
                 f"({len(skyline_set)} of {len(points)} players on the frontier)")
    ax.legend()
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_benchmark(results_csv: str | Path, out: str | Path) -> Path:
    """Plot runtime vs. n for each algorithm, one subplot per dimension."""
    results_csv = Path(results_csv)
    with results_csv.open() as f:
        rows = list(csv.DictReader(f))

    dims = sorted({int(r["d"]) for r in rows})
    algos = sorted({r["algo"] for r in rows})
    colors = {"bnl": "#7f8c8d", "sfs": "#2980b9", "dnc": "#e67e22"}

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.5), sharey=False)
    if len(dims) == 1:
        axes = [axes]
    for ax, d in zip(axes, dims, strict=True):
        for algo in algos:
            pts = sorted(
                (int(r["n"]), float(r["ms"]))
                for r in rows
                if r["algo"] == algo and int(r["d"]) == d
            )
            ns = [p[0] for p in pts]
            ms = [p[1] for p in pts]
            ax.plot(ns, ms, marker="o", label=algo, color=colors.get(algo))
        ax.set_title(f"d = {d}")
        ax.set_xlabel("n (points)")
        ax.set_ylabel("time (ms)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Skyline algorithm runtime (anti-correlated data, best of 3)")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
