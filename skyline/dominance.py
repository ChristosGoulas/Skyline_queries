"""Dominance relations for skyline queries.

A point p dominates q (written p ≺ q) iff p is at least as good as q on
*every* dimension and strictly better on at least one. "Better" is defined
per-dimension by a ``Mode``: MAX means larger values win, MIN means smaller
values win.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence

Point = tuple[float, ...]


class Mode(enum.Enum):
    """Optimization direction for a single dimension."""

    MAX = "max"
    MIN = "min"

    def better(self, a: float, b: float) -> bool:
        """Return True if value ``a`` is strictly better than ``b``."""
        return a > b if self is Mode.MAX else a < b

    def key(self, v: float) -> float:
        """Map a value so that ascending sort order = best-first."""
        return -v if self is Mode.MAX else v


def dominates(p: Point, q: Point, modes: Sequence[Mode]) -> bool:
    """Return True if point ``p`` dominates point ``q``.

    ``p`` must be at least as good as ``q`` on every dimension and strictly
    better on at least one. Points with fewer dimensions than ``modes`` are
    rejected.
    """
    if len(p) != len(q) or len(p) != len(modes):
        raise ValueError("points and modes must have equal length")
    strictly_better = False
    for a, b, mode in zip(p, q, modes, strict=True):
        if a == b:
            continue
        if mode.better(a, b):
            strictly_better = True
        else:
            return False
    return strictly_better


def dominates_or_equal(p: Point, q: Point, modes: Sequence[Mode]) -> bool:
    """Return True if ``p`` dominates ``q`` or is identical on all dims."""
    if len(p) != len(q) or len(p) != len(modes):
        raise ValueError("points and modes must have equal length")
    return all(
        not (a != b and not mode.better(a, b))
        for a, b, mode in zip(p, q, modes, strict=True)
    )


def is_dominated(p: Point, points: Sequence[Point], modes: Sequence[Mode]) -> bool:
    """Return True if any point in ``points`` dominates ``p``."""
    return any(dominates(q, p, modes) for q in points)
