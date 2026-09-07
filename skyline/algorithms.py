"""Skyline (Pareto frontier) algorithms.

All algorithms share the same contract:

* Input: a sequence of points (tuples of floats) and one ``Mode`` per dim.
* Output: the list of input points that are not dominated by any other
  input point. Order may differ between algorithms; compare as sets.

Implemented:

* :func:`bnl`  — Block Nested Loop. O(n² · d) worst case, no preprocessing.
* :func:`sfs`  — Sort-First-Skyline. Sorts by a monotone score so a point
  can only be dominated by points seen earlier; each candidate is checked
  only against the current skyline. Same worst case, much faster in practice.
* :func:`dnc`  — Divide & Conquer along the median of the first dimension.
  O(n · log^(d-2) n) for d ≥ 3, O(n log n) for d = 2.
"""

from __future__ import annotations

from collections.abc import Sequence

from .dominance import Mode, Point, dominates


def bnl(points: Sequence[Point], modes: Sequence[Mode]) -> list[Point]:
    """Block Nested Loop skyline.

    Maintains a window of incomparable points. Each candidate is compared
    against every window member: members it dominates are evicted, and the
    candidate is added only if no member dominates it.
    """
    _validate(points, modes)
    window: list[Point] = []
    for candidate in points:
        dominated_by_candidate = [
            member for member in window if dominates(candidate, member, modes)
        ]
        if any(dominates(member, candidate, modes) for member in window):
            continue
        # An identical point already in the window: dominance requires a
        # strictly-better dim, so equal points never dominate each other —
        # skip explicitly to keep a single copy.
        if any(member == candidate for member in window):
            continue
        for member in dominated_by_candidate:
            window.remove(member)
        window.append(candidate)
    return window


def sfs(points: Sequence[Point], modes: Sequence[Mode]) -> list[Point]:
    """Sort-First-Skyline.

    Points are sorted best-first by the sum of per-dimension sort keys (a
    monotone scoring function). After sorting, a point can only be dominated
    by earlier points, so it suffices to check each candidate against the
    skyline accumulated so far — a candidate that survives is guaranteed to
    be in the final result.
    """

    def score(p: Point) -> float:
        return sum(mode.key(v) for v, mode in zip(p, modes, strict=True))

    _validate(points, modes)
    window: list[Point] = []
    for candidate in sorted(points, key=score):
        if any(member == candidate for member in window):
            continue
        if not any(dominates(member, candidate, modes) for member in window):
            window.append(candidate)
    return window


def dnc(points: Sequence[Point], modes: Sequence[Mode]) -> list[Point]:
    """Divide & Conquer skyline.

    Splits the input at the median of the first dimension so that no point
    in the "worse" half can dominate a point in the "better" half. After
    recursing on both halves, each point of the worse half's skyline is
    merged only if it is not dominated by any point of the better half's
    skyline.
    """
    _validate(points, modes)
    # Dedupe identical points: a point never dominates an identical one, but
    # two copies must not both survive, and ties on the split dimension would
    # otherwise let a duplicate straddle the divide & conquer partition.
    pts = list(dict.fromkeys(points))
    if len(pts) <= 1:
        return pts

    pts.sort(key=lambda p: (modes[0].key(p[0]), p))
    median = len(pts) // 2
    split_value = pts[median][0]
    # Move the split boundary so points sharing the median's first-dimension
    # value stay together — otherwise one could dominate a point across the
    # split and the cross check would never compare them.
    while median > 1 and pts[median - 1][0] == split_value:
        median -= 1
    better_half, worse_half = pts[:median], pts[median:]

    skyline_better = dnc(better_half, modes)
    skyline_worse = dnc(worse_half, modes)

    # Merge: filter in *both* directions. Points of the worse half can be
    # dominated by better-half points (checked here), but a better-half point
    # can also be dominated by a worse-half one — e.g. when the better half
    # was a singleton, its recursive "skyline" is itself, unchecked.
    result = [p for p in skyline_better
              if not any(dominates(q, p, modes) for q in skyline_worse)]
    for candidate in skyline_worse:
        if not any(dominates(p, candidate, modes) for p in skyline_better):
            result.append(candidate)
    return result


def _validate(points: Sequence[Point], modes: Sequence[Mode]) -> None:
    if not modes:
        raise ValueError("at least one dimension/mode is required")
    for p in points:
        if len(p) != len(modes):
            raise ValueError(
                f"point {p!r} has {len(p)} dims, expected {len(modes)}"
            )
