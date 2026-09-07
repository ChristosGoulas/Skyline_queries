"""Correctness harness: brute-force oracle + per-algorithm test cases.

The oracle computes the skyline by definition — a point is in the skyline
iff *no other* input point dominates it — with O(n²·d) clarity. Every
optimized algorithm is tested against it on a variety of datasets,
including the real NBA CSVs and a property-style randomized sweep.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from skyline import Mode, bnl, dnc, dominates, extract_points, load_csv, sfs
from skyline.dominance import dominates_or_equal, is_dominated

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALL_MODES_5D = [Mode.MAX] * 5
ALGORITHMS = [bnl, sfs, dnc]


def brute_force(points, modes):
    """Skyline by definition: points dominated by no other input point."""
    result = []
    for i, p in enumerate(points):
        if not any(
            j != i and dominates_or_equal(q, p, modes)
            for j, q in enumerate(points)
        ):
            result.append(p)
    return result


# --------------------------------------------------------------------------
# Dominance primitives
# --------------------------------------------------------------------------

class TestDominance:
    def test_strictly_better_on_one_dim_dominates(self):
        assert dominates((2, 1), (1, 1), [Mode.MAX, Mode.MAX])

    def test_equal_points_do_not_dominate(self):
        assert not dominates((1, 1), (1, 1), [Mode.MAX, Mode.MAX])
        assert dominates_or_equal((1, 1), (1, 1), [Mode.MAX, Mode.MAX])

    def test_worse_on_one_dim_does_not_dominate(self):
        assert not dominates((1, 5), (2, 1), [Mode.MAX, Mode.MAX])

    def test_min_mode_inverts_direction(self):
        assert dominates((1, 1), (2, 1), [Mode.MIN, Mode.MAX])
        assert not dominates((3, 1), (2, 1), [Mode.MIN, Mode.MAX])

    def test_mixed_modes(self):
        # maximize points, minimize turnovers
        assert dominates((30, 2), (25, 5), [Mode.MAX, Mode.MIN])

    def test_is_dominated(self):
        window = [(5, 5), (1, 9)]
        assert is_dominated((4, 4), window, [Mode.MAX, Mode.MAX])
        assert not is_dominated((6, 1), window, [Mode.MAX, Mode.MAX])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            dominates((1,), (1, 2), [Mode.MAX, Mode.MAX])


# --------------------------------------------------------------------------
# Algorithms vs. oracle on hand-picked cases
# --------------------------------------------------------------------------

HAND_CASES = [
    # (points, modes, expected skyline as a set)
    ([(1, 2), (2, 1)], [Mode.MAX, Mode.MAX], {(1, 2), (2, 1)}),
    ([(1, 1), (2, 2)], [Mode.MAX, Mode.MAX], {(2, 2)}),
    # staircase: everything is incomparable
    ([(i, 10 - i) for i in range(10)], [Mode.MAX, Mode.MAX],
     {(i, 10 - i) for i in range(10)}),
    # single point
    ([(7, 7)], [Mode.MAX, Mode.MAX], {(7, 7)}),
    # MIN semantics: (1,5) dominates (2,3) under [MIN, MAX] because 1<=2
    # and 5>=3 — so the whole chain collapses to the single best corner.
    ([(1, 5), (2, 3), (3, 1)], [Mode.MIN, Mode.MAX], {(1, 5)}),
    # Under [MIN, MIN] the anti-diagonal is incomparable: each point is
    # smaller on one dim but larger on the other.
    ([(1, 5), (2, 3), (3, 1)], [Mode.MIN, Mode.MIN], {(1, 5), (2, 3), (3, 1)}),
    # 3-D case: (2,2,2) is dominated by (3,2,1)? No — worse on dim 2.
    # (1,1,1) is dominated by all three.
    ([(1, 2, 3), (3, 2, 1), (2, 2, 2), (1, 1, 1)], [Mode.MAX] * 3,
     {(1, 2, 3), (3, 2, 1), (2, 2, 2)}),
]


def test_duplicates_collapse_to_single_copy():
    """All algorithms must return exactly one copy of duplicate points."""
    points = [(2, 2), (2, 2), (1, 1)]
    for algo in ALGORITHMS:
        result = algo(points, [Mode.MAX, Mode.MAX])
        assert result.count((2, 2)) == 1, f"{algo.__name__} kept duplicates"
        assert set(result) == {(2, 2)}, f"{algo.__name__} wrong result"


@pytest.mark.parametrize("algo", ALGORITHMS, ids=lambda a: a.__name__)
@pytest.mark.parametrize("points,modes,expected", HAND_CASES)
def test_matches_expected_on_hand_cases(algo, points, modes, expected):
    assert set(algo(points, modes)) == expected


@pytest.mark.parametrize("algo", ALGORITHMS, ids=lambda a: a.__name__)
def test_empty_input(algo):
    assert algo([], [Mode.MAX, Mode.MAX]) == []


@pytest.mark.parametrize("algo", ALGORITHMS, ids=lambda a: a.__name__)
def test_dimension_mismatch_raises(algo):
    with pytest.raises(ValueError):
        algo([(1, 2)], [Mode.MAX] * 3)


# --------------------------------------------------------------------------
# Property-style randomized sweep: every algorithm must agree with the
# brute-force oracle on random data of varying size and dimensionality.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS, ids=lambda a: a.__name__)
@pytest.mark.parametrize("seed", range(10))
def test_matches_oracle_on_random_data(algo, seed):
    rng = random.Random(seed)
    n_dims = rng.choice([2, 3, 5])
    modes = [rng.choice([Mode.MAX, Mode.MIN]) for _ in range(n_dims)]
    # small integer ranges force duplicates and near-ties
    points = [
        tuple(rng.randint(0, 20) for _ in range(n_dims))
        for _ in range(rng.randint(0, 60))
    ]
    assert set(algo(points, modes)) == set(brute_force(points, modes))


# --------------------------------------------------------------------------
# Real data: 2017 NBA season CSVs
# --------------------------------------------------------------------------

def test_real_dataset_all_five_stats():
    dataset = load_csv(DATA_DIR / "2017_ALL.csv")
    points, _labels = extract_points(dataset, ["TRB", "AST", "STL", "BLK", "PTS"])
    expected = set(brute_force(points, ALL_MODES_5D))
    for algo in ALGORITHMS:
        result = set(algo(points, ALL_MODES_5D))
        assert result == expected, f"{algo.__name__} disagrees with oracle"
    # sanity: on real NBA data the skyline is non-empty and small
    assert 0 < len(expected) < len(points)


def test_real_dataset_per_stat_files_agree():
    """The per-stat CSVs are headerless two-column files; their 1-D skyline
    must be exactly the set of rows achieving the maximum."""
    for stat in ["PTS", "TRB", "AST", "STL", "BLK"]:
        dataset = load_csv(DATA_DIR / f"2017_{stat}.csv")
        points, _ = extract_points(dataset, [stat])
        expected = {p for p in points if not is_dominated(p, points, [Mode.MAX])}
        for algo in ALGORITHMS:
            result = algo(points, [Mode.MAX])
            assert set(result) == expected, f"{algo.__name__} on 2017_{stat}"
            assert len(result) == len(set(result)), f"{algo.__name__} kept dups"
