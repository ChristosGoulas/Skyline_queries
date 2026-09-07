"""Skyline queries over NBA statistics.

Public API:

* :class:`Mode`, :func:`dominates` — dominance primitives
* :func:`bnl`, :func:`sfs`, :func:`dnc` — skyline algorithms
* :func:`load_csv`, :func:`extract_points` — data loading
"""

from .algorithms import bnl, dnc, sfs
from .data import Dataset, extract_points, load_csv
from .dominance import Mode, Point, dominates, dominates_or_equal, is_dominated

__all__ = [
    "Dataset",
    "Mode",
    "Point",
    "bnl",
    "dnc",
    "dominates",
    "dominates_or_equal",
    "extract_points",
    "is_dominated",
    "load_csv",
    "sfs",
]

__version__ = "0.1.0"
