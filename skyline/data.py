"""Load NBA stat CSVs and extract points for skyline queries.

The datasets in this repo use the schema::

    ,Player,Tm,TRB,AST,STL,BLK,PTS
    1,Alex Abrines,OKC,86,40,37,8,406

i.e. an unused row number, a player name, a team, then numeric stats.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .dominance import Mode, Point

METADATA_COLUMNS = ("", "Player", "Tm")


@dataclass(frozen=True)
class Dataset:
    """A CSV loaded into memory with its stat columns."""

    path: Path
    headers: list[str]
    rows: list[list[str]]

    @property
    def stat_columns(self) -> list[str]:
        """Names of numeric (non-metadata) columns."""
        return [h for h in self.headers if h not in METADATA_COLUMNS]


def load_csv(path: str | Path) -> Dataset:
    """Load a CSV file into a :class:`Dataset`.

    Handles two on-disk shapes found in this repo:

    * ``2017_ALL.csv``: header row, then ``,Player,Tm,TRB,AST,STL,BLK,PTS``
    * ``2017_<STAT>.csv``: headerless, two columns ``row_number,stat`` —
      the stat column is named after the file suffix.
    """
    path = Path(path)
    with path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        return Dataset(path=path, headers=[], rows=[])

    first = rows[0]
    if _looks_numeric(first[1:] if len(first) > 1 else first):
        # Headerless: synthesize schema from shape.
        if len(first) == 2:
            stat = path.stem.rsplit("_", 1)[-1]  # e.g. "2017_PTS" -> "PTS"
            headers = ["", stat]
        else:
            n_stats = len(first) - len(METADATA_COLUMNS)
            headers = ["", "Player", "Tm"] + [f"STAT{i}" for i in range(n_stats)]
        return Dataset(path=path, headers=headers, rows=rows)
    return Dataset(path=path, headers=first, rows=rows[1:])


def _looks_numeric(values: Sequence[str]) -> bool:
    if not values:
        return False
    try:
        for v in values:
            float(v)
        return True
    except ValueError:
        return False


def extract_points(
    dataset: Dataset, dimensions: Sequence[str]
) -> tuple[list[Point], list[str]]:
    """Extract (points, labels) for the given stat columns.

    ``dimensions`` are column names, e.g. ``("PTS", "TRB")``, or 1-based
    stat positions (``"1"`` = first stat column) for headerless files.
    Points are tuples of floats in that column order; labels are player
    names when a ``Player`` column exists, else the row number.
    """
    indices = _resolve_indices(dataset, dimensions)
    label_idx = dataset.headers.index("Player") if "Player" in dataset.headers else 0

    points: list[Point] = []
    labels: list[str] = []
    for row in dataset.rows:
        points.append(tuple(float(row[i]) for i in indices))
        labels.append(row[label_idx])
    return points, labels


def _resolve_indices(dataset: Dataset, dimensions: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for d in dimensions:
        if d in dataset.headers:
            indices.append(dataset.headers.index(d))
        elif d.isdigit() and 1 <= int(d) <= len(dataset.headers) - len(METADATA_COLUMNS):
            indices.append(len(METADATA_COLUMNS) + int(d) - 1)
        else:
            raise ValueError(
                f"unknown dimension {d!r}; available: {dataset.stat_columns}"
            )
    return indices


def parse_modes(spec: Sequence[str]) -> dict[str, Mode]:
    """Parse CLI-style mode specs like ``PTS=max`` into {column: Mode}."""
    modes: dict[str, Mode] = {}
    for item in spec:
        column, _, direction = item.partition("=")
        modes[column.strip()] = Mode(direction.strip().lower() or "max")
    return modes
