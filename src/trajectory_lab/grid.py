"""Occupancy-grid representation and ASCII fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Iterator

from ._numeric import (
    checked_finite_difference,
    checked_finite_division,
    checked_finite_product,
    checked_finite_sum,
    checked_floor,
    is_finite_real,
)
from .errors import InvalidInputError

Cell = tuple[int, int]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class OccupancyGrid:
    """A row-major occupancy grid.

    Cell ``(0, 0)`` is the top-left ASCII cell. World ``x`` grows right and
    world ``y`` grows down, matching row indices. World points are expressed in
    the same distance unit as ``resolution``.
    """

    width: int
    height: int
    cells: tuple[bool, ...]
    resolution: float = 1.0
    origin: Point = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width <= 0:
            raise InvalidInputError("width must be a positive integer")
        if not isinstance(self.height, int) or isinstance(self.height, bool) or self.height <= 0:
            raise InvalidInputError("height must be a positive integer")
        if self.width * self.height > 100_000_000:
            raise InvalidInputError("grid exceeds the 100,000,000-cell safety limit")
        if len(self.cells) != self.width * self.height:
            raise InvalidInputError("cell count does not match width * height")
        if any(type(value) is not bool for value in self.cells):
            raise InvalidInputError("cells must contain bool values")
        if not is_finite_real(self.resolution) or self.resolution <= 0.0:
            raise InvalidInputError("resolution must be finite and positive")
        _validate_point(self.origin, "origin")
        extent_message = "grid world extent must remain finite"
        checked_finite_sum(
            self.origin[0],
            checked_finite_product(
                self.width, self.resolution, message=extent_message
            ),
            message=extent_message,
        )
        checked_finite_sum(
            self.origin[1],
            checked_finite_product(
                self.height, self.resolution, message=extent_message
            ),
            message=extent_message,
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Iterable[bool]],
        *,
        resolution: float = 1.0,
        origin: Point = (0.0, 0.0),
    ) -> "OccupancyGrid":
        materialized = [tuple(row) for row in rows]
        if not materialized or not materialized[0]:
            raise InvalidInputError("grid rows must be non-empty")
        width = len(materialized[0])
        if any(len(row) != width for row in materialized):
            raise InvalidInputError("grid rows must have equal width")
        return cls(
            width,
            len(materialized),
            tuple(value for row in materialized for value in row),
            resolution,
            origin,
        )

    @classmethod
    def from_ascii(
        cls,
        text: str,
        *,
        resolution: float = 1.0,
        origin: Point = (0.0, 0.0),
    ) -> "OccupancyGrid":
        return parse_ascii_scenario(text, resolution=resolution, origin=origin).grid

    def in_bounds(self, cell: Cell) -> bool:
        return (
            isinstance(cell, tuple)
            and len(cell) == 2
            and type(cell[0]) is int
            and type(cell[1]) is int
            and 0 <= cell[0] < self.width
            and 0 <= cell[1] < self.height
        )

    def is_occupied(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            raise InvalidInputError(f"cell {cell!r} is outside the grid")
        return self.cells[cell[1] * self.width + cell[0]]

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.cells[cell[1] * self.width + cell[0]]

    def cell_center(self, cell: Cell) -> Point:
        if not self.in_bounds(cell):
            raise InvalidInputError(f"cell {cell!r} is outside the grid")
        message = f"cell {cell!r} center cannot be represented as finite world coordinates"
        return (
            checked_finite_sum(
                self.origin[0],
                checked_finite_product(
                    cell[0] + 0.5, self.resolution, message=message
                ),
                message=message,
            ),
            checked_finite_sum(
                self.origin[1],
                checked_finite_product(
                    cell[1] + 0.5, self.resolution, message=message
                ),
                message=message,
            ),
        )

    def world_to_cell(self, point: Point) -> Cell:
        _validate_point(point, "point")
        return (
            _world_axis_to_cell(point[0], self.origin[0], self.resolution, "x"),
            _world_axis_to_cell(point[1], self.origin[1], self.resolution, "y"),
        )

    def occupied_cells(self) -> Iterator[Cell]:
        for y in range(self.height):
            offset = y * self.width
            for x in range(self.width):
                if self.cells[offset + x]:
                    yield (x, y)

    def neighbors(
        self,
        cell: Cell,
        *,
        diagonal: bool = True,
        allow_corner_cutting: bool = False,
    ) -> tuple[tuple[Cell, float], ...]:
        if not self.in_bounds(cell):
            raise InvalidInputError(f"cell {cell!r} is outside the grid")
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if diagonal:
            directions += [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        result: list[tuple[Cell, float]] = []
        for dx, dy in directions:
            candidate = (cell[0] + dx, cell[1] + dy)
            if not self.is_free(candidate):
                continue
            if dx and dy and not allow_corner_cutting:
                if not self.is_free((cell[0] + dx, cell[1])):
                    continue
                if not self.is_free((cell[0], cell[1] + dy)):
                    continue
            cost = checked_finite_product(
                self.resolution,
                math.sqrt(2.0) if dx and dy else 1.0,
                message=(
                    "derived search cost is not representable as a finite number"
                ),
            )
            result.append((candidate, cost))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class GridScenario:
    grid: OccupancyGrid
    start: Cell | None
    goal: Cell | None


def parse_ascii_scenario(
    text: str,
    *,
    resolution: float = 1.0,
    origin: Point = (0.0, 0.0),
) -> GridScenario:
    if not isinstance(text, str):
        raise InvalidInputError("ASCII map must be text")
    rows = text.strip("\r\n").splitlines()
    if not rows or not rows[0]:
        raise InvalidInputError("ASCII map must contain at least one cell")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise InvalidInputError("ASCII map rows must have equal width")
    start: Cell | None = None
    goal: Cell | None = None
    cells: list[bool] = []
    for y, row in enumerate(rows):
        for x, character in enumerate(row):
            if character not in "#.SG ":
                raise InvalidInputError(
                    f"unsupported map character {character!r} at ({x}, {y})"
                )
            if character == "S":
                if start is not None:
                    raise InvalidInputError("ASCII map contains multiple starts")
                start = (x, y)
            elif character == "G":
                if goal is not None:
                    raise InvalidInputError("ASCII map contains multiple goals")
                goal = (x, y)
            cells.append(character == "#")
    return GridScenario(
        OccupancyGrid(width, len(rows), tuple(cells), resolution, origin),
        start,
        goal,
    )


def load_ascii_scenario(
    path: str | Path,
    *,
    resolution: float = 1.0,
    origin: Point = (0.0, 0.0),
) -> GridScenario:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise InvalidInputError(f"cannot read map {path!s}: {error}") from error
    return parse_ascii_scenario(text, resolution=resolution, origin=origin)


def _validate_point(point: Point, name: str) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 2
        or not all(is_finite_real(value) for value in point)
    ):
        raise InvalidInputError(f"{name} must be a tuple of two finite numbers")


def _world_axis_to_cell(
    coordinate: float, origin: float, resolution: float, axis: str
) -> int:
    message = f"point {axis}-coordinate cannot be mapped to a grid cell"
    offset = checked_finite_difference(coordinate, origin, message=message)
    normalized = checked_finite_division(offset, resolution, message=message)
    return checked_floor(normalized, message=message)
