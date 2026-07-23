"""Conservative continuous collision checks against occupied grid cells."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from ._numeric import (
    checked_finite_difference,
    checked_finite_product,
    checked_finite_sum,
    is_finite_real,
)
from .errors import InvalidInputError
from .grid import Cell, OccupancyGrid, Point


@dataclass(frozen=True, slots=True)
class CollisionHit:
    segment_index: int
    cell: Cell
    fraction: float
    point: Point


ExactPoint = tuple[Fraction, Fraction]


def segment_collision(
    grid: OccupancyGrid,
    start: Point,
    end: Point,
    *,
    segment_index: int = 0,
) -> CollisionHit | None:
    """Return the first occupied cell touched by the closed line segment.

    Each potentially intersected occupied cell is treated as a closed
    axis-aligned box and tested with a slab intersection. Boundary contact is
    therefore conservatively considered collision.
    """

    _validate_point(start, "start")
    _validate_point(end, "end")
    if type(segment_index) is not int or segment_index < 0:
        raise InvalidInputError("segment_index must be a non-negative integer")

    start_cell = grid.world_to_cell(start)
    end_cell = grid.world_to_cell(end)
    geometry_message = (
        f"segment {segment_index} geometry exceeds the finite numeric range"
    )
    # Reject endpoint differences that cannot be represented by the public
    # binary64 API, while retaining the exact represented endpoint values for
    # the narrow-phase intersection below.
    for start_coordinate, end_coordinate in zip(start, end):
        checked_finite_difference(
            end_coordinate, start_coordinate, message=geometry_message
        )
    exact_start = (Fraction(start[0]), Fraction(start[1]))
    exact_end = (Fraction(end[0]), Fraction(end[1]))
    resolution = grid.resolution
    ox, oy = grid.origin
    min_x = max(0, min(start_cell[0], end_cell[0]) - 1)
    max_x = min(grid.width - 1, max(start_cell[0], end_cell[0]) + 1)
    min_y = max(0, min(start_cell[1], end_cell[1]) - 1)
    max_y = min(grid.height - 1, max(start_cell[1], end_cell[1]) + 1)
    if min_x > max_x or min_y > max_y:
        return None

    candidates: list[tuple[Fraction, int, int]] = []
    for fraction, cell in ((Fraction(0), start_cell), (Fraction(1), end_cell)):
        if grid.in_bounds(cell) and grid.is_occupied(cell):
            candidates.append((fraction, cell[1], cell[0]))
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if not grid.is_occupied((x, y)):
                continue
            lower_x = checked_finite_sum(
                ox,
                checked_finite_product(x, resolution, message=geometry_message),
                message=geometry_message,
            )
            upper_x = checked_finite_sum(
                lower_x, resolution, message=geometry_message
            )
            lower_y = checked_finite_sum(
                oy,
                checked_finite_product(y, resolution, message=geometry_message),
                message=geometry_message,
            )
            upper_y = checked_finite_sum(
                lower_y, resolution, message=geometry_message
            )
            fraction = _segment_box_entry(
                exact_start,
                exact_end,
                Fraction(lower_x),
                Fraction(upper_x),
                Fraction(lower_y),
                Fraction(upper_y),
            )
            if fraction is not None:
                candidates.append((fraction, y, x))
    if not candidates:
        return None
    exact_fraction, y, x = min(candidates)
    fraction = float(exact_fraction)
    exact_point = (
        exact_start[0] + exact_fraction * (exact_end[0] - exact_start[0]),
        exact_start[1] + exact_fraction * (exact_end[1] - exact_start[1]),
    )
    point = (
        float(exact_point[0]),
        float(exact_point[1]),
    )
    return CollisionHit(segment_index, (x, y), fraction, point)


def first_path_collision(
    grid: OccupancyGrid, path: Sequence[Point]
) -> CollisionHit | None:
    materialized = _validate_path(path)
    for index, (start, end) in enumerate(zip(materialized, materialized[1:])):
        hit = segment_collision(grid, start, end, segment_index=index)
        if hit is not None:
            return hit
    return None


def path_is_collision_free(grid: OccupancyGrid, path: Sequence[Point]) -> bool:
    return first_path_collision(grid, path) is None


def _segment_box_entry(
    start: ExactPoint,
    end: ExactPoint,
    lower_x: Fraction,
    upper_x: Fraction,
    lower_y: Fraction,
    upper_y: Fraction,
) -> Fraction | None:
    """Return an exact entry parameter for represented binary64 geometry.

    ``Fraction`` preserves every input float exactly.  This avoids both fixed
    absolute tolerances (which can join genuinely disjoint slab intervals) and
    rounded-division gaps at a true closed-boundary contact.
    """

    t_min = Fraction(0)
    t_max = Fraction(1)
    for position, axis_end, lower, upper in (
        (start[0], end[0], lower_x, upper_x),
        (start[1], end[1], lower_y, upper_y),
    ):
        axis_delta = axis_end - position
        if axis_delta == 0:
            if position < lower or position > upper:
                return None
            continue
        first = (lower - position) / axis_delta
        second = (upper - position) / axis_delta
        if first > second:
            first, second = second, first
        t_min = max(t_min, first)
        t_max = min(t_max, second)
        if t_min > t_max:
            return None
    return t_min


def _validate_path(path: Sequence[Point]) -> tuple[Point, ...]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise InvalidInputError("path must be a sequence of points")
    materialized = tuple(path)
    if len(materialized) < 2:
        raise InvalidInputError("path must contain at least two points")
    for index, point in enumerate(materialized):
        _validate_point(point, f"path[{index}]")
    return materialized


def _validate_point(point: Point, name: str) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 2
        or not all(is_finite_real(value) for value in point)
    ):
        raise InvalidInputError(f"{name} must be a tuple of two finite numbers")
