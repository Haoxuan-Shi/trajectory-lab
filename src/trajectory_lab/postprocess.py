"""Line-of-sight shortcutting and curvature-aware deterministic smoothing."""

from __future__ import annotations

import math
from typing import Sequence

from .collision import first_path_collision, segment_collision
from .errors import InvalidInputError
from .grid import Cell, OccupancyGrid, Point


def cells_to_points(grid: OccupancyGrid, cells: Sequence[Cell]) -> tuple[Point, ...]:
    if not cells:
        raise InvalidInputError("cell path must not be empty")
    return tuple(grid.cell_center(cell) for cell in cells)


def path_length(path: Sequence[Point]) -> float:
    points = _materialize_path(path, minimum=1)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def polyline_curvatures(path: Sequence[Point]) -> tuple[float, ...]:
    points = _materialize_path(path, minimum=2)
    curvatures = [0.0] * len(points)
    for index in range(1, len(points) - 1):
        curvatures[index] = _curvature(
            points[index - 1], points[index], points[index + 1]
        )
    return tuple(curvatures)


def shortcut_path(grid: OccupancyGrid, path: Sequence[Point]) -> tuple[Point, ...]:
    """Greedily retain the farthest visible point from each anchor."""

    points = _materialize_path(path, minimum=2)
    original_hit = first_path_collision(grid, points)
    if original_hit is not None:
        raise InvalidInputError(
            f"input path collides with cell {original_hit.cell} on segment "
            f"{original_hit.segment_index}"
        )
    result = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        selected = anchor + 1
        for candidate in range(len(points) - 1, anchor, -1):
            if segment_collision(grid, points[anchor], points[candidate]) is None:
                selected = candidate
                break
        result.append(points[selected])
        anchor = selected
    return tuple(result)


def curvature_aware_smooth(
    grid: OccupancyGrid,
    path: Sequence[Point],
    *,
    data_weight: float = 0.2,
    smooth_weight: float = 0.35,
    max_curvature: float | None = None,
    iterations: int = 120,
    tolerance: float = 1e-9,
    preserve_collision_free: bool = True,
) -> tuple[Point, ...]:
    """Smooth interior points with optional local curvature and collision guards.

    The routine uses a fixed-order Gauss-Seidel update. If a candidate exceeds
    ``max_curvature``, deterministic bisection moves it toward the neighboring
    midpoint. Collision-preserving mode rejects any update whose adjacent
    closed segments touch an occupied cell.
    """

    points = _materialize_path(path, minimum=2)
    _nonnegative_finite(data_weight, "data_weight")
    _nonnegative_finite(smooth_weight, "smooth_weight")
    if data_weight == 0.0 and smooth_weight == 0.0:
        raise InvalidInputError("at least one smoothing weight must be positive")
    if max_curvature is not None and (
        not math.isfinite(max_curvature) or max_curvature <= 0.0
    ):
        raise InvalidInputError("max_curvature must be finite and positive")
    if type(iterations) is not int or iterations <= 0 or iterations > 1_000_000:
        raise InvalidInputError("iterations must be in [1, 1000000]")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise InvalidInputError("tolerance must be finite and non-negative")
    if type(preserve_collision_free) is not bool:
        raise InvalidInputError("preserve_collision_free must be a bool")
    if preserve_collision_free:
        hit = first_path_collision(grid, points)
        if hit is not None:
            raise InvalidInputError(
                f"input path collides with cell {hit.cell} on segment {hit.segment_index}"
            )

    original = points
    smoothed = [tuple(point) for point in points]
    for _iteration in range(iterations):
        maximum_change = 0.0
        for index in range(1, len(smoothed) - 1):
            current = smoothed[index]
            candidate = (
                current[0]
                + data_weight * (original[index][0] - current[0])
                + smooth_weight
                * (smoothed[index - 1][0] + smoothed[index + 1][0] - 2.0 * current[0]),
                current[1]
                + data_weight * (original[index][1] - current[1])
                + smooth_weight
                * (smoothed[index - 1][1] + smoothed[index + 1][1] - 2.0 * current[1]),
            )
            if max_curvature is not None:
                candidate = _limit_curvature(
                    smoothed[index - 1], candidate, smoothed[index + 1], max_curvature
                )
            if preserve_collision_free and (
                segment_collision(grid, smoothed[index - 1], candidate) is not None
                or segment_collision(grid, candidate, smoothed[index + 1]) is not None
            ):
                continue
            maximum_change = max(maximum_change, math.dist(current, candidate))
            smoothed[index] = candidate
        if maximum_change <= tolerance:
            break
    result = tuple(smoothed)
    if preserve_collision_free and first_path_collision(grid, result) is not None:
        raise RuntimeError("internal error: collision-preserving smoother lost feasibility")
    return result


def _limit_curvature(
    previous: Point, candidate: Point, following: Point, limit: float
) -> Point:
    if _curvature(previous, candidate, following) <= limit:
        return candidate
    midpoint = ((previous[0] + following[0]) / 2.0, (previous[1] + following[1]) / 2.0)
    low = 0.0
    high = 1.0
    for _ in range(40):
        fraction = (low + high) / 2.0
        trial = (
            candidate[0] + fraction * (midpoint[0] - candidate[0]),
            candidate[1] + fraction * (midpoint[1] - candidate[1]),
        )
        if _curvature(previous, trial, following) <= limit:
            high = fraction
        else:
            low = fraction
    return (
        candidate[0] + high * (midpoint[0] - candidate[0]),
        candidate[1] + high * (midpoint[1] - candidate[1]),
    )


def _curvature(first: Point, middle: Point, last: Point) -> float:
    a = math.dist(first, middle)
    b = math.dist(middle, last)
    c = math.dist(first, last)
    if min(a, b, c) <= 1e-15:
        return math.inf
    twice_area = abs(
        (middle[0] - first[0]) * (last[1] - first[1])
        - (middle[1] - first[1]) * (last[0] - first[0])
    )
    return 2.0 * twice_area / (a * b * c)


def _materialize_path(path: Sequence[Point], *, minimum: int) -> tuple[Point, ...]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise InvalidInputError("path must be a sequence of points")
    points = tuple(path)
    if len(points) < minimum:
        raise InvalidInputError(f"path must contain at least {minimum} points")
    for index, point in enumerate(points):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in point
            )
        ):
            raise InvalidInputError(f"path[{index}] must contain two finite numbers")
    return points


def _nonnegative_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise InvalidInputError(f"{name} must be finite and non-negative")

