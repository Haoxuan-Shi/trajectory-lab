"""Deterministic A* and Dijkstra grid search."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable

from ._numeric import checked_finite_product, checked_finite_sum
from .errors import InvalidInputError, NoPathError
from .grid import Cell, OccupancyGrid


_SEARCH_ARITHMETIC_ERROR = "derived search cost is not representable as a finite number"


@dataclass(frozen=True, slots=True)
class SearchResult:
    path: tuple[Cell, ...]
    cost: float
    expanded: int


def astar(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    *,
    diagonal: bool = True,
    allow_corner_cutting: bool = False,
) -> SearchResult:
    """Find a minimum-cost path using a consistent octile/Manhattan heuristic."""

    def heuristic(cell: Cell) -> float:
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        if diagonal:
            diagonal_adjustment = checked_finite_product(
                math.sqrt(2.0) - 1.0,
                min(dx, dy),
                message=_SEARCH_ARITHMETIC_ERROR,
            )
            distance_in_cells = checked_finite_sum(
                max(dx, dy),
                diagonal_adjustment,
                message=_SEARCH_ARITHMETIC_ERROR,
            )
        else:
            distance_in_cells = dx + dy
        return checked_finite_product(
            grid.resolution,
            distance_in_cells,
            message=_SEARCH_ARITHMETIC_ERROR,
        )

    return _search(
        grid,
        start,
        goal,
        heuristic,
        diagonal=diagonal,
        allow_corner_cutting=allow_corner_cutting,
    )


def dijkstra(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    *,
    diagonal: bool = True,
    allow_corner_cutting: bool = False,
) -> SearchResult:
    """Find a minimum-cost path with a zero heuristic."""

    return _search(
        grid,
        start,
        goal,
        lambda _cell: 0.0,
        diagonal=diagonal,
        allow_corner_cutting=allow_corner_cutting,
    )


def _search(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    heuristic: Callable[[Cell], float],
    *,
    diagonal: bool,
    allow_corner_cutting: bool,
) -> SearchResult:
    _validate_endpoint(grid, start, "start")
    _validate_endpoint(grid, goal, "goal")
    if not isinstance(diagonal, bool) or not isinstance(allow_corner_cutting, bool):
        raise InvalidInputError("search flags must be bool values")
    if start == goal:
        return SearchResult((start,), 0.0, 0)

    initial_priority = checked_finite_sum(
        0.0, heuristic(start), message=_SEARCH_ARITHMETIC_ERROR
    )
    frontier: list[tuple[float, float, int, int]] = [
        (initial_priority, 0.0, start[1], start[0])
    ]
    costs: dict[Cell, float] = {start: 0.0}
    parents: dict[Cell, Cell] = {}
    expanded = 0
    while frontier:
        _priority, cost, y, x = heapq.heappop(frontier)
        current = (x, y)
        if cost > costs.get(current, math.inf):
            continue
        if current == goal:
            return SearchResult(_reconstruct(parents, start, goal), cost, expanded)
        expanded += 1
        for neighbor, edge_cost in grid.neighbors(
            current,
            diagonal=diagonal,
            allow_corner_cutting=allow_corner_cutting,
        ):
            candidate = checked_finite_sum(
                cost, edge_cost, message=_SEARCH_ARITHMETIC_ERROR
            )
            if candidate < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate
                parents[neighbor] = current
                priority = checked_finite_sum(
                    candidate,
                    heuristic(neighbor),
                    message=_SEARCH_ARITHMETIC_ERROR,
                )
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        candidate,
                        neighbor[1],
                        neighbor[0],
                    ),
                )
    raise NoPathError(f"goal {goal!r} is unreachable from start {start!r}")


def _validate_endpoint(grid: OccupancyGrid, cell: Cell, name: str) -> None:
    if not grid.in_bounds(cell):
        raise InvalidInputError(f"{name} cell {cell!r} is outside the grid")
    if grid.is_occupied(cell):
        raise InvalidInputError(f"{name} cell {cell!r} is occupied")


def _reconstruct(parents: dict[Cell, Cell], start: Cell, goal: Cell) -> tuple[Cell, ...]:
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return tuple(path)
