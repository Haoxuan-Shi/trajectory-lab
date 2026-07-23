"""Heading-aware deterministic state-lattice planning."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from ._numeric import (
    checked_ceil,
    checked_finite_division,
    checked_finite_product,
    checked_finite_sum,
    checked_point_distance,
)
from .collision import segment_collision
from .errors import InvalidInputError, NoPathError
from .grid import Cell, OccupancyGrid, Point


_LATTICE_ARITHMETIC_ERROR = (
    "derived state-lattice geometry or cost is not representable as a finite number"
)


@dataclass(frozen=True, slots=True, order=True)
class LatticeState:
    x: int
    y: int
    heading_bin: int

    @property
    def cell(self) -> Cell:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class LatticeResult:
    states: tuple[LatticeState, ...]
    path: tuple[Point, ...]
    cost: float
    expanded: int
    heading_bins: int
    min_turn_radius: float


def state_lattice_plan(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    *,
    start_heading: float = 0.0,
    goal_heading: float | None = None,
    heading_bins: int = 8,
    min_turn_radius: float | None = None,
    turn_penalty: float = 0.2,
    max_expansions: int | None = None,
) -> LatticeResult:
    """Plan over ``(cell, heading_bin)`` using straight/left/right primitives.

    Turning primitives travel far enough that discrete heading change divided
    by chord length does not exceed ``1 / min_turn_radius``. Every primitive is
    continuously collision checked against closed occupied cells.
    """

    _validate_free_cell(grid, start, "start")
    _validate_free_cell(grid, goal, "goal")
    if type(heading_bins) is not int or not 4 <= heading_bins <= 72:
        raise InvalidInputError("heading_bins must be an integer in [4, 72]")
    if not math.isfinite(start_heading):
        raise InvalidInputError("start_heading must be finite")
    if goal_heading is not None and not math.isfinite(goal_heading):
        raise InvalidInputError("goal_heading must be finite")
    radius = grid.resolution if min_turn_radius is None else min_turn_radius
    if not math.isfinite(radius) or radius <= 0.0:
        raise InvalidInputError("min_turn_radius must be finite and positive")
    if not math.isfinite(turn_penalty) or turn_penalty < 0.0:
        raise InvalidInputError("turn_penalty must be finite and non-negative")
    state_limit = grid.width * grid.height * heading_bins
    if max_expansions is None:
        max_expansions = state_limit
    if type(max_expansions) is not int or max_expansions <= 0:
        raise InvalidInputError("max_expansions must be a positive integer")

    angle_step = 2.0 * math.pi / heading_bins
    start_bin = _quantize_heading(start_heading, heading_bins)
    goal_bin = (
        None if goal_heading is None else _quantize_heading(goal_heading, heading_bins)
    )
    initial = LatticeState(start[0], start[1], start_bin)
    frontier: list[tuple[float, float, int, int, int]] = [
        (_heuristic(grid, start, goal), 0.0, initial.y, initial.x, initial.heading_bin)
    ]
    costs: dict[LatticeState, float] = {initial: 0.0}
    parents: dict[LatticeState, LatticeState] = {}
    expanded = 0

    while frontier:
        _priority, cost, y, x, heading_bin = heapq.heappop(frontier)
        current = LatticeState(x, y, heading_bin)
        if cost > costs.get(current, math.inf):
            continue
        if current.cell == goal and (
            goal_bin is None or current.heading_bin == goal_bin
        ):
            states = _reconstruct(parents, initial, current)
            return LatticeResult(
                states,
                tuple(grid.cell_center(state.cell) for state in states),
                cost,
                expanded,
                heading_bins,
                radius,
            )
        if expanded >= max_expansions:
            raise NoPathError(
                f"state-lattice expansion limit {max_expansions} reached"
            )
        expanded += 1
        for turn in (0, -1, 1):
            successor = _primitive_successor(
                grid, current, turn, heading_bins, radius
            )
            if successor is None:
                continue
            start_point = grid.cell_center(current.cell)
            end_point = grid.cell_center(successor.cell)
            if segment_collision(grid, start_point, end_point) is not None:
                continue
            edge_length = checked_point_distance(
                start_point, end_point, message=_LATTICE_ARITHMETIC_ERROR
            )
            turn_cost = checked_finite_product(
                turn_penalty, abs(turn), message=_LATTICE_ARITHMETIC_ERROR
            )
            candidate = checked_finite_sum(
                checked_finite_sum(
                    cost, edge_length, message=_LATTICE_ARITHMETIC_ERROR
                ),
                turn_cost,
                message=_LATTICE_ARITHMETIC_ERROR,
            )
            if candidate < costs.get(successor, math.inf):
                costs[successor] = candidate
                parents[successor] = current
                priority = checked_finite_sum(
                    candidate,
                    _heuristic(grid, successor.cell, goal),
                    message=_LATTICE_ARITHMETIC_ERROR,
                )
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        candidate,
                        successor.y,
                        successor.x,
                        successor.heading_bin,
                    ),
                )
    raise NoPathError(f"goal {goal!r} is unreachable from start {start!r}")


def lattice_is_kinematically_feasible(
    result: LatticeResult, *, tolerance: float = 1e-9
) -> bool:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise InvalidInputError("tolerance must be finite and non-negative")
    angle_step = 2.0 * math.pi / result.heading_bins
    maximum_curvature = 1.0 / result.min_turn_radius
    for first, second, first_point, second_point in zip(
        result.states, result.states[1:], result.path, result.path[1:]
    ):
        difference = (second.heading_bin - first.heading_bin) % result.heading_bins
        difference = min(difference, result.heading_bins - difference)
        heading_change = difference * angle_step
        distance = math.dist(first_point, second_point)
        required_distance = checked_finite_product(
            result.min_turn_radius,
            heading_change,
            message=_LATTICE_ARITHMETIC_ERROR,
        )
        if not _chord_safely_meets_requirement(distance, required_distance):
            return False
        if heading_change / distance > maximum_curvature + tolerance:
            return False
    return True


def _chord_safely_meets_requirement(
    actual_distance: float, required_distance: float
) -> bool:
    """Conservatively compare a represented chord with its hard minimum."""

    if (
        not math.isfinite(actual_distance)
        or not math.isfinite(required_distance)
        or actual_distance <= 0.0
        or required_distance < 0.0
    ):
        return False
    if required_distance == 0.0:
        return True
    return math.nextafter(actual_distance, 0.0) >= required_distance


def _primitive_successor(
    grid: OccupancyGrid,
    state: LatticeState,
    turn: int,
    heading_bins: int,
    radius: float,
) -> LatticeState | None:
    angle_step = 2.0 * math.pi / heading_bins
    new_heading = (state.heading_bin + turn) % heading_bins
    required_distance = (
        0.0
        if turn == 0
        else checked_finite_product(
            radius, angle_step, message=_LATTICE_ARITHMETIC_ERROR
        )
    )
    steps = max(
        1,
        checked_ceil(
            checked_finite_division(
                required_distance,
                grid.resolution,
                message=_LATTICE_ARITHMETIC_ERROR,
            ),
            message=_LATTICE_ARITHMETIC_ERROR,
        ),
    )
    maximum_steps = max(grid.width, grid.height) * 2
    if steps > maximum_steps:
        return None
    start_point = grid.cell_center(state.cell)
    while True:
        angle = new_heading * angle_step
        dx = _round_away_from_half(
            checked_finite_product(
                steps, math.cos(angle), message=_LATTICE_ARITHMETIC_ERROR
            )
        )
        dy = _round_away_from_half(
            checked_finite_product(
                steps, math.sin(angle), message=_LATTICE_ARITHMETIC_ERROR
            )
        )
        successor = LatticeState(state.x + dx, state.y + dy, new_heading)
        if not grid.in_bounds(successor.cell):
            return None
        actual_distance = checked_point_distance(
            start_point,
            grid.cell_center(successor.cell),
            message=_LATTICE_ARITHMETIC_ERROR,
        )
        if (dx != 0 or dy != 0) and _chord_safely_meets_requirement(
            actual_distance, required_distance
        ):
            break
        steps += 1
        if steps > maximum_steps:
            return None
    if not grid.is_free(successor.cell):
        return None
    return successor


def _round_away_from_half(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _quantize_heading(angle: float, bins: int) -> int:
    wrapped = angle % (2.0 * math.pi)
    return math.floor(wrapped / (2.0 * math.pi / bins) + 0.5) % bins


def _heuristic(grid: OccupancyGrid, cell: Cell, goal: Cell) -> float:
    distance_in_cells = checked_point_distance(
        cell, goal, message=_LATTICE_ARITHMETIC_ERROR
    )
    return checked_finite_product(
        grid.resolution,
        distance_in_cells,
        message=_LATTICE_ARITHMETIC_ERROR,
    )


def _validate_free_cell(grid: OccupancyGrid, cell: Cell, name: str) -> None:
    if not grid.in_bounds(cell):
        raise InvalidInputError(f"{name} cell {cell!r} is outside the grid")
    if grid.is_occupied(cell):
        raise InvalidInputError(f"{name} cell {cell!r} is occupied")


def _reconstruct(
    parents: dict[LatticeState, LatticeState],
    start: LatticeState,
    goal: LatticeState,
) -> tuple[LatticeState, ...]:
    states = [goal]
    while states[-1] != start:
        states.append(parents[states[-1]])
    states.reverse()
    return tuple(states)
