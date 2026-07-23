"""Deterministic CPU-only trajectory planning and validation."""

from .collision import (
    CollisionHit,
    first_path_collision,
    path_is_collision_free,
    segment_collision,
)
from .errors import InvalidInputError, NoPathError, TrajectoryLabError
from .grid import Cell, GridScenario, OccupancyGrid, Point, load_ascii_scenario, parse_ascii_scenario
from .hybrid import (
    LatticeResult,
    LatticeState,
    lattice_is_kinematically_feasible,
    state_lattice_plan,
)
from .postprocess import (
    cells_to_points,
    curvature_aware_smooth,
    path_length,
    polyline_curvatures,
    shortcut_path,
)
from .search import SearchResult, astar, dijkstra
from .timing import (
    MotionLimits,
    TimedPoint,
    TimedSegment,
    TrajectoryProfile,
    parameterize_velocity,
)
from .validation import ValidationIssue, ValidationReport, validate_trajectory

__all__ = [
    "Cell",
    "CollisionHit",
    "GridScenario",
    "InvalidInputError",
    "LatticeResult",
    "LatticeState",
    "MotionLimits",
    "NoPathError",
    "OccupancyGrid",
    "Point",
    "SearchResult",
    "TimedPoint",
    "TimedSegment",
    "TrajectoryLabError",
    "TrajectoryProfile",
    "ValidationIssue",
    "ValidationReport",
    "astar",
    "cells_to_points",
    "curvature_aware_smooth",
    "dijkstra",
    "first_path_collision",
    "lattice_is_kinematically_feasible",
    "load_ascii_scenario",
    "parameterize_velocity",
    "parse_ascii_scenario",
    "path_is_collision_free",
    "path_length",
    "polyline_curvatures",
    "segment_collision",
    "shortcut_path",
    "state_lattice_plan",
    "validate_trajectory",
]

__version__ = "0.1.0"

