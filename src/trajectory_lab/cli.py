"""Command-line planning and validation interface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Sequence, TextIO

from .errors import InvalidInputError, NoPathError, TrajectoryLabError
from .grid import Cell, GridScenario, Point, load_ascii_scenario
from .hybrid import state_lattice_plan
from .postprocess import cells_to_points, curvature_aware_smooth, shortcut_path
from .search import astar, dijkstra
from .timing import MotionLimits, TrajectoryProfile, parameterize_velocity
from .validation import ValidationReport, validate_trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajectory-lab",
        description="Deterministic occupancy-grid trajectory planning and validation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="plan, post-process, and time a trajectory")
    plan.add_argument("--map", required=True, help="ASCII occupancy-map fixture")
    plan.add_argument("--algorithm", choices=("astar", "dijkstra", "lattice"), default="astar")
    plan.add_argument("--start", type=_parse_cell, help="override map S as X,Y")
    plan.add_argument("--goal", type=_parse_cell, help="override map G as X,Y")
    plan.add_argument("--resolution", type=_positive_float, default=1.0)
    plan.add_argument("--no-diagonal", action="store_true")
    plan.add_argument("--allow-corner-cutting", action="store_true")
    plan.add_argument("--start-heading", type=_finite_float, default=0.0)
    plan.add_argument("--goal-heading", type=_finite_float)
    plan.add_argument("--heading-bins", type=int, default=8)
    plan.add_argument("--min-turn-radius", type=_positive_float)
    plan.add_argument("--shortcut", action="store_true")
    plan.add_argument("--smooth", action="store_true")
    plan.add_argument("--smooth-iterations", type=int, default=120)
    plan.add_argument("--max-curvature", type=_positive_float)
    plan.add_argument("--max-speed", type=_positive_float, default=5.0)
    plan.add_argument("--max-acceleration", type=_positive_float, default=2.0)
    plan.add_argument("--max-lateral-acceleration", type=_positive_float, default=2.5)
    plan.add_argument("--start-speed", type=_nonnegative_float, default=0.0)
    plan.add_argument("--end-speed", type=_nonnegative_float, default=0.0)
    plan.add_argument("--output", default="-", help="trajectory CSV path or '-' for stdout")
    plan.add_argument("--report", help="validation JSON path or '-' for stdout")

    validate = subparsers.add_parser("validate", help="validate a path CSV against a map")
    validate.add_argument("--map", required=True)
    validate.add_argument("--path", required=True, help="CSV with x and y columns")
    validate.add_argument("--resolution", type=_positive_float, default=1.0)
    validate.add_argument("--max-curvature", type=_positive_float)
    validate.add_argument("--report", default="-", help="JSON path or '-' for stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "plan":
            return _run_plan(arguments)
        return _run_validate(arguments)
    except NoPathError as error:
        print(f"trajectory-lab: no path: {error}", file=sys.stderr)
        return 3
    except (InvalidInputError, OSError, csv.Error) as error:
        print(f"trajectory-lab: invalid input: {error}", file=sys.stderr)
        return 2
    except TrajectoryLabError as error:
        print(f"trajectory-lab: {error}", file=sys.stderr)
        return 2


def _run_plan(arguments: argparse.Namespace) -> int:
    if arguments.output == "-" and arguments.report == "-":
        raise InvalidInputError("CSV output and JSON report cannot both use stdout")
    scenario = load_ascii_scenario(arguments.map, resolution=arguments.resolution)
    start = _select_endpoint(arguments.start, scenario.start, "start")
    goal = _select_endpoint(arguments.goal, scenario.goal, "goal")
    if start == goal:
        raise InvalidInputError("start and goal must differ for trajectory output")

    if arguments.algorithm == "lattice":
        result = state_lattice_plan(
            scenario.grid,
            start,
            goal,
            start_heading=arguments.start_heading,
            goal_heading=arguments.goal_heading,
            heading_bins=arguments.heading_bins,
            min_turn_radius=arguments.min_turn_radius,
        )
        points = result.path
        cost = result.cost
        expanded = result.expanded
    else:
        search = astar if arguments.algorithm == "astar" else dijkstra
        result = search(
            scenario.grid,
            start,
            goal,
            diagonal=not arguments.no_diagonal,
            allow_corner_cutting=arguments.allow_corner_cutting,
        )
        points = cells_to_points(scenario.grid, result.path)
        cost = result.cost
        expanded = result.expanded

    if arguments.shortcut:
        points = shortcut_path(scenario.grid, points)
    if arguments.smooth:
        points = curvature_aware_smooth(
            scenario.grid,
            points,
            max_curvature=arguments.max_curvature,
            iterations=arguments.smooth_iterations,
        )
    limits = MotionLimits(
        arguments.max_speed,
        arguments.max_acceleration,
        arguments.max_lateral_acceleration,
    )
    profile = parameterize_velocity(
        points,
        limits,
        start_speed=arguments.start_speed,
        end_speed=arguments.end_speed,
    )
    report = validate_trajectory(
        scenario.grid,
        points,
        profile=profile,
        max_curvature=arguments.max_curvature,
    )
    _write_profile(arguments.output, profile)
    if arguments.report:
        payload = report.to_dict()
        payload["planner"] = {
            "algorithm": arguments.algorithm,
            "cost": cost,
            "expanded": expanded,
        }
        _write_json(arguments.report, payload)
    print(
        f"trajectory-lab: {arguments.algorithm} produced {len(points)} points; "
        f"length={profile.total_length:.6g}; valid={str(report.valid).lower()}",
        file=sys.stderr,
    )
    return 0 if report.valid else 4


def _run_validate(arguments: argparse.Namespace) -> int:
    scenario = load_ascii_scenario(arguments.map, resolution=arguments.resolution)
    points = _read_path_csv(arguments.path)
    report = validate_trajectory(
        scenario.grid, points, max_curvature=arguments.max_curvature
    )
    _write_json(arguments.report, report.to_dict())
    return 0 if report.valid else 4


def _write_profile(destination: str, profile: TrajectoryProfile) -> None:
    stream, close = _open_text_output(destination)
    try:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("x", "y", "arc_length", "speed", "time", "curvature"))
        for point in profile.points:
            writer.writerow(
                tuple(
                    format(value, ".17g")
                    for value in (
                        point.x,
                        point.y,
                        point.arc_length,
                        point.speed,
                        point.time,
                        point.curvature,
                    )
                )
            )
    finally:
        if close:
            stream.close()


def _write_json(destination: str, payload: object) -> None:
    stream, close = _open_text_output(destination)
    try:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
    finally:
        if close:
            stream.close()


def _open_text_output(destination: str) -> tuple[TextIO, bool]:
    if destination == "-":
        return sys.stdout, False
    try:
        return Path(destination).open("w", encoding="utf-8", newline=""), True
    except OSError as error:
        raise InvalidInputError(f"cannot open output {destination!r}: {error}") from error


def _read_path_csv(path: str) -> tuple[Point, ...]:
    points: list[Point] = []
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader)
                x_index, y_index = _path_csv_coordinate_indices(header)
                expected_fields = len(header)
                for row in reader:
                    row_number = reader.line_num
                    if len(row) != expected_fields:
                        raise InvalidInputError(
                            f"path CSV row {row_number} has {len(row)} fields; "
                            f"expected {expected_fields}"
                        )
                    try:
                        x = float(row[x_index])
                        y = float(row[y_index])
                    except (TypeError, ValueError) as error:
                        raise InvalidInputError(
                            f"path CSV row {row_number} has invalid coordinates"
                        ) from error
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise InvalidInputError(
                            f"path CSV row {row_number} has non-finite coordinates"
                        )
                    points.append((x, y))
            except StopIteration as error:
                raise InvalidInputError("path CSV header is missing") from error
            except csv.Error as error:
                raise InvalidInputError(
                    f"path CSV is malformed near line {reader.line_num}: {error}"
                ) from error
    except (OSError, UnicodeError) as error:
        raise InvalidInputError(f"cannot read path CSV {path!r}: {error}") from error
    if len(points) < 2:
        raise InvalidInputError("path CSV must contain at least two data rows")
    return tuple(points)


def _path_csv_coordinate_indices(header: list[str]) -> tuple[int, int]:
    if not header:
        raise InvalidInputError("path CSV header must not be blank")
    names = tuple(name.strip() for name in header)
    seen: set[str] = set()
    for column, name in enumerate(names, start=1):
        if not name:
            raise InvalidInputError(
                f"path CSV header column {column} must not be blank"
            )
        if name in seen:
            raise InvalidInputError(
                f"path CSV header contains duplicate column {name!r}"
            )
        seen.add(name)
    if "x" not in seen or "y" not in seen:
        raise InvalidInputError(
            "path CSV header must contain exactly one x and one y column"
        )
    return names.index("x"), names.index("y")


def _select_endpoint(override: Cell | None, fixture: Cell | None, name: str) -> Cell:
    selected = override if override is not None else fixture
    if selected is None:
        raise InvalidInputError(f"{name} is missing; add it to the map or pass --{name}")
    return selected


def _parse_cell(text: str) -> Cell:
    pieces = text.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("cell must use X,Y")
    try:
        return (int(pieces[0]), int(pieces[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError("cell coordinates must be integers") from error


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _positive_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(text: str) -> float:
    value = _finite_float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    raise SystemExit(main())
