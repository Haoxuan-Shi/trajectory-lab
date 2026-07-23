"""Structured path and trajectory validation reports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from ._numeric import (
    checked_finite_sum,
    checked_point_distance,
    exceeds_upper_bound,
    is_finite_real,
    is_real,
)
from .collision import first_path_collision
from .errors import InvalidInputError
from .grid import OccupancyGrid, Point
from .postprocess import polyline_curvatures
from .timing import MotionLimits, TimedPoint, TimedSegment, TrajectoryProfile


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    segment_index: int | None = None
    cell: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.segment_index is not None:
            result["segment_index"] = self.segment_index
        if self.cell is not None:
            result["cell"] = list(self.cell)
        return result


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    issues: tuple[ValidationIssue, ...]
    point_count: int
    total_length: float
    max_curvature: float
    max_speed: float
    max_abs_acceleration: float
    max_lateral_acceleration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": {
                "point_count": self.point_count,
                "total_length": _json_number(self.total_length),
                "max_curvature": _json_number(self.max_curvature),
                "max_speed": _json_number(self.max_speed),
                "max_abs_acceleration": _json_number(self.max_abs_acceleration),
                "max_lateral_acceleration": _json_number(
                    self.max_lateral_acceleration
                ),
            },
        }


def validate_trajectory(
    grid: OccupancyGrid,
    path: Sequence[Point],
    *,
    profile: TrajectoryProfile | None = None,
    max_curvature: float | None = None,
    limits: MotionLimits | None = None,
) -> ValidationReport:
    if max_curvature is not None and (
        not is_finite_real(max_curvature) or max_curvature <= 0.0
    ):
        raise InvalidInputError("max_curvature must be finite and positive")
    points = _materialize(path)
    segment_lengths, expected_arc_lengths = _checked_path_geometry(points)
    issues: list[ValidationIssue] = []
    length = expected_arc_lengths[-1]
    curvatures = polyline_curvatures(points)
    measured_max_curvature = max(curvatures, default=0.0)
    for index, point in enumerate(points):
        cell = grid.world_to_cell(point)
        if not grid.in_bounds(cell):
            issues.append(
                ValidationIssue(
                    "out_of_bounds",
                    f"point {index} lies outside the occupancy grid",
                )
            )
    for index, curvature in enumerate(curvatures):
        if not math.isfinite(curvature):
            issues.append(
                ValidationIssue(
                    "undefined_curvature",
                    f"point {index} forms a degenerate reversal",
                )
            )
    hit = first_path_collision(grid, points)
    if hit is not None:
        issues.append(
            ValidationIssue(
                "collision",
                f"segment {hit.segment_index} touches occupied cell {hit.cell}",
                hit.segment_index,
                hit.cell,
            )
        )
    for index, distance in enumerate(segment_lengths):
        if distance <= 1e-12:
            issues.append(
                ValidationIssue(
                    "zero_length_segment",
                    f"segment {index} has coincident endpoints",
                    index,
                )
            )
    if max_curvature is not None:
        for index, curvature in enumerate(curvatures):
            if exceeds_upper_bound(curvature, max_curvature):
                issues.append(
                    ValidationIssue(
                        "curvature_limit",
                        f"point {index} curvature {curvature:.12g} exceeds "
                        f"limit {max_curvature:.12g}",
                    )
                )

    max_speed = 0.0
    max_abs_acceleration = 0.0
    max_lateral_acceleration = 0.0
    if profile is not None and not _profile_structure_is_valid(profile, limits):
        issues.append(
            ValidationIssue(
                "malformed_profile",
                "timed profile has an invalid type or container structure",
            )
        )
        return ValidationReport(
            False,
            tuple(issues),
            len(points),
            length,
            measured_max_curvature,
            max_speed,
            max_abs_acceleration,
            max_lateral_acceleration,
        )
    if profile is not None:
        active_limits = limits or profile.limits
        if not is_real(profile.total_length):
            issues.append(
                ValidationIssue(
                    "malformed_profile",
                    "timed profile total length must be numeric",
                )
            )
        elif not is_finite_real(profile.total_length):
            issues.append(
                ValidationIssue(
                    "non_finite_profile",
                    "timed profile total length must be finite",
                )
            )
        elif profile.total_length < 0.0:
            issues.append(
                ValidationIssue(
                    "negative_profile",
                    "timed profile total length must be non-negative",
                )
            )
        elif not math.isclose(
            profile.total_length, length, rel_tol=1e-9, abs_tol=1e-12
        ):
            issues.append(
                ValidationIssue(
                    "profile_total_length",
                    "timed profile total length does not match path geometry",
                )
            )
        if not is_real(profile.total_time):
            issues.append(
                ValidationIssue(
                    "malformed_profile",
                    "timed profile total time must be numeric",
                )
            )
            total_time_matches = False
        elif not is_finite_real(profile.total_time):
            issues.append(
                ValidationIssue(
                    "non_finite_profile",
                    "timed profile total time must be finite",
                )
            )
            total_time_matches = False
        elif profile.total_time < 0.0:
            issues.append(
                ValidationIssue(
                    "negative_profile",
                    "timed profile total time must be non-negative",
                )
            )
            total_time_matches = False
        else:
            total_time_matches = True
        if len(profile.points) == len(points) and all(
            is_finite_real(point.time) for point in profile.points
        ):
            total_time_matches = total_time_matches and math.isclose(
                profile.total_time,
                profile.points[-1].time,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        if len(profile.segments) == len(points) - 1 and all(
            is_finite_real(segment.duration) for segment in profile.segments
        ):
            total_time_matches = total_time_matches and math.isclose(
                profile.total_time,
                math.fsum(segment.duration for segment in profile.segments),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        if not total_time_matches:
            issues.append(
                ValidationIssue(
                    "profile_total_time",
                    "timed profile total time does not match points and segments",
                )
            )
        if len(profile.segments) != len(points) - 1:
            issues.append(
                ValidationIssue(
                    "profile_size",
                    "timed profile segment count does not match path geometry",
                )
            )
        if len(profile.points) != len(points):
            issues.append(
                ValidationIssue(
                    "profile_size",
                    "timed profile point count does not match path point count",
                )
            )
        else:
            for index, (point, timed) in enumerate(zip(points, profile.points)):
                point_values = (
                    timed.x,
                    timed.y,
                    timed.arc_length,
                    timed.speed,
                    timed.time,
                    timed.curvature,
                )
                if not all(is_real(value) for value in point_values):
                    issues.append(
                        ValidationIssue(
                            "malformed_profile",
                            f"timed point {index} contains a non-numeric field",
                        )
                    )
                    continue
                if not all(is_finite_real(value) for value in point_values):
                    issues.append(
                        ValidationIssue(
                            "non_finite_profile",
                            f"timed point {index} is non-finite",
                        )
                    )
                    continue
                if any(
                    value < 0.0
                    for value in (
                        timed.arc_length,
                        timed.speed,
                        timed.time,
                        timed.curvature,
                    )
                ):
                    issues.append(
                        ValidationIssue(
                            "negative_profile",
                            f"timed point {index} contains a negative field",
                        )
                    )
                    continue
                if point != (timed.x, timed.y):
                    issues.append(
                        ValidationIssue(
                            "profile_geometry",
                            f"timed point {index} does not match path geometry",
                        )
                    )
                if not math.isclose(
                    timed.arc_length,
                    expected_arc_lengths[index],
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    issues.append(
                        ValidationIssue(
                            "profile_arc_length",
                            f"timed point {index} arc length does not match path geometry",
                        )
                    )
                if not math.isclose(
                    timed.curvature,
                    curvatures[index],
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    issues.append(
                        ValidationIssue(
                            "profile_curvature",
                            f"timed point {index} curvature does not match path geometry",
                        )
                    )
                if timed.speed < 0.0 or exceeds_upper_bound(
                    timed.speed, active_limits.max_speed
                ):
                    issues.append(
                        ValidationIssue(
                            "speed_limit",
                            f"timed point {index} speed is outside configured limits",
                        )
                    )
                lateral = timed.speed * timed.speed * curvatures[index]
                max_lateral_acceleration = max(max_lateral_acceleration, lateral)
                if exceeds_upper_bound(
                    lateral, active_limits.max_lateral_acceleration
                ):
                    issues.append(
                        ValidationIssue(
                            "lateral_acceleration_limit",
                            f"timed point {index} lateral acceleration exceeds limit",
                        )
                    )
                max_speed = max(max_speed, timed.speed)
                if (
                    index
                    and is_finite_real(profile.points[index - 1].time)
                    and timed.time <= profile.points[index - 1].time
                ):
                    issues.append(
                        ValidationIssue(
                            "non_monotonic_time",
                            f"timed point {index} does not advance time",
                        )
                    )
        for index, segment in enumerate(profile.segments[: len(points) - 1]):
            if type(segment.index) is not int or segment.index != index:
                issues.append(
                    ValidationIssue(
                        "segment_index",
                        f"timed segment at position {index} has an invalid index",
                        index,
                    )
                )
            if not isinstance(segment.profile, str) or segment.profile not in {
                "constant-acceleration",
                "cruise",
                "triangular",
                "trapezoidal",
            }:
                issues.append(
                    ValidationIssue(
                        "malformed_profile",
                        f"timed segment {index} has an unsupported profile label",
                        index,
                    )
                )
                continue
            segment_values = (
                segment.length,
                segment.duration,
                segment.start_speed,
                segment.end_speed,
                segment.peak_speed,
                segment.max_abs_acceleration,
            )
            if not all(is_real(value) for value in segment_values):
                issues.append(
                    ValidationIssue(
                        "malformed_profile",
                        f"timed segment {index} contains a non-numeric field",
                        index,
                    )
                )
                continue
            if not all(is_finite_real(value) for value in segment_values):
                issues.append(
                    ValidationIssue(
                        "non_finite_profile",
                        f"timed segment {index} is non-finite",
                        index,
                    )
                )
                continue
            if any(value < 0.0 for value in segment_values):
                issues.append(
                    ValidationIssue(
                        "negative_profile",
                        f"timed segment {index} contains a negative field",
                        index,
                    )
                )
                continue
            expected_length = segment_lengths[index]
            if not math.isclose(
                segment.length, expected_length, rel_tol=1e-9, abs_tol=1e-12
            ):
                issues.append(
                    ValidationIssue(
                        "segment_length",
                        f"timed segment {index} length does not match path geometry",
                        index,
                    )
                )
            if len(profile.points) == len(points) and all(
                is_finite_real(value)
                for value in (
                    segment.start_speed,
                    segment.end_speed,
                    profile.points[index].speed,
                    profile.points[index + 1].speed,
                )
            ) and (
                not math.isclose(
                    segment.start_speed,
                    profile.points[index].speed,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    segment.end_speed,
                    profile.points[index + 1].speed,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                issues.append(
                    ValidationIssue(
                        "segment_speed",
                        f"timed segment {index} endpoint speeds do not match timed points",
                        index,
                    )
                )
            if len(profile.points) == len(points) and all(
                is_finite_real(value)
                for value in (
                    segment.duration,
                    profile.points[index].time,
                    profile.points[index + 1].time,
                )
            ):
                expected_duration = (
                    profile.points[index + 1].time - profile.points[index].time
                )
                if not math.isclose(
                    segment.duration,
                    expected_duration,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    issues.append(
                        ValidationIssue(
                            "segment_duration",
                            f"timed segment {index} duration does not match its point times",
                            index,
                        )
                    )
            if segment.duration <= 0.0:
                issues.append(
                    ValidationIssue(
                        "invalid_duration",
                        f"timed segment {index} has invalid duration",
                        index,
                    )
                )
            inferred_acceleration = _infer_segment_acceleration(segment)
            if inferred_acceleration is None:
                issues.append(
                    ValidationIssue(
                        "segment_kinematics",
                        f"timed segment {index} is inconsistent with its motion profile",
                        index,
                    )
                )
            else:
                max_speed = max(max_speed, segment.peak_speed)
                max_abs_acceleration = max(
                    max_abs_acceleration, inferred_acceleration
                )
            if exceeds_upper_bound(
                segment.peak_speed, active_limits.max_speed
            ):
                issues.append(
                    ValidationIssue(
                        "speed_limit",
                        f"timed segment {index} peak speed exceeds limit",
                        index,
                    )
                )
            if (
                inferred_acceleration is not None
                and exceeds_upper_bound(
                    inferred_acceleration,
                    active_limits.max_acceleration,
                )
            ):
                issues.append(
                    ValidationIssue(
                        "acceleration_limit",
                        f"timed segment {index} acceleration exceeds limit",
                        index,
                    )
                )

    return ValidationReport(
        not issues,
        tuple(issues),
        len(points),
        length,
        measured_max_curvature,
        max_speed,
        max_abs_acceleration,
        max_lateral_acceleration,
    )


def _materialize(path: Sequence[Point]) -> tuple[Point, ...]:
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise InvalidInputError("path must be a sequence")
    points = tuple(path)
    if len(points) < 2:
        raise InvalidInputError("path must contain at least two points")
    for index, point in enumerate(points):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or not all(is_finite_real(value) for value in point)
        ):
            raise InvalidInputError(f"path[{index}] must contain two finite numbers")
    return points


def _checked_path_geometry(
    points: tuple[Point, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    distances: list[float] = []
    arc_lengths = [0.0]
    for index, (first, second) in enumerate(zip(points, points[1:])):
        message = f"path segment {index} geometry exceeds the finite numeric range"
        distance = checked_point_distance(first, second, message=message)
        distances.append(distance)
        arc_lengths.append(
            checked_finite_sum(arc_lengths[-1], distance, message=message)
        )
    return tuple(distances), tuple(arc_lengths)


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _profile_structure_is_valid(
    profile: object, limits: MotionLimits | None
) -> bool:
    return (
        isinstance(profile, TrajectoryProfile)
        and isinstance(profile.points, tuple)
        and isinstance(profile.segments, tuple)
        and all(isinstance(point, TimedPoint) for point in profile.points)
        and all(isinstance(segment, TimedSegment) for segment in profile.segments)
        and isinstance(profile.limits, MotionLimits)
        and (limits is None or isinstance(limits, MotionLimits))
    )


def _infer_segment_acceleration(segment: TimedSegment) -> float | None:
    """Independently recover acceleration from the segment's bound kinematics."""

    if segment.length <= 0.0 or segment.duration <= 0.0:
        return None
    endpoint_peak = max(segment.start_speed, segment.end_speed)
    if segment.peak_speed + 1e-12 < endpoint_peak:
        return None

    if segment.profile in {"constant-acceleration", "cruise"}:
        inferred_acceleration = (
            abs(segment.end_speed - segment.start_speed) / segment.duration
        )
        expected_profile = (
            "constant-acceleration" if inferred_acceleration > 1e-12 else "cruise"
        )
        expected_length = (
            0.5
            * (segment.start_speed + segment.end_speed)
            * segment.duration
        )
        if (
            segment.profile != expected_profile
            or not math.isclose(
                segment.peak_speed,
                endpoint_peak,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isclose(
                segment.length,
                expected_length,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or not math.isclose(
                segment.max_abs_acceleration,
                inferred_acceleration,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            return None
        return inferred_acceleration

    if segment.profile not in {"triangular", "trapezoidal"}:
        return None
    if segment.peak_speed <= 0.0:
        return None
    speed_change_energy = (
        (segment.peak_speed - segment.start_speed) ** 2
        + (segment.peak_speed - segment.end_speed) ** 2
    )
    distance_deficit = segment.peak_speed * segment.duration - segment.length
    if speed_change_energy <= 1e-15 or distance_deficit <= 1e-15:
        return None
    inferred_acceleration = speed_change_energy / (2.0 * distance_deficit)
    acceleration_time = (
        2.0 * segment.peak_speed
        - segment.start_speed
        - segment.end_speed
    ) / inferred_acceleration
    cruise_time = segment.duration - acceleration_time
    if cruise_time < -1e-9:
        return None
    expected_profile = "triangular" if abs(cruise_time) <= 1e-9 else "trapezoidal"
    if (
        segment.profile != expected_profile
        or not math.isclose(
            segment.max_abs_acceleration,
            inferred_acceleration,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        return None
    return inferred_acceleration
