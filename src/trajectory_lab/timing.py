"""Velocity and time parameterization under longitudinal/lateral limits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ._numeric import (
    checked_finite_difference,
    checked_finite_division,
    checked_finite_product,
    checked_finite_sum,
    checked_point_distance,
    exceeds_upper_bound,
    is_finite_real,
)
from .errors import InvalidInputError
from .grid import Point
from .postprocess import polyline_curvatures


_TIMING_ARITHMETIC_ERROR = (
    "derived trajectory timing is not representable as a finite number"
)


@dataclass(frozen=True, slots=True)
class MotionLimits:
    max_speed: float = 5.0
    max_acceleration: float = 2.0
    max_lateral_acceleration: float = 2.5

    def __post_init__(self) -> None:
        for name, value in (
            ("max_speed", self.max_speed),
            ("max_acceleration", self.max_acceleration),
            ("max_lateral_acceleration", self.max_lateral_acceleration),
        ):
            if not is_finite_real(value) or value <= 0.0:
                raise InvalidInputError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class TimedPoint:
    x: float
    y: float
    arc_length: float
    speed: float
    time: float
    curvature: float


@dataclass(frozen=True, slots=True)
class TimedSegment:
    index: int
    length: float
    duration: float
    start_speed: float
    end_speed: float
    peak_speed: float
    max_abs_acceleration: float
    profile: str


@dataclass(frozen=True, slots=True)
class TrajectoryProfile:
    points: tuple[TimedPoint, ...]
    segments: tuple[TimedSegment, ...]
    total_length: float
    total_time: float
    limits: MotionLimits


def parameterize_velocity(
    path: Sequence[Point],
    limits: MotionLimits = MotionLimits(),
    *,
    start_speed: float = 0.0,
    end_speed: float = 0.0,
) -> TrajectoryProfile:
    """Apply curvature limits and deterministic forward/backward speed passes."""

    points = _validate_path(path)
    _validate_boundary_speed(start_speed, limits, "start_speed")
    _validate_boundary_speed(end_speed, limits, "end_speed")
    distances = [
        checked_point_distance(
            first, second, message=_TIMING_ARITHMETIC_ERROR
        )
        for first, second in zip(points, points[1:])
    ]
    if any(distance <= 1e-12 for distance in distances):
        raise InvalidInputError("adjacent path points must be distinct")
    try:
        curvatures = polyline_curvatures(points)
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(_TIMING_ARITHMETIC_ERROR) from error
    if any(not math.isfinite(curvature) for curvature in curvatures):
        raise InvalidInputError(
            "path contains a degenerate reversal with undefined curvature"
        )
    speed_limits = []
    for curvature in curvatures:
        if curvature == 0.0:
            local_limit = limits.max_speed
        else:
            local_limit = _checked_lateral_speed_limit(
                curvature, limits.max_lateral_acceleration
            )
        speed_limits.append(min(limits.max_speed, local_limit))
    if exceeds_upper_bound(start_speed, speed_limits[0]):
        raise InvalidInputError("start_speed exceeds the local lateral speed limit")
    if exceeds_upper_bound(end_speed, speed_limits[-1]):
        raise InvalidInputError("end_speed exceeds the local lateral speed limit")

    speeds = speed_limits[:]
    speeds[0] = start_speed
    for index, distance in enumerate(distances, start=1):
        reachable = _reachable_speed(
            speeds[index - 1], distance, limits.max_acceleration
        )
        speeds[index] = min(speeds[index], reachable)
    if (
        end_speed > speeds[-1]
        and exceeds_upper_bound(
            _constant_segment_acceleration(
                speeds[-2], end_speed, distances[-1]
            ),
            limits.max_acceleration,
        )
    ):
        raise InvalidInputError(
            "path is too short to reach end_speed from start_speed"
        )
    speeds[-1] = end_speed
    for index in range(len(speeds) - 2, -1, -1):
        reachable = _reachable_speed(
            speeds[index + 1], distances[index], limits.max_acceleration
        )
        speeds[index] = min(speeds[index], reachable)
    if (
        start_speed > speeds[0]
        and exceeds_upper_bound(
            _constant_segment_acceleration(
                start_speed, speeds[1], distances[0]
            ),
            limits.max_acceleration,
        )
    ):
        raise InvalidInputError(
            "path is too short to decelerate from start_speed"
        )
    speeds[0] = start_speed

    timed_points: list[TimedPoint] = [
        TimedPoint(points[0][0], points[0][1], 0.0, speeds[0], 0.0, curvatures[0])
    ]
    timed_segments: list[TimedSegment] = []
    elapsed = 0.0
    arc_length = 0.0
    for index, distance in enumerate(distances):
        segment = _time_segment(
            index, distance, speeds[index], speeds[index + 1], limits
        )
        if exceeds_upper_bound(
            segment.max_abs_acceleration, limits.max_acceleration
        ):
            raise InvalidInputError(
                "derived trajectory acceleration exceeds max_acceleration"
            )
        timed_segments.append(segment)
        elapsed = checked_finite_sum(
            elapsed, segment.duration, message=_TIMING_ARITHMETIC_ERROR
        )
        arc_length = checked_finite_sum(
            arc_length, distance, message=_TIMING_ARITHMETIC_ERROR
        )
        timed_points.append(
            TimedPoint(
                points[index + 1][0],
                points[index + 1][1],
                arc_length,
                speeds[index + 1],
                elapsed,
                curvatures[index + 1],
            )
        )
    return TrajectoryProfile(
        tuple(timed_points),
        tuple(timed_segments),
        arc_length,
        elapsed,
        limits,
    )


def _checked_lateral_speed_limit(
    curvature: float, max_lateral_acceleration: float
) -> float:
    lateral_ratio = checked_finite_division(
        max_lateral_acceleration,
        curvature,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    speed_limit = _checked_sqrt(lateral_ratio)
    if (
        _checked_lateral_acceleration(speed_limit, curvature)
        <= max_lateral_acceleration
    ):
        return speed_limit

    conservative_ratio = math.nextafter(lateral_ratio, 0.0)
    speed_limit = _checked_sqrt(conservative_ratio)
    if (
        _checked_lateral_acceleration(speed_limit, curvature)
        <= max_lateral_acceleration
    ):
        return speed_limit

    speed_limit = math.nextafter(speed_limit, 0.0)
    if (
        _checked_lateral_acceleration(speed_limit, curvature)
        > max_lateral_acceleration
    ):
        raise InvalidInputError(_TIMING_ARITHMETIC_ERROR)
    return speed_limit


def _checked_lateral_acceleration(speed: float, curvature: float) -> float:
    speed_squared = checked_finite_product(
        speed, speed, message=_TIMING_ARITHMETIC_ERROR
    )
    return checked_finite_product(
        speed_squared, curvature, message=_TIMING_ARITHMETIC_ERROR
    )


def _time_segment(
    index: int,
    length: float,
    start_speed: float,
    end_speed: float,
    limits: MotionLimits,
) -> TimedSegment:
    combined_speed = checked_finite_sum(
        start_speed, end_speed, message=_TIMING_ARITHMETIC_ERROR
    )
    if combined_speed > 1e-12:
        duration = checked_finite_division(
            checked_finite_product(
                2.0, length, message=_TIMING_ARITHMETIC_ERROR
            ),
            combined_speed,
            message=_TIMING_ARITHMETIC_ERROR,
        )
        speed_change = abs(
            checked_finite_difference(
                end_speed, start_speed, message=_TIMING_ARITHMETIC_ERROR
            )
        )
        acceleration = checked_finite_division(
            speed_change, duration, message=_TIMING_ARITHMETIC_ERROR
        )
        return TimedSegment(
            index,
            length,
            duration,
            start_speed,
            end_speed,
            max(start_speed, end_speed),
            acceleration,
            "constant-acceleration" if acceleration > 1e-12 else "cruise",
        )

    peak = min(
        limits.max_speed,
        _checked_sqrt(
            checked_finite_product(
                limits.max_acceleration,
                length,
                message=_TIMING_ARITHMETIC_ERROR,
            )
        ),
    )
    if peak <= 0.0:
        raise InvalidInputError("cannot time a positive-length zero-speed segment")
    acceleration_distance = checked_finite_division(
        checked_finite_product(peak, peak, message=_TIMING_ARITHMETIC_ERROR),
        limits.max_acceleration,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    cruise_distance = max(
        0.0,
        checked_finite_difference(
            length, acceleration_distance, message=_TIMING_ARITHMETIC_ERROR
        ),
    )
    acceleration_duration = checked_finite_division(
        checked_finite_product(2.0, peak, message=_TIMING_ARITHMETIC_ERROR),
        limits.max_acceleration,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    cruise_duration = checked_finite_division(
        cruise_distance, peak, message=_TIMING_ARITHMETIC_ERROR
    )
    duration = checked_finite_sum(
        acceleration_duration,
        cruise_duration,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    return TimedSegment(
        index,
        length,
        duration,
        start_speed,
        end_speed,
        peak,
        limits.max_acceleration,
        "triangular" if cruise_distance <= 1e-12 else "trapezoidal",
    )


def _constant_segment_acceleration(
    start_speed: float,
    end_speed: float,
    length: float,
) -> float:
    """Recover the acceleration represented by two endpoint speeds."""

    combined_speed = checked_finite_sum(
        start_speed, end_speed, message=_TIMING_ARITHMETIC_ERROR
    )
    if combined_speed == 0.0:
        return 0.0
    duration = checked_finite_division(
        checked_finite_product(
            2.0, length, message=_TIMING_ARITHMETIC_ERROR
        ),
        combined_speed,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    speed_change = abs(
        checked_finite_difference(
            end_speed, start_speed, message=_TIMING_ARITHMETIC_ERROR
        )
    )
    return checked_finite_division(
        speed_change, duration, message=_TIMING_ARITHMETIC_ERROR
    )


def _validate_boundary_speed(speed: float, limits: MotionLimits, name: str) -> None:
    if (
        not is_finite_real(speed)
        or speed < 0.0
        or speed > limits.max_speed
    ):
        raise InvalidInputError(
            f"{name} must be finite and in [0, {limits.max_speed}]"
        )


def _reachable_speed(speed: float, distance: float, acceleration: float) -> float:
    speed_squared = checked_finite_product(
        speed, speed, message=_TIMING_ARITHMETIC_ERROR
    )
    acceleration_term = checked_finite_product(
        checked_finite_product(
            2.0, acceleration, message=_TIMING_ARITHMETIC_ERROR
        ),
        distance,
        message=_TIMING_ARITHMETIC_ERROR,
    )
    reachable = _checked_sqrt(
        max(
            0.0,
            checked_finite_sum(
                speed_squared,
                acceleration_term,
                message=_TIMING_ARITHMETIC_ERROR,
            ),
        )
    )
    if reachable <= speed:
        return speed
    if not exceeds_upper_bound(
        _constant_segment_acceleration(speed, reachable, distance),
        acceleration,
    ):
        return reachable

    conservative = math.nextafter(reachable, speed)
    if conservative <= speed:
        return speed
    if exceeds_upper_bound(
        _constant_segment_acceleration(speed, conservative, distance),
        acceleration,
    ):
        return speed
    return conservative


def _checked_sqrt(value: float) -> float:
    if not is_finite_real(value) or value < 0.0:
        raise InvalidInputError(_TIMING_ARITHMETIC_ERROR)
    try:
        result = math.sqrt(value)
    except (ArithmeticError, TypeError, ValueError) as error:
        raise InvalidInputError(_TIMING_ARITHMETIC_ERROR) from error
    if not math.isfinite(result):
        raise InvalidInputError(_TIMING_ARITHMETIC_ERROR)
    return result


def _validate_path(path: Sequence[Point]) -> tuple[Point, ...]:
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
